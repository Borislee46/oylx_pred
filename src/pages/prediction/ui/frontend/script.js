let _streamlitEventSeq = 0;
function sendToStreamlit(value) {
  const payload = {
    ...value,
    event_id: _sessionId + ":" + ++_streamlitEventSeq,
  };
  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type: "streamlit:setComponentValue",
      value: payload,
      dataType: "json",
    },
    "*",
  );
}

function setFrameHeight(height) {
  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type: "streamlit:setFrameHeight",
      height: height || document.documentElement.scrollHeight,
    },
    "*",
  );
}

function setComponentReady() {
  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type: "streamlit:componentReady",
      apiVersion: 1,
    },
    "*",
  );
}

const textarea = document.getElementById("textarea");
const backdrop = document.getElementById("backdrop");
const statusEl = document.getElementById("status");
const acceptChip = document.getElementById("acceptChip");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeLabel = document.getElementById("analyzeLabel");

let ghostToken = "";
let apiEnabled = false;
let apiBaseUrl = "https://api.deepseek.com/beta";
let apiModel = "deepseek-v4-flash";
let currentSuggestion = "";
let _suggestionForText = "";
let _suggestionQueue = [];
let debounceTimer = null;
let isComposing = false;
let fetchController = null;
let lastRequest = "";

const _sessionId = Math.random().toString(36).slice(2, 10);
const _counters = {
  fetch_attempt: 0,
  fetch_ok: 0,
  fetch_fail: 0,
  fetch_retry: 0,
  cache_hit: 0,
  cache_set: 0,
  suggestion_shown: 0,
  suggestion_accepted: 0,
  suggestion_dismissed: 0,
  rate_limited: 0,
  dedup_blocked: 0,
  rule_hit: 0,
  rule_miss: 0,
};
const _events = [];
let _phaseStart = Date.now();

function _log(action, detail) {
  const entry = { a: action, t: Date.now() - _phaseStart, d: detail || "" };
  _events.push(entry);
  if (_events.length > 20) _events.shift();
  if (action in _counters) _counters[action]++;
}

function _buildTelemetry() {
  return {
    session: _sessionId,
    duration_s: ((Date.now() - _phaseStart) / 1000).toFixed(0),
    counters: { ..._counters },
    recent: _events.slice(-10),
  };
}

textarea.addEventListener("compositionstart", () => {
  isComposing = true;
});
textarea.addEventListener("compositionend", () => {
  isComposing = false;
  scheduleFetch();
});

textarea.addEventListener("input", () => {
  updateBackdrop();
  recordKeystroke();
  if (!isComposing) {
    scheduleFetch();
  }
});

textarea.addEventListener("click", scheduleFetch);
textarea.addEventListener("keyup", (e) => {
  if (
    ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(
      e.key,
    )
  ) {
    scheduleFetch();
  }
});

textarea.addEventListener("keydown", (e) => {
  if ((e.key === "Tab" || e.key === "ArrowRight") && currentSuggestion) {
    e.preventDefault();
    acceptSuggestion();
    return;
  }
  if (e.key === "Escape" && currentSuggestion) {
    e.preventDefault();
    dismissSuggestion();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    triggerAnalyze();
    return;
  }
});

let _lastSyncedText = "";
let _leadInBusy = false;
let _throttleHint = "none";
let _analyzeLocked = false;

let _suppressBlurUntil = 0;
textarea.addEventListener("blur", () => {
  if (Date.now() < _suppressBlurUntil) return;
  if (textarea.value === _lastSyncedText) return;
  _lastSyncedText = textarea.value;
  sendToStreamlit({
    text: textarea.value,
    action: "blur",
    telemetry: _buildTelemetry(),
  });
});

new ResizeObserver(() => {
  backdrop.style.height = textarea.offsetHeight + "px";
  backdrop.style.width = textarea.offsetWidth + "px";
}).observe(textarea);

let keystrokeTimestamps = [];

const TRIGGER_END_RE =
  /(GPA|雅思|托福|GRE|GMAT|科研|实习|论文|获奖|目标|申请|一段|两段|二段|三段)\s*$/;
const SENTENCE_END_RE = /[。！？\n]\s*$/;
const KNOWN_FIELDS_RE =
  /(院校|学校|大学|专业|GPA|均分|雅思|托福|GRE|实习|科研|论文|获奖|目标)/g;

function typingSpeed() {
  const now = keystrokeTimestamps;
  if (now.length < 2) return 0;
  const d = (now[now.length - 1] - now[0]) / 1000;
  return d > 0 ? (now.length - 1) / d : 999;
}

function recordKeystroke() {
  const now = Date.now();
  keystrokeTimestamps.push(now);
  if (keystrokeTimestamps.length > 6) keystrokeTimestamps.shift();
}

function adaptiveDebounce(text) {
  if (SENTENCE_END_RE.test(text)) return -1;

  if (TRIGGER_END_RE.test(text)) return 120;

  let delay;
  const speed = typingSpeed();
  if (speed > 3.5) delay = 150;
  else if (speed > 1.5) delay = 280;
  else delay = 450;

  if (_throttleHint === "high") delay = Math.round(delay * 1.5);
  else if (_throttleHint === "normal") delay = Math.round(delay * 1.2);

  return delay;
}

function scheduleFetch() {
  clearTimeout(debounceTimer);
  const text = textarea.value;

  if (!text || text.length < 2 || textarea.selectionStart !== text.length) {
    dismissSuggestion();
    return;
  }

  const delay = adaptiveDebounce(text);
  if (delay < 0) {
    dismissSuggestion();
    return;
  }

  debounceTimer = setTimeout(() => doFetch(text), delay);
}

const CACHE_PREFIX = "ghost_cache:";
const CACHE_MAX = 50;
const CACHE_TTL = 10 * 60 * 1000;

function _simpleHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h).toString(36);
}

function cacheGet(prefixText) {
  try {
    const raw = localStorage.getItem(CACHE_PREFIX + _simpleHash(prefixText));
    if (!raw) return null;
    const entry = JSON.parse(raw);
    if (Date.now() - entry.ts > CACHE_TTL) {
      localStorage.removeItem(CACHE_PREFIX + _simpleHash(prefixText));
      return null;
    }
    return entry.completion;
  } catch (e) {
    return null;
  }
}

function cacheSet(prefixText, completion) {
  try {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(CACHE_PREFIX)) keys.push(k);
    }
    if (keys.length >= CACHE_MAX) {
      const sorted = keys
        .map((k) => {
          try {
            return { k, ts: JSON.parse(localStorage.getItem(k)).ts };
          } catch (e) {
            return { k, ts: 0 };
          }
        })
        .sort((a, b) => a.ts - b.ts);
      for (let i = 0; i < Math.min(10, sorted.length); i++) {
        localStorage.removeItem(sorted[i].k);
      }
    }
    localStorage.setItem(
      CACHE_PREFIX + _simpleHash(prefixText),
      JSON.stringify({ completion, ts: Date.now() }),
    );
  } catch (e) {}
}

let requestTimestamps = [];
let RATE_MAX = 30;
let RATE_WINDOW = 60000;
let RATE_COOLDOWN = 15000;
let rateLimitedUntil = 0;

function checkRateLimit() {
  const now = Date.now();
  if (now < rateLimitedUntil) return false;
  const cutoff = now - RATE_WINDOW;
  requestTimestamps = requestTimestamps.filter((t) => t > cutoff);
  if (requestTimestamps.length >= RATE_MAX) {
    _log("rate_limited", "cooldown " + RATE_COOLDOWN / 1000 + "s");
    rateLimitedUntil = now + RATE_COOLDOWN;
    statusEl.textContent = "太快了，休息 15 秒...";
    statusEl.classList.add("rate-limited");
    setTimeout(() => {
      statusEl.classList.remove("rate-limited");
      statusEl.textContent = "";
    }, RATE_COOLDOWN);
    dismissSuggestion();
    return false;
  }
  requestTimestamps.push(now);
  return true;
}

let lastSuggestions = [];

function isDuplicateSuggestion(completion) {
  const clean = completion.trim();
  if (lastSuggestions.includes(clean)) return true;
  lastSuggestions.push(clean);
  if (lastSuggestions.length > 5) lastSuggestions.shift();
  return false;
}

const DOMAIN_SIGNALS_RE =
  /(大学|学院|学校|院校|专业|系|工程|科学|管理|经济|金融|计算机|数据|电子|机械|材料|化学|生物|医学|法律|教育|传媒|设计|艺术|建筑|数学|统计|物理|环境|GPA|均分|绩点|雅思|托福|GRE|GMAT|科研|实习|论文|获奖|项目|经历|实验室|导师|课程|排名|申请|录取|目标|冲刺|保底|一段|两段|二段|三段|一篇|两篇|港|香港|新加坡|NUS|NTU|HKU|CUHK|HKUST|CityU|PolyU)/;

const NONSENSE_RE = /^[\s，,、。！？：；:;\d]+$/;
const AGE_RE = /岁|年龄|出生|生日|几岁|多大/;
const CONTACT_RE = /电话|手机|微信|邮箱|QQ|地址|住|省|市/;
const POLITICAL_RE = /党|政府|政治|习|主席|领导人/;

function isQualityCompletion(completion) {
  if (!completion) return false;
  if (NONSENSE_RE.test(completion)) return false;
  if (AGE_RE.test(completion)) return false;
  if (CONTACT_RE.test(completion)) return false;
  if (POLITICAL_RE.test(completion)) return false;
  if (_FORBIDDEN_REGIONS_RE.test(completion)) return false;

  const meaningful = completion.replace(/[，,、\s\.。！？：:；;]/g, "");
  if (meaningful.length < 2) return false;

  const text = textarea.value;
  if (completion.length >= 4 && text.includes(completion)) return false;

  if (completion.length >= 3 && !DOMAIN_SIGNALS_RE.test(completion)) {
    if (!/^\s*\d+\.?\d*\s*$/.test(completion)) {
      return false;
    }
  }

  return true;
}

const POSTFIX_RULES = {
  GPA: [" 3.5", " 3.2", " 3.0", " 3.8", " 3.3", " 3.7", " 2.8"],
  均分: [" 85", " 80", " 88", " 82", " 78", " 90", " 83"],
  雅思: [" 7.0", " 6.5", " 6.0", " 7.5", " 5.5"],
  托福: [" 100", " 105", " 90", " 95", " 110", " 85"],
  GRE: [" 320", " 325", " 330", " 315"],
  GMAT: [" 700", " 680", " 720", " 650"],
  一段: [" 科研", " 实习"],
  两段: [" 科研", " 实习"],
  二段: [" 科研", " 实习"],
  一篇: [" 论文"],
};

function tryRules(text) {
  for (let len = 4; len >= 1; len--) {
    if (text.length < len) continue;
    const candidates = POSTFIX_RULES[text.slice(-len)];
    if (candidates && candidates.length > 0) return candidates[0];
  }
  return null;
}

function cacheGetFuzzy(prefixText) {
  const exact = cacheGet(prefixText);
  if (exact) return exact;
  let i = prefixText.length - 1;
  while (i >= 6) {
    if (/[，,、\s]/.test(prefixText[i])) {
      const shorter = prefixText.slice(0, i);
      const cached = cacheGet(shorter);
      if (cached) {
        const beyond = prefixText.slice(i).trimStart();
        if (cached.startsWith(beyond)) return cached.slice(beyond.length);
        return null;
      }
    }
    i--;
  }
  return null;
}

async function doFetch(text) {
  // 当前文本已有对应建议时，不再重复请求/重复出建议
  if (currentSuggestion && _suggestionForText === text) return;

  const cached = cacheGetFuzzy(text);
  if (cached && isQualityCompletion(cached) && !isDuplicateSuggestion(cached)) {
    _log("cache_hit", text.slice(-20));
    if (textarea.value === text) {
      _enqueueSuggestion(cached);
      updateBackdrop();
    }
    return;
  }

  if (!currentSuggestion) {
    const ruled = tryRules(text);
    if (ruled && isQualityCompletion(ruled)) {
      _log("rule_hit", text.slice(-20));
      if (textarea.value === text) {
        _enqueueSuggestion(ruled);
        updateBackdrop();
      }
      return;
    }
    _log("rule_miss", "…");
  }

  if (!checkRateLimit()) return;

  // 未配置专用 key 时停用 LLM 请求（规则补全仍可用），避免空凭据请求与错误遥测
  if (!apiEnabled) return;

  if (fetchController) fetchController.abort();
  fetchController = new AbortController();
  lastRequest = text;

  const prefixLen =
    text.length < 4
      ? text.length
      : Math.min(6, Math.max(3, Math.floor(text.length / 6)));
  const prefix = text.slice(-prefixLen);
  const mt = text.length < 15 ? 16 : text.length < 30 ? 24 : 32;
  const present = (text.match(KNOWN_FIELDS_RE) || []).length;
  const guide =
    present >= 4
      ? "，补充描述其他背景信息"
      : "，推断可能缺少的背景信息（院校、专业、GPA、语言成绩、科研、实习、论文、获奖、目标院校等）";

  statusEl.classList.add("loading");

  let raw = null;
  let fetchMs = 0;
  for (let attempt = 0; attempt < 2; attempt++) {
    _log("fetch_attempt", "try" + (attempt + 1));
    const t0 = Date.now();
    const controller = fetchController;
    const signal = controller.signal;
    try {
      const timeoutId = setTimeout(() => signal.abort(), 8000);

      const resp = await fetch(apiBaseUrl + "/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + ghostToken,
        },
        body: JSON.stringify({
          model: apiModel,
          messages: [
            {
              role: "system",
              content:
                "你是留学申请背景填写助手。用户正在逐步输入学生背景信息。你需要预测用户接下来会输入什么文字。" +
                "核心规则：只输出预测文字，不要解释。补全应自然衔接用户已输入的文字。无法预测时输出空。" +
                "系统仅支持以下地区院校：" +
                (_allowedRegions.join("、") || "香港、新加坡、澳门、马来西亚") +
                "。" +
                "允许的院校：" +
                (_allowedUniversities.slice(0, 25).join("、") ||
                  "香港大学、香港中文大学...") +
                "。" +
                "严禁提及任何非以上地区的院校（如美国、英国、澳洲、加拿大等）。" +
                "补全内容可包括：院校名（仅限以上列表）、专业名、GPA/均分、雅思/托福/GRE/GMAT分数、科研/实习/论文/获奖经历、目标方向。" +
                "严禁输出年龄、生日、联系方式、家庭信息、政治内容。" +
                guide +
                "。",
            },
            { role: "user", content: text },
            { role: "assistant", content: prefix, prefix: true },
          ],
          max_tokens: mt,
          temperature: 0.2,
          stop: ["\n", "。", "；", "！"],
        }),
        signal,
      });

      clearTimeout(timeoutId);
      fetchMs = Date.now() - t0;

      if (!resp.ok) {
        if (resp.status >= 500 && attempt === 0) {
          _log("fetch_retry", "HTTP " + resp.status);
          continue;
        }
        _log("fetch_fail", "HTTP " + resp.status);
        throw new Error("HTTP " + resp.status);
      }

      const data = await resp.json();
      raw = (data.choices?.[0]?.message?.content || "").trim();
      _log("fetch_ok", fetchMs + "ms " + raw.slice(0, 15));
      break;
    } catch (err) {
      fetchMs = Date.now() - t0;
      if (signal.aborted) {
        if (lastRequest !== text) break; // 已被更新的请求取代，直接放弃
        if (attempt === 0) {
          _log("fetch_retry", "timeout " + fetchMs + "ms");
          fetchController = new AbortController();
          await new Promise((r) => setTimeout(r, 300));
          continue;
        }
        _log("fetch_fail", "timeout " + fetchMs + "ms (final)");
        break;
      }
      if (attempt === 0 && err.message.includes("fetch")) {
        _log("fetch_retry", "network");
        await new Promise((r) => setTimeout(r, 600));
        continue;
      }
      _log("fetch_fail", err.message.slice(0, 40));
      break;
    }
  }

  if (lastRequest !== text) return; // 结果已过期，丢弃

  statusEl.classList.remove("loading");
  lastRequest = "";
  fetchController = null;

  if (raw && isQualityCompletion(raw)) {
    if (isDuplicateSuggestion(raw)) {
      _log("dedup_blocked", raw.slice(0, 15));
    } else {
      _log("cache_set", text.slice(-20) + " → " + raw.slice(0, 15));
      cacheSet(text, raw);
      if (textarea.value === text) {
        _enqueueSuggestion(raw);
        updateBackdrop();
      }
    }
  } else if (textarea.value === text && !raw) {
    const fallback = cacheGet(text);
    if (fallback && isQualityCompletion(fallback)) {
      _log("cache_hit", "fallback " + text.slice(-20));
      _enqueueSuggestion(fallback);
      updateBackdrop();
    } else {
      dismissSuggestion();
    }
  } else if (textarea.value === text) {
    dismissSuggestion();
  }
}

function _enqueueSuggestion(suggestion) {
  if (!suggestion) {
    _suggestionQueue = [];
    currentSuggestion = "";
    _suggestionForText = "";
    return;
  }
  _suggestionForText = textarea.value;
  const parts = suggestion.split(/(?= )/);
  _suggestionQueue = parts.filter((p) => p.trim());
  currentSuggestion = _suggestionQueue.length > 0 ? _suggestionQueue[0] : "";
}
function _advanceSegment() {
  _suggestionQueue.shift();
  currentSuggestion = _suggestionQueue.length > 0 ? _suggestionQueue[0] : "";
}

function updateBackdrop() {
  const text = textarea.value;
  const suggestion = currentSuggestion;

  const esc = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  if (suggestion) {
    if (!backdrop.querySelector(".ghost-suggestion"))
      _log("suggestion_shown", suggestion.slice(0, 20));
    backdrop.innerHTML =
      esc(text) +
      '<span class="ghost-suggestion">' +
      esc(suggestion) +
      "</span>";
    acceptChip.classList.add("visible");
  } else {
    backdrop.textContent = text;
    acceptChip.classList.remove("visible");
  }

  backdrop.scrollTop = textarea.scrollTop;
  backdrop.scrollLeft = textarea.scrollLeft;
  setFrameHeight();
}

textarea.addEventListener("scroll", () => {
  backdrop.scrollTop = textarea.scrollTop;
  backdrop.scrollLeft = textarea.scrollLeft;
});

function acceptSuggestion() {
  if (!currentSuggestion) return;
  _log("suggestion_accepted", currentSuggestion.slice(0, 20));
  textarea.value += currentSuggestion;
  _advanceSegment();
  updateBackdrop();
  _lastSyncedText = textarea.value;
  _suggestionForText = "";
  sendToStreamlit({
    text: textarea.value,
    action: "accept",
    telemetry: _buildTelemetry(),
  });
  textarea.focus();
  if (!currentSuggestion) scheduleFetch();
}

function dismissSuggestion() {
  if (currentSuggestion) {
    _log("suggestion_dismissed", currentSuggestion.slice(0, 20));
  }
  _suggestionQueue = [];
  currentSuggestion = "";
  _suggestionForText = "";
  updateBackdrop();
  statusEl.classList.remove("loading");
}

let _hasInit = false;
let _appliedTextRevision = null;
let _allowedUniversities = [];
let _allowedRegions = [];
let _FORBIDDEN_REGIONS_RE =
  /(美国|英国|澳洲|澳大利亚|加拿大|日本|韩国|欧洲|德国|法国|意大利|瑞士|荷兰|瑞典|挪威|丹麦|芬兰|US|UK|USA|Australia|Canada|Europe)/i;

function init(cfg) {
  cfg = cfg || {};
  ghostToken = cfg.ghost_token || "";
  const ghostPort = Number(cfg.ghost_port || 0);
  const ghostSameOrigin = cfg.ghost_same_origin === true;
  if (ghostSameOrigin) {
    apiBaseUrl = "/ghost";
  } else if (ghostPort > 0) {
    apiBaseUrl =
      "http://" +
      (window.location.hostname || "localhost") +
      ":" +
      ghostPort +
      "/ghost";
  } else {
    apiBaseUrl = "";
  }
  apiEnabled = cfg.api_enabled === true && !!ghostToken && !!apiBaseUrl;
  apiModel = cfg.api_model || "deepseek-v4-flash";
  RATE_MAX = cfg.rate_max || 30;
  RATE_WINDOW = cfg.rate_window_ms || 60000;
  RATE_COOLDOWN = cfg.rate_cooldown_ms || 15000;
  _allowedUniversities = cfg.allowed_universities || [];
  _allowedRegions = cfg.allowed_regions || [];
  _leadInBusy = cfg.lead_in_busy === true;
  _throttleHint = cfg.throttle_hint || "none";

  if (_hasInit) {
    const busy = _leadInBusy || _analyzeLocked;
    if (busy) {
      analyzeBtn.disabled = true;
      analyzeBtn.classList.add("loading");
    } else {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove("loading");
      _analyzeLocked = false;
    }
    if ((cfg.text_revision ?? null) !== _appliedTextRevision) {
      // Python 侧程序化更新文本（如“可继续补充”气泡）：同步到输入框
      textarea.value = cfg.initial_text || "";
      _lastSyncedText = textarea.value;
      _suggestionQueue = [];
      currentSuggestion = "";
      _suggestionForText = "";
      updateBackdrop();
      if (textarea.value.length >= 2) {
        setTimeout(() => scheduleFetch(), 200);
      }
    }
    setFrameHeight(cfg.height);
    return;
  }
  _hasInit = true;

  if (!apiEnabled) {
    statusEl.textContent = "AI 补全未配置，仅提供规则补全";
    statusEl.classList.add("rate-limited");
    setTimeout(() => {
      statusEl.classList.remove("rate-limited");
      statusEl.textContent = "";
    }, 3000);
  }

  const initialText = cfg.initial_text || "";
  const placeholder = cfg.placeholder ?? "";

  textarea.value = initialText;
  _lastSyncedText = initialText;
  _appliedTextRevision = cfg.text_revision ?? null;
  textarea.placeholder = placeholder;
  textarea.rows = cfg.rows || 4;

  try {
    const now = Date.now();
    const toRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(CACHE_PREFIX)) {
        try {
          const entry = JSON.parse(localStorage.getItem(k));
          if (now - entry.ts > CACHE_TTL) toRemove.push(k);
        } catch (e) {
          toRemove.push(k);
        }
      }
    }
    toRemove.forEach((k) => localStorage.removeItem(k));
  } catch (e) {}

  if (initialText && initialText.length >= 2) {
    setTimeout(() => scheduleFetch(), 200);
  }

  updateBackdrop();
  setTimeout(() => setFrameHeight(), 50);
}

acceptChip.addEventListener("click", () => {
  if (currentSuggestion) acceptSuggestion();
});

function triggerAnalyze() {
  if (_analyzeLocked) return;
  const text = textarea.value.trim();
  if (!text) {
    statusEl.textContent = "请先输入学生背景信息";
    statusEl.classList.add("rate-limited");
    setTimeout(() => {
      statusEl.classList.remove("rate-limited");
      statusEl.textContent = "";
    }, 2000);
    return;
  }
  _suppressBlurUntil = Date.now() + 300;
  _analyzeLocked = true;
  analyzeBtn.disabled = true;
  analyzeBtn.classList.add("loading");
  _lastSyncedText = textarea.value;
  sendToStreamlit({
    text: text,
    action: "analyze",
    telemetry: _buildTelemetry(),
  });
  setTimeout(() => {
    if (_leadInBusy) return;
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove("loading");
    _analyzeLocked = false;
  }, 20000);
}
analyzeBtn.addEventListener("mousedown", (e) => {
  e.preventDefault();
  _suppressBlurUntil = Date.now() + 1000;
});
analyzeBtn.addEventListener("click", triggerAnalyze);

window.addEventListener("message", (event) => {
  const data = event.data || {};
  if (data.type !== "streamlit:render") return;
  init((data.args && data.args.config) || {});
});

function boot() {
  setComponentReady();
  if (window.__GHOST_CONFIG__) init(window.__GHOST_CONFIG__);
  setFrameHeight();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
