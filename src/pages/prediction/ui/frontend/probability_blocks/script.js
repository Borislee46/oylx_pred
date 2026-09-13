let cfg = {};
let selected = new Set();
let basePct = 0;
let lastSentKey = null;
let lastCfgFp = "";
let blockContrib = {};
let rollTimer = null;

function selKey(names) {
  return [...names].sort().join("|");
}

function sendToStreamlit(value) {
  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type: "streamlit:setComponentValue",
      value: value,
      dataType: "json",
    },
    "*",
  );
}

function setFrameHeight() {
  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type: "streamlit:setFrameHeight",
      height: document.body.offsetHeight,
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

function sendValue() {
  lastSentKey = selKey(selected);
  sendToStreamlit({ selected: [...selected] });
}

const EPS = 1e-6;
function logit(p) {
  p = Math.min(1 - EPS, Math.max(EPS, p));
  return Math.log(p / (1 - p));
}
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function findBlock(name) {
  return (cfg.blocks || []).find((x) => x.name === name);
}

function capPct(v) {
  return Math.min(100, Math.max(0, Math.round(v)));
}

function findApplication(names) {
  for (const n of names) {
    const b = findBlock(n);
    if (b && b.kind === "application") return b;
  }
  return null;
}

function baseProb() {
  const b = (cfg.combos || {})[""];
  return b && b.best_prob != null ? b.best_prob : (cfg.base_pct || 0) / 100;
}

function fixedBeta(fixedPp, base) {
  if (!fixedPp || fixedPp <= 0) return 0;
  return logit(Math.min(0.99, base + fixedPp / 100)) - logit(base);
}

function pipelineKey(names) {
  return names
    .filter((n) => {
      const b = findBlock(n);
      return b && b.uplift_mode === "pipeline";
    })
    .sort()
    .join("|");
}

function pipelineProbOf(names) {
  const row = (cfg.combos || {})[pipelineKey(names)];
  return row && row.best_prob != null ? row.best_prob : baseProb();
}

function valueOf(names) {
  const base = baseProb();
  let z = logit(pipelineProbOf(names));
  const nUpgrades = upgradeProducts(names).length;
  for (const n of names) {
    const b = findBlock(n);
    if (b && b.uplift_mode === "fixed")
      z += fixedBeta(effectiveFixedPp(b, nUpgrades), base);
  }
  return sigmoid(z);
}

function upgradeProducts(names) {
  return names.filter((n) => {
    const b = findBlock(n);
    return b && b.kind === "upgrade";
  });
}

function synergyMultiplier(nUpgrades) {
  const s = cfg.synergy || {};
  if (nUpgrades >= 3) return s.three_plus ?? 1.0;
  if (nUpgrades === 2) return s.two_products ?? 1.0;
  return 1.0;
}

function effectiveFixedPp(block, nUpgrades) {
  const syn = synergyMultiplier(nUpgrades);
  return Math.max(1, Math.round(block.fixed_pp * syn));
}

function factorial(n) {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

function subsetsOf(rest) {
  const out = [[]];
  for (const x of rest) {
    const len = out.length;
    for (let i = 0; i < len; i++) out.push(out[i].concat(x));
  }
  return out;
}

function shapley(products) {
  const phi = {};
  products.forEach((p) => (phi[p] = 0));
  const n = products.length;
  if (n === 0) return phi;
  for (const p of products) {
    const rest = products.filter((q) => q !== p);
    for (const s of subsetsOf(rest)) {
      const k = s.length;
      const w = (factorial(k) * factorial(n - k - 1)) / factorial(n);
      phi[p] += w * (valueOf(s.concat(p)) - valueOf(s));
    }
  }
  return phi;
}

function lookupSinglePct(names) {
  if (!names || names.length === 0) return capPct(cfg.base_pct || 0);
  return capPct(valueOf(names) * 100);
}

function lookupDisplay(names) {
  const singlePct = lookupSinglePct(names);
  const app = findApplication(names);
  if (!app) {
    return {
      mode: "single",
      pct: singlePct,
      singlePct: singlePct,
      appName: "",
    };
  }
  const pk = app.id + "|" + pipelineKey(names);
  const row = (cfg.portfolio || {})[pk];
  if (!row) {
    // 服务端还没回传这个组合的整案概率：不能拿「单校最优」冒充「至少录1所」，
    // 否则下掉/上 block 时数字会先跳到一个错误口径的值、再被服务端纠正。
    return {
      mode: "portfolio",
      pending: true,
      key: pk,
      pct: singlePct,
      singlePct: singlePct,
      appName: app.name,
      planDelta: 0,
      planBase: null,
    };
  }
  return {
    mode: "portfolio",
    key: pk,
    pct: capPct(row.admit_one_pct),
    singlePct: singlePct,
    appName: row.app_name || app.name,
    planDelta: row.planning_delta_pct ?? 0,
    planBase: row.planning_base_pct != null ? row.planning_base_pct : null,
  };
}

function applyHeroCopy(display) {
  const label = document.getElementById("heroLabel");
  const hero = document.getElementById("hero");
  if (display.mode === "portfolio") {
    label.textContent = "按「" + display.appName + "」优化后 · 至少录取 1 所";
    hero.classList.add("portfolio-mode");
  } else {
    label.textContent = "模拟后最优录取概率";
    hero.classList.remove("portfolio-mode");
  }
}

function spawnParticles() {
  const box = document.getElementById("particles");
  box.innerHTML = "";
  for (let i = 0; i < 12; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    p.style.left = 30 + Math.random() * 40 + "%";
    p.style.top = 40 + Math.random() * 20 + "%";
    p.style.setProperty("--dx", Math.random() * 80 - 40 + "px");
    p.style.setProperty("--dy", -20 - Math.random() * 60 + "px");
    box.appendChild(p);
  }
}

function setOdometer(el, value) {
  const str = String(capPct(value));
  el.classList.add("odometer");
  if (el.childElementCount !== str.length) {
    el.innerHTML = "";
    for (let i = 0; i < str.length; i++) {
      const col = document.createElement("span");
      col.className = "odo-col";
      const strip = document.createElement("span");
      strip.className = "odo-strip";
      for (let d = 0; d <= 9; d++) {
        const cell = document.createElement("span");
        cell.textContent = d;
        strip.appendChild(cell);
      }
      col.appendChild(strip);
      el.appendChild(col);
    }
    void el.offsetWidth;
  }
  for (let i = 0; i < str.length; i++) {
    const strip = el.children[i].firstChild;
    strip.style.transitionDelay = (str.length - 1 - i) * 0.07 + "s";
    strip.style.transform = "translateY(-" + Number(str[i]) + "em)";
  }
  el.classList.add("rolling");
  clearTimeout(rollTimer);
  rollTimer = setTimeout(() => el.classList.remove("rolling"), 700);
}

function animateDisplay(from, display) {
  const el = document.getElementById("pct");
  const bar = document.getElementById("bar");
  const hero = document.getElementById("hero");
  const deltaEl = document.getElementById("delta");
  const footEl = document.getElementById("heroFoot");
  const to = display.pct;
  const changed = to !== from;
  stopPendingDots();
  deltaEl.classList.remove("pending");
  applyHeroCopy(display);
  hero.classList.remove("lift", "drop");
  void hero.offsetWidth;
  if (to > from) hero.classList.add("lift");
  else if (to < from) hero.classList.add("drop");
  setOdometer(el, to);
  bar.style.width = capPct(to) + "%";

  if (display.mode === "portfolio") {
    if (display.planDelta > 0 && display.planBase != null) {
      deltaEl.textContent =
        "科学配置 · 至少录1所 " +
        display.planBase +
        "% → " +
        display.pct +
        "%（+" +
        display.planDelta +
        "pp）";
      if (changed) spawnParticles();
    } else {
      deltaEl.textContent = "";
    }
    footEl.textContent = "同额度冲稳保优化 vs 全冲名校";
    return;
  }

  const base = capPct(cfg.base_pct || 0);
  const d = to - base;
  if (d > 0) {
    deltaEl.textContent = "较原始预测 +" + capPct(d) + "%";
    footEl.textContent = "";
    if (changed) spawnParticles();
  } else if (selected.size > 0) {
    deltaEl.textContent = "";
    footEl.textContent = "已选产品暂无明显概率提升";
  } else {
    deltaEl.textContent = "";
    footEl.textContent = "拖入产品查看概率变化";
  }
}

// 服务端整案概率尚未回传的过渡态：保持上一个已确认的数字与标签不动，
// 只在提示行说明在重算，避免数字先跳一次再跳一次。
function renderPendingText() {
  const dots = ".".repeat(dotsCount);
  document.getElementById("delta").innerHTML =
    '整案把握重算中<span class="dots">' + dots + "</span>";
}

// 点 → 点点 → 点点点 循环，表示还在算（点放在定宽容器里，行宽不变、不触发重排）
function startPendingDots() {
  if (dotsTimer !== null) return;
  dotsCount = 1;
  renderPendingText();
  if (
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  ) {
    return;
  }
  dotsTimer = setInterval(function () {
    dotsCount = (dotsCount % 3) + 1;
    renderPendingText();
  }, DOTS_INTERVAL_MS);
}

function stopPendingDots() {
  if (dotsTimer !== null) {
    clearInterval(dotsTimer);
    dotsTimer = null;
  }
}

function showPending() {
  document.getElementById("hero").classList.remove("lift", "drop");
  document.getElementById("delta").classList.add("pending");
  startPendingDots();
  lastMode = "portfolio";
}

// 组件只在 Streamlit 下一次 rerun 时才重新求值，所以等待期的两个时间点必须自己用
// 定时器驱动：0.5s 到点才显示提示，3s 到点重新求值（届时没回传就走降级）。
function clearPendingTimers() {
  if (pendingHintTimer !== null) {
    clearTimeout(pendingHintTimer);
    pendingHintTimer = null;
  }
  if (pendingDegradeTimer !== null) {
    clearTimeout(pendingDegradeTimer);
    pendingDegradeTimer = null;
  }
}

function onPendingHintDue() {
  pendingHintTimer = null;
  if (pendingKey !== null) showPending();
}

function onPendingHoldDue() {
  pendingDegradeTimer = null;
  updateUI();
}

function schedulePendingTimers() {
  clearPendingTimers();
  pendingHintTimer = setTimeout(onPendingHintDue, PENDING_HINT_DELAY_MS);
  pendingDegradeTimer = setTimeout(onPendingHoldDue, PENDING_HOLD_MS);
}

function buildTooltip(b) {
  const lines = [];
  if (b.tooltip) {
    lines.push(b.name + "  |  " + b.tooltip);
  } else {
    lines.push(b.name);
  }
  if (b.diag) {
    lines.push("诊断: " + b.diag);
  }
  if (b.pp_breakdown && b.pp_breakdown.final !== b.pp_breakdown.base) {
    const bd = b.pp_breakdown;
    const parts = ["加成分解: 基础" + bd.base + "pp"];
    if (bd.school_tier && bd.school_tier !== 1.0) {
      parts.push("院校×" + bd.school_tier.toFixed(2));
    }
    if (bd.contract_tier && bd.contract_tier !== 1.0) {
      parts.push("合同×" + bd.contract_tier.toFixed(2));
    }
    if (bd.gap) {
      parts.push("缺口×" + bd.gap.toFixed(2));
    }
    if (bd.synergy) {
      parts.push("协同×" + bd.synergy.toFixed(2));
    }
    parts.push("= +" + bd.final + "pp");
    lines.push(parts.join(" "));
  }
  return lines.join("\n");
}

function makeBlock(b, inSlot) {
  const el = document.createElement("div");
  el.className = "block" + (inSlot ? " in-slot" : "");
  el.draggable = !inSlot;
  el.dataset.id = b.id;
  el.dataset.name = b.name;
  el.dataset.kind = b.kind;
  if (b.highlight) el.classList.add("highlight");
  el.title = buildTooltip(b);
  const dot = document.createElement("span");
  dot.className = "block-dot";
  dot.style.background = b.dot || "#06b6d4";
  el.appendChild(dot);
  const label = document.createElement("span");
  label.textContent = b.name;
  el.appendChild(label);
  if (inSlot && b.kind === "upgrade" && blockContrib[b.name] != null) {
    const pp = blockContrib[b.name];
    if (pp >= 8) el.dataset.size = "lg";
    else if (pp >= 4) el.dataset.size = "md";
    const c = document.createElement("span");
    c.className = "block-contrib" + (pp >= 0.5 ? "" : " muted");
    c.textContent = pp >= 0.5 ? "+" + pp.toFixed(1) + "pp" : "≈0";
    el.appendChild(c);
  }
  el.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData(
      "text/plain",
      JSON.stringify({ id: b.id, kind: b.kind }),
    );
    e.dataTransfer.effectAllowed = "move";
  });
  if (inSlot) {
    el.addEventListener("click", () => removeBlock(b.id));
  }
  return el;
}

function blocksForKind(kind) {
  return (cfg.blocks || []).filter((b) => b.kind === kind);
}

function renderCols() {
  const area = document.getElementById("colsArea");
  area.innerHTML = "";
  (cfg.slots || []).forEach((slot) => {
    const col = document.createElement("div");
    col.className = "slot-group";

    const dropEl = document.createElement("div");
    dropEl.className = "slot";
    dropEl.dataset.slotId = slot.id;
    dropEl.dataset.kind = slot.kind;
    const hint = document.createElement("span");
    hint.className = "slot-hint";
    hint.textContent = slot.hint;
    dropEl.appendChild(hint);
    blocksForKind(slot.kind).forEach((b) => {
      if (selected.has(b.name)) {
        hint.style.display = "none";
        dropEl.appendChild(makeBlock(b, true));
      }
    });
    dropEl.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropEl.classList.add("over");
    });
    dropEl.addEventListener("dragleave", (e) => {
      if (!dropEl.contains(e.relatedTarget)) dropEl.classList.remove("over");
    });
    dropEl.addEventListener("drop", (e) => {
      e.preventDefault();
      dropEl.classList.remove("over");
      try {
        const payload = JSON.parse(e.dataTransfer.getData("text/plain"));
        addBlock(payload.id, slot.kind);
      } catch (_) {}
    });
    col.appendChild(dropEl);

    const avail = blocksForKind(slot.kind).filter((b) => !selected.has(b.name));
    if (avail.length) {
      const pool = document.createElement("div");
      pool.className = "pool";
      avail.sort(function (a, b) {
        if (a.priority === "primary" && b.priority !== "primary") return -1;
        if (b.priority === "primary" && a.priority !== "primary") return 1;
        return 0;
      });
      avail.forEach((b) => pool.appendChild(makeBlock(b, false)));
      col.appendChild(pool);
    }

    area.appendChild(col);
  });
}

function removeBlock(id) {
  const b = (cfg.blocks || []).find((x) => x.id === id);
  if (b) {
    selected.delete(b.name);
    updateUI();
    sendValue();
  }
}

function addBlock(id, targetKind) {
  const b = (cfg.blocks || []).find((x) => x.id === id);
  if (!b || b.kind !== targetKind) return;
  if (b.kind === "application") {
    blocksForKind("application").forEach((x) => selected.delete(x.name));
  }
  selected.add(b.name);
  updateUI();
  sendValue();
}

let lastPct = 0;
let lastMode = "single";
let pendingSince = null;
let pendingKey = null;
let pendingHintTimer = null;
let pendingDegradeTimer = null;
let dotsTimer = null;
let dotsCount = 0;
const PENDING_HOLD_MS = 3000;
// 与 Streamlit 自身的 transient 元素（st.spinner / st.skeleton）同约定：等超过 0.5s
// 才显示提示，算得快的时候一个字都不闪。
const PENDING_HINT_DELAY_MS = 500;
const DOTS_INTERVAL_MS = 450;

function updateUI() {
  const names = [...selected];
  const phi = shapley(upgradeProducts(names));
  blockContrib = {};
  for (const k in phi) blockContrib[k] = phi[k] * 100;
  renderCols();
  let display = lookupDisplay(names);
  if (display.pending) {
    if (pendingKey !== display.key) {
      // 换了组合键（连续上下 block）→ 重新计时，避免前一次等待把后一次提前降级
      pendingKey = display.key;
      pendingSince = Date.now();
      schedulePendingTimers();
    }
    if (Date.now() - pendingSince < PENDING_HOLD_MS) {
      // 等待期内什么都不动：到点由 pendingHintTimer / pendingDegradeTimer 接管
      setFrameHeight();
      return;
    }
    // 超时仍未回传：退回单校口径并同步标签，避免长期挂着旧数字
    display = {
      mode: "single",
      pct: display.singlePct,
      singlePct: display.singlePct,
      appName: "",
    };
  }
  clearPendingTimers();
  pendingSince = null;
  pendingKey = null;
  let from = lastPct;
  if (lastMode !== display.mode) {
    from = display.mode === "portfolio" ? display.singlePct : lastPct;
  }
  animateDisplay(from, display);
  lastPct = display.pct;
  lastMode = display.mode;
  setFrameHeight();
}

function syncSelectionFromServer(incoming) {
  const incomingKey = selKey(incoming);
  const localKey = selKey(selected);
  if (incomingKey === localKey) {
    lastSentKey = null;
    return;
  }
  if (lastSentKey !== null && incomingKey !== lastSentKey) {
    return;
  }
  selected = new Set(incoming);
  if (incomingKey === lastSentKey) lastSentKey = null;
}

function onRender(nextCfg) {
  cfg = nextCfg || {};
  const fp = JSON.stringify({
    base: cfg.base_pct,
    blocks: (cfg.blocks || []).map((b) => b.id),
  });
  const incoming = cfg.initial || [];
  if (fp !== lastCfgFp) {
    lastCfgFp = fp;
    lastSentKey = null;
    selected = new Set(incoming);
    basePct = capPct(cfg.base_pct || 0);
    lastPct = basePct;
    lastMode = "single";
    setOdometer(document.getElementById("pct"), basePct);
    document.getElementById("bar").style.width = basePct + "%";
    document.getElementById("delta").textContent = "";
    document.getElementById("heroFoot").textContent =
      "拖动下方产品，查看概率变化";
    document.getElementById("heroLabel").textContent = "模拟后最优录取概率";
    document.getElementById("hero").classList.remove("portfolio-mode");
    updateUI();
    return;
  }
  basePct = capPct(cfg.base_pct || 0);
  syncSelectionFromServer(incoming);
  updateUI();
}

window.addEventListener("message", (event) => {
  const data = event.data || {};
  if (data.type !== "streamlit:render") return;
  onRender((data.args && data.args.config) || {});
});

function boot() {
  setComponentReady();
  setFrameHeight();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
