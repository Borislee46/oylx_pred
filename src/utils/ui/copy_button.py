from __future__ import annotations

import html

import streamlit.components.v1 as components

_COPY_JS_TEMPLATE = """\
<div style="position:relative;">
  <textarea id="{textarea_id}" readonly
   style="width:100%;height:{height}px;padding:12px 14px;
          border:1px solid #e2e8f0;
          border-radius:8px;
          background:linear-gradient(180deg,rgba(255,255,255,0.88),rgba(248,250,252,0.55));
          color:#334155;font-size:13px;line-height:1.6;
          resize:vertical;outline:none;
          font-family:'Noto Sans SC','Microsoft YaHei',sans-serif;
          box-shadow:0 1px 2px rgba(0,0,0,0.05);"
   onfocus="this.style.borderColor='#06b6d4';this.style.boxShadow='0 0 0 2px rgba(6,182,212,0.2)'"
   onblur="this.style.borderColor='#e2e8f0';this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'"
  >{content}</textarea>
  <button onclick="
    var ta=document.getElementById('{textarea_id}');
    ta.select();
    try {{
      navigator.clipboard.writeText(ta.value).then(function(){{
        var b=document.getElementById('{btn_id}');
        b.textContent='已复制!';b.style.background='#10b981';
        setTimeout(function(){{b.textContent='复制';b.style.background='#0891b2';}},1500);
      }}).catch(function(){{alert('复制失败，请手动 Ctrl+C');}});
    }} catch(e) {{
      document.execCommand('copy');
      var b=document.getElementById('{btn_id}');
      b.textContent='已复制!';b.style.background='#10b981';
      setTimeout(function(){{b.textContent='复制';b.style.background='#0891b2';}},1500);
    }}
  " id="{btn_id}" style="
    position:absolute;top:8px;right:8px;
    padding:5px 14px;background:#0891b2;color:white;
    border:none;border-radius:6px;cursor:pointer;
    font-size:12px;font-weight:600;
    font-family:'Noto Sans SC','Microsoft YaHei',sans-serif;
    transition:background 0.2s,filter 0.2s;
  " onmouseover="this.style.filter='brightness(1.1)'"
   onmouseout="this.style.filter='none'"
  >复制</button>
</div>"""


def render_copyable_text(text: str, height: int = 200, key: str | None = None) -> None:
    if key is None:
        key = f"copy_{hash(text) & 0x7FFFFFFF:x}"
    safe_key = key.replace("-", "_").replace(".", "_")

    safe_text = html.escape(text)
    js = _COPY_JS_TEMPLATE.format(
        textarea_id=f"copy_ta_{safe_key}",
        btn_id=f"copy_btn_{safe_key}",
        content=safe_text,
        height=height,
    )
    components.html(js, height=height + 44)


_COPY_BTN_TEMPLATE = """\
<div style="display:flex;align-items:center;gap:8px;">
  <textarea id="{textarea_id}" readonly
   style="position:absolute;opacity:0;pointer-events:none;width:1px;height:1px;"
  >{content}</textarea>
  <button onclick="
    var ta=document.getElementById('{textarea_id}');
    try {{
      navigator.clipboard.writeText(ta.value).then(function(){{
        var b=document.getElementById('{btn_id}');
        b.textContent='已复制';b.style.background='#10b981';b.style.borderColor='#10b981';
        setTimeout(function(){{b.textContent='{label}';b.style.background='transparent';b.style.borderColor='#0891b2';}},1500);
      }}).catch(function(){{alert('复制失败，请手动 Ctrl+C');}});
    }} catch(e) {{
      document.execCommand('copy');
      var b=document.getElementById('{btn_id}');
      b.textContent='已复制';b.style.background='#10b981';b.style.borderColor='#10b981';
      setTimeout(function(){{b.textContent='{label}';b.style.background='transparent';b.style.borderColor='#0891b2';}},1500);
    }}
  " id="{btn_id}" style="
    width:100%;padding:8px 16px;
    background:transparent;color:#0891b2;
    border:1.5px solid #0891b2;
    border-radius:8px;cursor:pointer;
    font-size:13px;font-weight:600;
    font-family:'Noto Sans SC','Microsoft YaHei',sans-serif;
    transition:all 0.2s;
  " onmouseover="this.style.background='#0891b2';this.style.color='white';"
   onmouseout="this.style.background='transparent';this.style.color='#0891b2';"
  >{label}</button>
</div>"""


def render_copy_button(text: str, label: str = "复制", key: str | None = None) -> None:
    if key is None:
        key = f"copy_btn_{hash(text) & 0x7FFFFFFF:x}"
    safe_key = key.replace("-", "_").replace(".", "_")
    safe_text = html.escape(text)
    safe_label = html.escape(label)
    js = _COPY_BTN_TEMPLATE.format(
        textarea_id=f"copy_btn_ta_{safe_key}",
        btn_id=f"copy_btn_btn_{safe_key}",
        content=safe_text,
        label=safe_label,
    )
    components.html(js, height=44)
