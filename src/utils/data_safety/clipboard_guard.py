import streamlit as st
import streamlit.components.v1 as components


def inject_clipboard_guard() -> None:
    st.markdown(
        """
    <style>
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] * {
        -webkit-user-select: none !important;
        -moz-user-select: none !important;
        -ms-user-select: none !important;
        user-select: none !important;
        -webkit-touch-callout: none;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    components.html(
        """
    <script>
    (function(){
      const PARENT_DOC = (window.parent && window.parent.document) ? window.parent.document : document;

      // 创建焦点陷阱，避免 DataFrame 内元素获得焦点
      let FOCUS_TRAP = null;
      function ensureFocusTrap(doc){
        try {
          if (FOCUS_TRAP && FOCUS_TRAP.ownerDocument === doc) return FOCUS_TRAP;
          const trap = doc.createElement('div');
          trap.setAttribute('tabindex', '0');
          trap.setAttribute('aria-hidden', 'true');
          trap.style.position = 'fixed';
          trap.style.left = '-10000px';
          trap.style.top = '-10000px';
          trap.style.width = '0px';
          trap.style.height = '0px';
          trap.style.opacity = '0';
          trap.style.pointerEvents = 'none';
          doc.body && doc.body.appendChild(trap);
          FOCUS_TRAP = trap;
          return trap;
        } catch(_) { return null; }
      }

      function hasDataFrameInPath(e){
        try {
          if (!e || !e.composedPath) return false;
          const path = e.composedPath();
          for (let i=0;i<path.length;i++){
            const el = path[i];
            if (el && el.getAttribute && el.getAttribute('data-testid') === 'stDataFrame') return true;
          }
        } catch(_) {}
        return false;
      }

      function isInsideDataFrame(el){
        try {
          while (el) {
            if (el.getAttribute && el.getAttribute('data-testid') === 'stDataFrame') return true;
            el = el.parentElement;
          }
        } catch(_) {}
        return false;
      }

      function selectionInsideDF(doc){
        try {
          const sel = doc.getSelection ? doc.getSelection() : null;
          if (!sel || sel.rangeCount === 0) return false;
          const node = sel.anchorNode;
          const el = node && node.nodeType === 1 ? node : (node ? node.parentElement : null);
          return el ? isInsideDataFrame(el) : false;
        } catch(_) { return false; }
      }

      let hoverInsideDF = false;
      function updateHoverFlagFromEvent(e){
        const t = e.target;
        hoverInsideDF = isInsideDataFrame(t);
      }

      function installClipboardGuards(doc, alwaysBlock){
        if (doc.__dfClipboardGuardsInstalled__) return;
        doc.__dfClipboardGuardsInstalled__ = true;
        // Patch execCommand('copy')
        try {
          const originalExec = doc.execCommand && doc.execCommand.bind(doc);
          if (originalExec) {
            doc.execCommand = function(command){
              try {
                const cmd = String(command || '').toLowerCase();
                if (cmd === 'copy') {
                  const shouldBlock = alwaysBlock || selectionInsideDF(doc) || isInsideDataFrame(doc.activeElement);
                  if (shouldBlock) return false;
                }
              } catch(_) {}
              return originalExec.apply(doc, arguments);
            };
          }
        } catch(_) {}
        // Patch navigator.clipboard.writeText
        try {
          const win = doc.defaultView || window;
          const nav = win.navigator || navigator;
          if (nav && nav.clipboard && typeof nav.clipboard.writeText === 'function') {
            const origWriteText = nav.clipboard.writeText.bind(nav.clipboard);
            nav.clipboard.writeText = function(text){
              try {
                const shouldBlock = alwaysBlock || selectionInsideDF(doc) || isInsideDataFrame(doc.activeElement);
                if (shouldBlock) {
                  return Promise.reject(new Error('复制已被禁止'));
                }
              } catch(_) {}
              return origWriteText(text);
            };
          }
        } catch(_) {}
      }

      function attachBlockers(doc, alwaysBlock){
        installClipboardGuards(doc, alwaysBlock);
        ensureFocusTrap(doc);

        function shouldBlock(e){
          if (alwaysBlock) return true;
          if (hasDataFrameInPath(e)) return true;
          if (selectionInsideDF(doc)) return true;
          const active = doc.activeElement;
          if (active && isInsideDataFrame(active)) return true;
          if (hoverInsideDF) return true;
          return false;
        }

        function preventIfNeeded(e){
          if (shouldBlock(e)) {
            e.preventDefault();
            e.stopPropagation();
            return false;
          }
        }

        // 复制相关事件
        doc.addEventListener('copy', preventIfNeeded, true);
        doc.addEventListener('cut', preventIfNeeded, true);
        doc.addEventListener('paste', preventIfNeeded, true);
        doc.addEventListener('beforecopy', preventIfNeeded, true);
        doc.addEventListener('contextmenu', preventIfNeeded, true);

        // 键盘事件全禁用（位于 DataFrame 内）
        const keyHandler = function(e){
          if (shouldBlock(e)) {
            e.preventDefault();
            e.stopPropagation();
            return false;
          }
        };
        doc.addEventListener('keydown', keyHandler, true);
        doc.addEventListener('keypress', keyHandler, true);
        doc.addEventListener('keyup', keyHandler, true);
        if (doc.defaultView) {
          doc.defaultView.addEventListener('keydown', keyHandler, true);
          doc.defaultView.addEventListener('keypress', keyHandler, true);
          doc.defaultView.addEventListener('keyup', keyHandler, true);
        }

        // 选区变化时，若位于表格内则清空
        doc.addEventListener('selectionchange', function(){
          try {
            const sel = doc.getSelection ? doc.getSelection() : null;
            if (!sel || sel.rangeCount === 0) return;
            const node = sel.anchorNode;
            const el = node && node.nodeType === 1 ? node : (node ? node.parentElement : null);
            if (alwaysBlock || (el && isInsideDataFrame(el))) {
              sel.removeAllRanges();
              // 选区清空后将焦点移到陷阱
              const trap = ensureFocusTrap(doc);
              trap && trap.focus && trap.focus();
            }
          } catch(_) {}
        }, true);

        // 焦点进入 DataFrame 时，立即转移到陷阱
        doc.addEventListener('focusin', function(e){
          if (shouldBlock(e)) {
            const trap = ensureFocusTrap(doc);
            if (trap && trap.focus) {
              try { trap.focus(); } catch(_) {}
            }
          }
        }, true);

        // 跟踪鼠标是否悬停在 DataFrame 内
        doc.addEventListener('pointermove', updateHoverFlagFromEvent, true);
        doc.addEventListener('mouseover', updateHoverFlagFromEvent, true);
      }

      function attachToIframesUnderDataFrame(){
        const dfRoots = PARENT_DOC.querySelectorAll('[data-testid="stDataFrame"]');
        dfRoots.forEach(function(df){
          const iframes = df.querySelectorAll('iframe');
          iframes.forEach(function(iframe){
            try {
              const idoc = iframe.contentWindow && iframe.contentWindow.document;
              if (idoc && !idoc.__dfBlockersInstalled__) {
                idoc.__dfBlockersInstalled__ = true;
                attachBlockers(idoc, true);
                installClipboardGuards(idoc, true);
              }
            } catch(_) {}
          });
        });
      }

      // 父文档拦截
      attachBlockers(PARENT_DOC, false);

      // 为 DataFrame 下的同源 iframe 注入拦截
      attachToIframesUnderDataFrame();

      // 监听 DOM 变化，处理动态渲染
      const mo = new MutationObserver(function(){
        attachToIframesUnderDataFrame();
      });
      try { mo.observe(PARENT_DOC, { childList: true, subtree: true }); } catch(_) {}

      // 定时兜底
      setInterval(attachToIframesUnderDataFrame, 1000);
    })();
    </script>
    """,
        height=0,
    )
