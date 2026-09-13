let ready = false;

export function post(type, payload = {}) {
  window.parent.postMessage(
    { isStreamlitMessage: true, type, ...payload },
    "*",
  );
}

export function setComponentReady() {
  if (ready) return;
  ready = true;
  post("streamlit:componentReady", { apiVersion: 1 });
}

export function setFrameHeight(height) {
  post("streamlit:setFrameHeight", { height });
}

export function onRender(listener) {
  const handler = (event) => {
    const data = event.data;
    if (!data || data.type !== "streamlit:render") return;
    listener(data.args || {} || {});
  };
  window.addEventListener("message", handler);
  setComponentReady();
  return () => window.removeEventListener("message", handler);
}
