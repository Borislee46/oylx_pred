export type WaitArgs = {
  title?: string;
  stage?: string;
  subtitle?: string;
  hint?: string;
  elapsed?: number;
  retry?: number;
  stage_index?: number;
  stage_count?: number;
  persona?: string;
  llm_text?: string;
  details?: string[] | string;
  tags?: string[] | string;
  gpa_norm?: number;
  lang_norm?: number;
  gpa_label?: string;
  lang_label?: string;
  sse_url?: string;
  sse_port?: number;
  sse_run_id?: string;
  started_at?: number;
  retry_max?: number;
  dark?: boolean;
  bg_color?: string;
};

type RenderListener = (args: WaitArgs) => void;

let ready = false;

export function post(type: string, payload: Record<string, unknown> = {}) {
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

export function setFrameHeight(height: number) {
  post("streamlit:setFrameHeight", { height });
}

export function onRender(listener: RenderListener) {
  const handler = (event: MessageEvent) => {
    const data = event.data;
    if (!data || data.type !== "streamlit:render") return;
    listener((data.args || {}) as WaitArgs);
  };
  window.addEventListener("message", handler);
  setComponentReady();
  return () => window.removeEventListener("message", handler);
}
