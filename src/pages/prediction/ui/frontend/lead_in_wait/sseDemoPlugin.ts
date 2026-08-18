import type { Plugin } from "vite";

/** Dev/preview SSE endpoint: GET /sse/wait-demo?q=<urlencoded text> */
export function sseDemoPlugin(): Plugin {
  const handler = (req: { url?: string }, res: {
    writeHead: (code: number, headers: Record<string, string>) => void;
    write: (chunk: string) => void;
    end: () => void;
    on: (ev: string, cb: () => void) => void;
  }) => {
    const raw = req.url || "";
    if (!raw.startsWith("/sse/wait-demo")) return false;
    const q = new URL(raw, "http://local").searchParams.get("q") || "Signals streaming…";
    const text = q.slice(0, 240);
    const tokens = text.match(/[\u4e00-\u9fff]|[A-Za-z0-9.]+|[^\s]/g) || text.split("");

    res.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
    });

    let i = 0;
    let closed = false;
    res.on("close", () => {
      closed = true;
    });

    const tick = () => {
      if (closed) return;
      if (i >= tokens.length) {
        res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
        res.end();
        return;
      }
      const token = tokens[i++];
      res.write(`data: ${JSON.stringify({ token })}\n\n`);
      const slow = /[。．.!？?]/.test(token);
      setTimeout(tick, slow ? 70 : 28 + Math.floor(Math.random() * 35));
    };
    tick();
    return true;
  };

  return {
    name: "lead-in-sse-demo",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!handler(req, res as never)) next();
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!handler(req, res as never)) next();
      });
    },
  };
}
