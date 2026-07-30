import { join } from "node:path";

const host = process.env.AELOON_CORE_WEB_HOST || "127.0.0.1";
const port = Number.parseInt(process.env.AELOON_CORE_WEB_PORT || "7331", 10);
const token = process.env.AELOON_CORE_WEB_TOKEN || "";
const python = process.env.AELOON_CORE_WEB_PYTHON || "python3";
const workspace = process.env.AELOON_CORE_WEB_WORKSPACE || process.cwd();
const allowedHosts = new Set(["127.0.0.1", "::1", "localhost"]);
const sessionCookie = "aeloon_web_session";
const sessionToken = crypto.randomUUID();
let bootstrapAvailable = true;

if (!token) {
  throw new Error("AELOON_CORE_WEB_TOKEN is required");
}
if (!allowedHosts.has(host)) {
  throw new Error("The Web UI only binds to the local loopback interface");
}

const bridge = Bun.spawn([python, "-m", "aeloon_core.web.bridge"], {
  cwd: workspace,
  env: process.env,
  stdin: "pipe",
  stdout: "pipe",
  stderr: "pipe",
});

const clients = new Set();
let bridgeClosed = false;
let shuttingDown = false;

function broadcast(record) {
  const encoded = typeof record === "string" ? record : JSON.stringify(record);
  for (const socket of clients) {
    try {
      socket.send(encoded);
    } catch {
      clients.delete(socket);
    }
  }
}

async function readBridge() {
  const reader = bridge.stdout.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    pending += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = pending.indexOf("\n")) >= 0) {
      const line = pending.slice(0, newline).trim();
      pending = pending.slice(newline + 1);
      if (!line) continue;
      try {
        const record = JSON.parse(line);
        broadcast(record);
        if (
          record.type === "response" &&
          record.command === "shutdown" &&
          record.ok
        ) {
          void shutdown({ requestBridge: false });
        }
      } catch {
        broadcast({
          type: "server.error",
          error: { code: "invalid_bridge_record", message: "Runtime emitted invalid JSON." },
        });
      }
    }
  }
  bridgeClosed = true;
  if (!shuttingDown) {
    broadcast({
      type: "server.error",
      error: { code: "bridge_closed", message: "Runtime bridge stopped." },
    });
  }
}

async function readBridgeErrors() {
  const text = await new Response(bridge.stderr).text();
  if (text.trim()) process.stderr.write(text);
}

readBridge();
readBridgeErrors();

const assets = new Map([
  ["/app.js", ["app.js", "text/javascript; charset=utf-8"]],
  ["/commands.js", ["commands.js", "text/javascript; charset=utf-8"]],
  ["/dom.js", ["dom.js", "text/javascript; charset=utf-8"]],
  ["/state.js", ["state.js", "text/javascript; charset=utf-8"]],
  ["/markdown.js", ["markdown.js", "text/javascript; charset=utf-8"]],
  ["/tool-display.js", ["tool-display.js", "text/javascript; charset=utf-8"]],
  ["/worker-timeline.js", ["worker-timeline.js", "text/javascript; charset=utf-8"]],
  ["/styles.css", ["styles.css", "text/css; charset=utf-8"]],
]);

const securityHeaders = {
  "Cache-Control": "no-store",
  "Content-Security-Policy":
    "default-src 'self'; connect-src 'self' ws://127.0.0.1:* ws://localhost:* ws://[::1]:*; " +
    "img-src 'self' data:; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
};

function response(body, status = 200, contentType = "text/plain; charset=utf-8") {
  return new Response(body, {
    status,
    headers: { ...securityHeaders, "Content-Type": contentType },
  });
}

function hasSession(request) {
  const cookie = request.headers.get("cookie") || "";
  return cookie
    .split(";")
    .map((item) => item.trim())
    .some((item) => item === `${sessionCookie}=${sessionToken}`);
}

function claimBootstrap(url) {
  if (!bootstrapAvailable || url.searchParams.get("t") !== token) return false;
  bootstrapAvailable = false;
  return true;
}

const server = Bun.serve({
  hostname: host,
  port,
  fetch(request, serverInstance) {
    const url = new URL(request.url);
    if (url.pathname === "/ws") {
      const origin = request.headers.get("origin");
      if (!hasSession(request) || (origin && origin !== url.origin)) {
        return response("Forbidden", 403);
      }
      if (bridgeClosed) {
        return response("Runtime unavailable", 503);
      }
      if (serverInstance.upgrade(request, { data: { connectedAt: Date.now() } })) {
        return;
      }
      return response("WebSocket upgrade failed", 400);
    }

    if (url.pathname === "/" || url.pathname === "/index.html") {
      const authenticated = hasSession(request);
      const claimed = !authenticated && claimBootstrap(url);
      if (!authenticated && !claimed) {
        return response("Forbidden", 403);
      }
      return new Response(Bun.file(join(import.meta.dir, "index.html")), {
        headers: {
          ...securityHeaders,
          "Content-Type": "text/html; charset=utf-8",
          ...(claimed
            ? {
                "Set-Cookie":
                  `${sessionCookie}=${sessionToken}; HttpOnly; SameSite=Strict; Path=/`,
              }
            : {}),
        },
      });
    }

    const asset = assets.get(url.pathname);
    if (asset) {
      if (!hasSession(request)) return response("Forbidden", 403);
      const body =
        typeof asset[0] === "string"
          ? Bun.file(join(import.meta.dir, asset[0]))
          : asset[0];
      return new Response(body, {
        headers: { ...securityHeaders, "Content-Type": asset[1] },
      });
    }
    return response("Not found", 404);
  },
  websocket: {
    open(socket) {
      clients.add(socket);
    },
    message(socket, message) {
      if (typeof message !== "string" || message.length > 1_000_000) {
        socket.send(JSON.stringify({
          type: "server.error",
          error: { code: "invalid_message", message: "Command must be bounded JSON text." },
        }));
        return;
      }
      try {
        JSON.parse(message);
      } catch {
        socket.send(JSON.stringify({
          type: "server.error",
          error: { code: "invalid_json", message: "Command is not valid JSON." },
        }));
        return;
      }
      bridge.stdin.write(`${message}\n`);
      bridge.stdin.flush();
    },
    close(socket) {
      clients.delete(socket);
    },
  },
});

const browserHost = host === "::1" ? "[::1]" : host;
const url = `http://${browserHost}:${server.port}/?t=${encodeURIComponent(token)}`;
console.log(JSON.stringify({ type: "server.ready", url, port: server.port }));

async function shutdown({ requestBridge = true } = {}) {
  if (shuttingDown) return;
  shuttingDown = true;
  broadcast({ type: "server.stopping" });
  if (requestBridge) {
    try {
      bridge.stdin.write(
        `${JSON.stringify({ type: "shutdown", request_id: `server-${Date.now()}` })}\n`,
      );
      bridge.stdin.flush();
      bridge.stdin.end();
    } catch {
      // The bridge may already be gone.
    }
  }
  await Bun.sleep(0);
  server.stop(true);
  const settled = await Promise.race([
    bridge.exited,
    Bun.sleep(1500).then(() => null),
  ]);
  if (settled === null) bridge.kill();
  process.exit(0);
}

process.on("SIGINT", () => void shutdown());
process.on("SIGTERM", () => void shutdown());
