import { afterEach, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

let serverProcess;
let testDirectory;

afterEach(async () => {
  if (serverProcess?.exitCode === null) serverProcess.kill();
  if (serverProcess) await serverProcess.exited;
  if (testDirectory) await rm(testDirectory, { recursive: true, force: true });
  serverProcess = undefined;
  testDirectory = undefined;
});

test("/quit shuts down both the runtime bridge and Web server", async () => {
  testDirectory = await mkdtemp(resolve(tmpdir(), "aeloon-web-test-"));
  const repository = resolve(import.meta.dir, "../..");
  serverProcess = Bun.spawn(["bun", "run", "server.js"], {
    cwd: import.meta.dir,
    env: {
      ...process.env,
      AELOON_CORE_WEB_CONFIG_JSON: JSON.stringify({
        workspace: repository,
        data_dir: testDirectory,
      }),
      AELOON_CORE_WEB_HOST: "127.0.0.1",
      AELOON_CORE_WEB_PORT: "0",
      AELOON_CORE_WEB_TOKEN: "shutdown-test-token",
      AELOON_CORE_WEB_PYTHON: resolve(repository, ".venv/bin/python"),
      AELOON_CORE_WEB_WORKSPACE: repository,
    },
    stdout: "pipe",
    stderr: "pipe",
  });

  const ready = JSON.parse(await firstLine(serverProcess.stdout));
  const pageResponse = await fetch(ready.url);
  expect(pageResponse.status).toBe(200);
  const cookie = pageResponse.headers.get("set-cookie")?.split(";", 1)[0];
  expect(cookie).toStartWith("aeloon_web_session=");
  await expectModuleGraphIsServed(ready.url, cookie, "/app.js");

  const pageUrl = new URL(ready.url);
  const socketUrl = new URL("/ws", pageUrl);
  socketUrl.protocol = "ws:";
  const socket = new WebSocket(socketUrl, {
    headers: {
      Cookie: cookie,
      Origin: pageUrl.origin,
    },
  });
  await opened(socket);
  const response = nextMessage(
    socket,
    (record) => record.type === "response" && record.command === "shutdown",
  );
  socket.send(
    JSON.stringify({
      type: "command",
      command: "shutdown",
      request_id: "shutdown-test",
      payload: {},
    }),
  );

  expect(await response).toMatchObject({ ok: true, request_id: "shutdown-test" });
  expect(await withTimeout(serverProcess.exited, 5_000, "server did not exit")).toBe(0);
  serverProcess = undefined;
});

async function expectModuleGraphIsServed(pageUrl, cookie, entryPath) {
  const pending = [entryPath];
  const visited = new Set();
  while (pending.length) {
    const modulePath = pending.shift();
    if (visited.has(modulePath)) continue;
    visited.add(modulePath);
    const moduleUrl = new URL(modulePath, pageUrl);
    const moduleResponse = await fetch(moduleUrl, {
      headers: { Cookie: cookie },
    });
    expect(moduleResponse.status).toBe(200);
    const source = await moduleResponse.text();
    for (const match of source.matchAll(/\bfrom\s+["'](\.\/[^"']+)["']/g)) {
      pending.push(new URL(match[1], moduleUrl).pathname);
    }
  }
}

async function firstLine(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  try {
    while (!buffered.includes("\n")) {
      const { value, done } = await withTimeout(
        reader.read(),
        5_000,
        "server did not become ready",
      );
      if (done) throw new Error("server stopped before becoming ready");
      buffered += decoder.decode(value, { stream: true });
    }
    return buffered.slice(0, buffered.indexOf("\n"));
  } finally {
    reader.releaseLock();
  }
}

function opened(socket) {
  return withTimeout(
    new Promise((resolveOpen, rejectOpen) => {
      socket.addEventListener("open", resolveOpen, { once: true });
      socket.addEventListener("error", rejectOpen, { once: true });
    }),
    5_000,
    "WebSocket did not open",
  );
}

function nextMessage(socket, predicate) {
  return withTimeout(
    new Promise((resolveMessage, rejectMessage) => {
      socket.addEventListener("error", rejectMessage, { once: true });
      socket.addEventListener("message", (event) => {
        const record = JSON.parse(String(event.data));
        if (predicate(record)) resolveMessage(record);
      });
    }),
    5_000,
    "expected WebSocket message was not received",
  );
}

function withTimeout(promise, timeoutMs, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), timeoutMs)),
  ]);
}
