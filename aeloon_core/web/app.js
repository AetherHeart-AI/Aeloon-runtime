import {
  applyEnvelope,
  createState,
  formatCompactNumber,
  formatDuration,
  hydrateSnapshot,
  projectTurnBlocks,
  queuePrompt,
  tokenTotal,
} from "./state.js";
import { COMMANDS, commandSuggestions, parseCommand } from "./commands.js";
import {
  captureDisclosureState,
  reconcileChildren,
  restoreDisclosureState,
} from "./dom.js";
import { appendMarkdown } from "./markdown.js";
import { describeToolBlock } from "./tool-display.js";

const state = createState();
const params = new URLSearchParams(location.search);
if (params.has("t")) history.replaceState(null, "", location.pathname);

const $ = (selector) => document.querySelector(selector);
const elements = {
  workspacePath: $("#workspace-path"),
  modelTag: $("#model-tag"),
  connection: $("#connection"),
  connectionLabel: $("#connection-label"),
  chatScroll: $("#chat-scroll"),
  composer: $("#composer"),
  promptInput: $("#prompt-input"),
  commandMenu: $("#command-menu"),
  queueMeta: $("#queue-meta"),
  agentCount: $("#agent-count"),
  agentsEmpty: $("#agents-empty"),
  agentsList: $("#agents-list"),
  logsTable: $("#logs-table"),
  sessionPopover: $("#session-popover"),
  sessionList: $("#session-list"),
  currentSession: $("#current-session"),
  toastRegion: $("#toast-region"),
};

const clientId = crypto.randomUUID().slice(0, 8);
const pending = new Map();
const promptRequests = new Set();
let socket;
let reconnectTimer;
let requestSequence = 0;
let renderQueued = false;
let queuedRenderMask = 0;
let currentView = "chat";
let logFilter = "all";
let commandSelection = 0;
let renderedHistory = state.history;
const historyTurnNodes = new Map();
const logNodes = new WeakMap();

const RENDER_CHROME = 1;
const RENDER_CHAT = 2;
const RENDER_AGENTS = 4;
const RENDER_LOGS = 8;
const RENDER_ALL = RENDER_CHROME | RENDER_CHAT | RENDER_AGENTS | RENDER_LOGS;

bindInteractions();
connect();
requestRender();
setInterval(() => {
  const hasRunningAgent = [...state.liveAgents.values()].some(
    (agent) => !agent.settled,
  );
  if (currentView === "chat" && hasRunningAgent) {
    renderAgentTimes();
  }
}, 1000);

function connect() {
  clearTimeout(reconnectTimer);
  state.connection = "connecting";
  requestRender();
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/ws`);
  socket.addEventListener("open", () => {
    state.connection = "connected";
    state.error = null;
    requestRender();
    refreshSnapshot();
  });
  socket.addEventListener("message", (event) => {
    let envelope;
    try {
      envelope = JSON.parse(event.data);
    } catch {
      showToast("服务器返回了无效数据。", "error");
      return;
    }
    if (envelope.type === "response") handleResponse(envelope);
    applyEnvelope(state, envelope);
    requestRender(renderMaskForEnvelope(envelope));
  });
  socket.addEventListener("close", () => {
    state.connection = "disconnected";
    rejectPending("连接已断开");
    requestRender();
    reconnectTimer = setTimeout(connect, 1800);
  });
  socket.addEventListener("error", () => {
    state.connection = "error";
    state.error = "WebSocket 连接失败";
    requestRender();
  });
}

function nextRequestId(command) {
  requestSequence += 1;
  return `${clientId}-${command}-${Date.now()}-${requestSequence}`;
}

function sendRecord(command, payload = {}, requestId = nextRequestId(command)) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    throw new Error("运行时尚未连接");
  }
  socket.send(JSON.stringify({ type: "command", command, request_id: requestId, payload }));
  return requestId;
}

function rpc(command, payload = {}, timeoutMs = 30_000) {
  const requestId = nextRequestId(command);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pending.delete(requestId);
      reject(new Error(`${command} 请求超时`));
    }, timeoutMs);
    pending.set(requestId, { command, resolve, reject, timeout });
    try {
      sendRecord(command, payload, requestId);
    } catch (error) {
      clearTimeout(timeout);
      pending.delete(requestId);
      reject(error);
    }
  });
}

function handleResponse(envelope) {
  if (promptRequests.has(envelope.request_id)) {
    promptRequests.delete(envelope.request_id);
    if (!envelope.ok) showToast(envelope.error?.message || "消息执行失败", "error");
    else refreshSnapshot();
  }
  const waiter = pending.get(envelope.request_id);
  if (waiter) {
    clearTimeout(waiter.timeout);
    pending.delete(envelope.request_id);
    if (envelope.ok) waiter.resolve(envelope.result);
    else waiter.reject(new Error(envelope.error?.message || `${waiter.command} 失败`));
  }
  if (
    envelope.ok &&
    ["refresh_snapshot", "new_session", "resume_session"].includes(envelope.command)
  ) {
    hydrateSnapshot(state, envelope.result, {
      preserveRuntime: envelope.command === "refresh_snapshot",
    });
  }
}

function rejectPending(message) {
  for (const waiter of pending.values()) {
    clearTimeout(waiter.timeout);
    waiter.reject(new Error(message));
  }
  pending.clear();
}

function refreshSnapshot() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  rpc("refresh_snapshot").catch((error) => showToast(error.message, "error"));
}

function bindInteractions() {
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  $("#session-trigger").addEventListener("click", toggleSessions);
  $("#new-session").addEventListener("click", () => createSession());
  elements.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    submitInput();
  });
  elements.promptInput.addEventListener("input", () => {
    autoSizeComposer();
    renderCommandMenu();
  });
  elements.promptInput.addEventListener("keydown", (event) => {
    const suggestions = commandSuggestions(elements.promptInput.value);
    if (!elements.commandMenu.classList.contains("hidden") && suggestions.length) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        commandSelection =
          (commandSelection + (event.key === "ArrowDown" ? 1 : -1) + suggestions.length) %
          suggestions.length;
        renderCommandMenu();
        return;
      }
      if (event.key === "Tab") {
        event.preventDefault();
        elements.promptInput.value = `${suggestions[commandSelection][0]} `;
        renderCommandMenu();
        return;
      }
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitInput();
    }
    if (event.key === "Escape") {
      elements.commandMenu.classList.add("hidden");
      elements.sessionPopover.classList.add("hidden");
    }
  });
  $("#log-filters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter]");
    if (!button) return;
    logFilter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    requestRender();
  });
  document.addEventListener("click", (event) => {
    if (
      !elements.sessionPopover.contains(event.target) &&
      !$("#session-trigger").contains(event.target)
    ) {
      elements.sessionPopover.classList.add("hidden");
    }
  });
}

function submitInput() {
  const value = elements.promptInput.value;
  if (!value.trim()) return;
  const command = parseCommand(value);
  elements.promptInput.value = "";
  autoSizeComposer();
  elements.commandMenu.classList.add("hidden");
  if (command) {
    executeCommand(command);
    return;
  }
  const requestId = nextRequestId("prompt");
  queuePrompt(state, requestId, value);
  promptRequests.add(requestId);
  try {
    sendRecord("prompt", { prompt: value, session_id: state.sessionId }, requestId);
  } catch (error) {
    promptRequests.delete(requestId);
    state.queuedPrompts = state.queuedPrompts.filter((item) => item.requestId !== requestId);
    showToast(error.message, "error");
  }
  requestRender();
}

async function executeCommand(command) {
  try {
    if (command.name === "help") {
      showToast(COMMANDS.map(([usage, description]) => `${usage} — ${description}`).join("\n"));
    } else if (command.name === "agents") {
      setView("chat");
      $("#agents-pane").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else if (command.name === "logs") {
      setView("logs");
    } else if (command.name === "master") {
      setView("chat");
    } else if (command.name === "new") {
      await createSession();
    } else if (command.name === "sessions") {
      await toggleSessions();
    } else if (command.name === "resume") {
      if (!command.args[0]) throw new Error("用法：/resume <session>");
      await resumeSession(command.args[0]);
    } else if (command.name === "cancel-turn") {
      const result = await rpc("cancel_turn");
      showToast(result.cancelled ? "已请求取消当前 turn。" : "当前没有运行中的 turn。");
    } else if (command.name === "clear") {
      state.history = [];
      requestRender();
    } else if (command.name === "quit") {
      await rpc("shutdown");
    } else {
      throw new Error(`未知命令：/${command.name}`);
    }
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function createSession() {
  const snapshot = await rpc("new_session");
  hydrateSnapshot(state, snapshot);
  elements.sessionPopover.classList.add("hidden");
  requestRender();
}

async function resumeSession(sessionId) {
  const snapshot = await rpc("resume_session", { session_id: sessionId });
  hydrateSnapshot(state, snapshot);
  elements.sessionPopover.classList.add("hidden");
  requestRender();
}

async function toggleSessions() {
  const opening = elements.sessionPopover.classList.contains("hidden");
  elements.sessionPopover.classList.toggle("hidden", !opening);
  if (!opening) return;
  elements.sessionList.replaceChildren(node("p", "empty-copy", "载入中…"));
  try {
    const sessions = await rpc("list_sessions");
    elements.sessionList.replaceChildren(
      ...(sessions.length
        ? sessions.map((session) => {
            const button = node("button", "session-item");
            button.type = "button";
            button.append(
              node("strong", "", session.title || session.session_id),
              node(
                "span",
                "",
                `${session.turns || 0} turns · ${shortId(session.session_id)}`,
              ),
            );
            button.addEventListener("click", () => resumeSession(session.session_id));
            return button;
          })
        : [node("p", "empty-copy", "还没有已保存的会话。")]),
    );
  } catch (error) {
    showToast(error.message, "error");
  }
}

function setView(view) {
  currentView = view === "logs" ? "logs" : "chat";
  document.querySelectorAll(".view").forEach((item) => {
    item.classList.toggle("active", item.id === `view-${currentView}`);
  });
  document.querySelectorAll("[data-view]").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === currentView);
  });
  requestRender();
}

function requestRender(mask = RENDER_ALL) {
  if (!mask) return;
  queuedRenderMask |= mask;
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    const currentMask = queuedRenderMask;
    queuedRenderMask = 0;
    render(currentMask);
  });
}

function render(mask) {
  if (mask & RENDER_CHROME) renderChrome();
  if (currentView === "chat") {
    if (mask & RENDER_CHAT) renderChat();
    if (mask & RENDER_AGENTS) renderAgents();
  } else if (mask & RENDER_LOGS) {
    renderLogs();
  }
}

function renderMaskForEnvelope(envelope) {
  if (envelope.type !== "event") return RENDER_ALL;
  if (envelope.event === "log.entry") return RENDER_LOGS;
  if (envelope.event === "chat.worker.lifecycle") return RENDER_AGENTS;
  if (["chat.status", "chat.llm.response"].includes(envelope.event)) return 0;
  if (
    ["bridge.prompt.started", "chat.turn.start", "bridge.turn.cancelled"].includes(
      envelope.event,
    )
  ) {
    return RENDER_CHROME | RENDER_CHAT | RENDER_AGENTS;
  }
  return RENDER_CHROME | RENDER_CHAT;
}

function renderChrome() {
  elements.workspacePath.textContent = state.workspace || "connecting…";
  elements.workspacePath.title = state.workspace;
  elements.modelTag.textContent = state.model;
  elements.currentSession.textContent = state.sessionId || "—";
  elements.connection.dataset.status = state.connection;
  elements.connectionLabel.textContent =
    { connected: "已连接", connecting: "连接中", disconnected: "已断开", error: "错误" }[
      state.connection
    ] || state.connection;
  elements.queueMeta.textContent = state.activeTurn
    ? `turn ${shortId(state.activeTurn.turnId)} 运行中`
    : state.queuedPrompts.length
      ? `${state.queuedPrompts.length} 条等待中`
      : "准备就绪";
}

function renderChat() {
  const disclosureState = captureDisclosureState(elements.chatScroll);
  const previousScrollTop = elements.chatScroll.scrollTop;
  const wasNearBottom =
    elements.chatScroll.scrollHeight -
      elements.chatScroll.scrollTop -
      elements.chatScroll.clientHeight <
    120;
  if (renderedHistory !== state.history) {
    renderedHistory = state.history;
    historyTurnNodes.clear();
  }
  const activeHistoryKeys = new Set();
  const turns = state.history.map((turn, index) => {
    const key = historyTurnKey(turn, index);
    activeHistoryKeys.add(key);
    let article = historyTurnNodes.get(key);
    if (!article) {
      article = renderTurn(turn, false, key);
      historyTurnNodes.set(key, article);
    }
    return article;
  });
  for (const key of historyTurnNodes.keys()) {
    if (!activeHistoryKeys.has(key)) historyTurnNodes.delete(key);
  }
  if (state.activeTurn) {
    turns.push(
      renderTurn(
        state.activeTurn,
        true,
        `live:${state.activeTurn.turnId || state.activeTurn.requestId || "current"}`,
      ),
    );
  }
  for (const queued of state.queuedPrompts) {
    turns.push(
      node(
        "article",
        "turn queued-turn",
        node("div", "turn-label", `QUEUED${queued.position ? ` · ${queued.position}` : ""}`),
        node("div", "user-message", queued.prompt),
      ),
    );
  }
  if (!turns.length) {
    turns.push(
      node(
        "div",
        "chat-empty",
        node("span", "empty-glyph", "◇"),
        node("h1", "", "Aeloon Core"),
        node("p", "", "Master 负责当前对话；需要时，Harness 在这个 turn 内组织临时 agents。"),
      ),
    );
  }
  reconcileChildren(elements.chatScroll, turns);
  restoreDisclosureState(elements.chatScroll, disclosureState);
  elements.chatScroll.scrollTop = wasNearBottom
    ? elements.chatScroll.scrollHeight
    : previousScrollTop;
}

function renderTurn(turn, live, turnKey) {
  const prompt = turn.user_prompt ?? turn.userPrompt ?? "";
  const blocks = Array.isArray(turn.blocks) ? turn.blocks : [];
  const projection = projectTurnBlocks(blocks, turn.final_content || "", live);
  const article = node("article", `turn${live ? " live" : ""}`);
  article.append(
    node(
      "div",
      "turn-label",
      `${live ? "RUNNING" : "TURN"} · ${formatClock(turn.created_at || turn.startedAt)}`,
    ),
  );
  if (prompt) article.append(node("div", "user-message", prompt));
  if (projection.processBlocks.length) {
    const details = node("details", "process");
    details.dataset.disclosureKey = `${turnKey}:process`;
    if (live) details.open = true;
    details.append(
      node(
        "summary",
        "",
        live ? "正在工作" : `执行过程 · ${projection.processBlocks.length}`,
      ),
    );
    const body = node("div", "process-body");
    body.append(
      ...projection.processBlocks.map((block, index) =>
        renderProcessBlock(block, `${turnKey}:block:${block.id || index}`),
      ),
    );
    details.append(body);
    article.append(details);
  }
  if (projection.finalText) {
    const answer = node("div", "assistant-answer markdown");
    appendMarkdown(answer, projection.finalText);
    article.append(answer);
  } else if (live) {
    article.append(node("div", "thinking-line", "Master 正在组织当前 turn…"));
  }
  const usage = tokenTotal(turn.usage || {});
  const metadata = [
    usage ? `${formatCompactNumber(usage)} tok` : "",
    turn.duration_ms ? formatDuration(turn.duration_ms) : "",
  ].filter(Boolean);
  if (metadata.length) article.append(node("div", "turn-meta", metadata.join(" · ")));
  return article;
}

function renderProcessBlock(block, disclosureKey) {
  if (block.type === "tool_call") {
    const display = describeToolBlock(block);
    const details = node("details", `process-entry tool-entry ${display.status}`);
    details.dataset.disclosureKey = disclosureKey;
    details.append(
      node(
        "summary",
        "tool-summary",
        node("span", `tool-status ${display.status}`, display.icon),
        node("strong", "tool-name", display.label),
        node("span", "tool-headline", display.headline),
        block.duration_ms
          ? node("span", "tool-duration", formatDuration(block.duration_ms))
          : null,
      ),
    );
    const body = node("div", "tool-entry-body");
    if (display.argumentsText) {
      body.append(renderToolDetail("调用参数", display.argumentsText));
    }
    if (display.resultText) {
      body.append(
        renderToolDetail(display.status === "error" ? "错误详情" : "执行结果", display.resultText),
      );
    } else {
      body.append(node("p", "tool-empty", display.headline));
    }
    details.append(body);
    return details;
  }
  const entry = node("div", `process-entry ${block.type || "text"}`);
  entry.textContent = String(block.content || "");
  return entry;
}

function renderToolDetail(label, content) {
  return node(
    "section",
    "tool-detail",
    node("div", "tool-detail-label", label),
    node("pre", "", content),
  );
}

function renderAgents() {
  const agents = [...state.liveAgents.values()];
  elements.agentCount.textContent = String(agents.length);
  elements.agentsEmpty.classList.toggle("hidden", agents.length > 0);
  elements.agentsList.replaceChildren(
    ...agents.map((agent) => {
      const card = node("article", `agent-row ${agent.settled ? "settled" : "active"}`);
      card.dataset.runId = agent.run_id;
      const heading = node("div", "agent-heading");
      heading.append(
        node("span", `agent-state ${agent.status || "running"}`),
        node("strong", "", agent.worker_type_id || "agent"),
        node("code", "", shortId(agent.run_id)),
      );
      heading.append(node("span", "agent-time", formatDuration(agentElapsed(agent))));
      card.append(heading);
      if (agent.objective) card.append(node("p", "agent-objective", agent.objective));
      if (agent.phase && !agent.settled) {
        card.append(node("div", "agent-step", agent.phase));
      }
      if (agent.summary) card.append(node("p", "agent-summary", agent.summary));
      const tokens = tokenTotal(agent.usage || {});
      card.append(
        node(
          "div",
          "agent-meta",
          [
            tokens ? `${formatCompactNumber(tokens)} tok` : "",
            agent.settled ? "ephemeral · context released" : "ephemeral",
          ]
            .filter(Boolean)
            .join(" · "),
        ),
      );
      return card;
    }),
  );
}

function renderAgentTimes() {
  for (const card of elements.agentsList.querySelectorAll("[data-run-id]")) {
    const agent = state.liveAgents.get(card.dataset.runId);
    const time = card.querySelector(".agent-time");
    if (agent && time) time.textContent = formatDuration(agentElapsed(agent));
  }
}

function agentElapsed(agent) {
  return (
    agent.duration_ms ??
    agent.elapsed_ms ??
    (agent.started_at ? Date.now() - new Date(agent.started_at).getTime() : 0)
  );
}

function renderLogs() {
  const rank = { TRACE: 0, DEBUG: 1, INFO: 2, SUCCESS: 2, WARNING: 3, ERROR: 4, CRITICAL: 5 };
  const logs = state.logs.filter((entry) => {
    if (logFilter === "all") return true;
    if (logFilter === "warning") return (rank[entry.level] || 0) >= 3;
    if (logFilter === "error") return (rank[entry.level] || 0) >= 4;
    if (logFilter === "gateway") return String(entry.source || "").includes("gateway");
    return true;
  });
  const rows = logs.length
    ? [...logs].reverse().map((entry) => {
        let row = logNodes.get(entry);
        if (!row) {
          row = node("details", "log-row");
          row.append(
            node(
              "summary",
              "",
              node("time", "", formatClock(entry.ts)),
              node("span", `log-level ${String(entry.level || "INFO").toLowerCase()}`, entry.level),
              node("code", "", entry.source || "runtime"),
              node("span", "log-message", entry.message || ""),
            ),
          );
          row.append(node("pre", "", JSON.stringify(entry.detail || {}, null, 2)));
          logNodes.set(entry, row);
        }
        return row;
      })
    : [node("p", "empty-copy", "当前筛选下没有日志。")];
  reconcileChildren(elements.logsTable, rows);
}

function renderCommandMenu() {
  const suggestions = commandSuggestions(elements.promptInput.value);
  commandSelection = Math.min(commandSelection, Math.max(0, suggestions.length - 1));
  elements.commandMenu.classList.toggle("hidden", suggestions.length === 0);
  elements.commandMenu.replaceChildren(
    ...suggestions.map(([usage, description], index) => {
      const button = node("button", `command-item${index === commandSelection ? " active" : ""}`);
      button.type = "button";
      button.append(node("code", "", usage), node("span", "", description));
      button.addEventListener("mousedown", (event) => {
        event.preventDefault();
        elements.promptInput.value = `${usage} `;
        elements.promptInput.focus();
        renderCommandMenu();
      });
      return button;
    }),
  );
}

function autoSizeComposer() {
  elements.promptInput.style.height = "auto";
  elements.promptInput.style.height = `${Math.min(180, elements.promptInput.scrollHeight)}px`;
}

function showToast(message, tone = "info") {
  const toast = node("div", `toast ${tone}`, String(message));
  elements.toastRegion.append(toast);
  setTimeout(() => toast.remove(), tone === "error" ? 6000 : 3800);
}

function node(tag, className = "", ...children) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === "") continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

function shortId(value) {
  return String(value || "—").slice(0, 8);
}

function historyTurnKey(turn, index) {
  return String(
    turn.turn_id ||
      turn.request_id ||
      `${turn.created_at || ""}:${turn.user_prompt || ""}:${index}`,
  );
}

function formatClock(value) {
  if (!value) return "now";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}
