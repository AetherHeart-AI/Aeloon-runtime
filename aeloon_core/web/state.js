export const SETTLED_AGENT_STATES = new Set(["completed", "failed", "cancelled"]);

export function createState() {
  return {
    connection: "connecting",
    workspace: "",
    model: "",
    sessionId: "",
    history: [],
    activeTurn: null,
    queuedPrompts: [],
    liveAgents: new Map(),
    logs: [],
    error: null,
    hydrated: false,
  };
}

export function hydrateSnapshot(state, snapshot = {}, options = {}) {
  const sessionId = String(snapshot.session_id || snapshot.session || "");
  const preserveRuntime =
    options.preserveRuntime === true && state.sessionId && state.sessionId === sessionId;
  state.workspace = String(snapshot.workspace || "");
  state.model = String(snapshot.model || "");
  state.sessionId = sessionId;
  state.history = Array.isArray(snapshot.history) ? snapshot.history : [];
  if (!preserveRuntime) {
    state.activeTurn = null;
    state.queuedPrompts = [];
    state.liveAgents = new Map();
  }
  state.error = null;
  state.hydrated = true;
  return state;
}

export function queuePrompt(state, requestId, prompt) {
  state.queuedPrompts.push({ requestId, prompt });
}

export function applyEnvelope(state, envelope) {
  if (!envelope || typeof envelope !== "object") return state;
  if (envelope.type === "ready") {
    state.connection = "connected";
    return hydrateSnapshot(state, envelope.payload || {});
  }
  if (envelope.type === "event") {
    if (
      envelope.event !== "log.entry" &&
      !runtimeEventBelongsToSession(state, envelope.payload || {})
    ) {
      return state;
    }
    applyRuntimeEvent(state, envelope.event, envelope.payload || {});
  } else if (
    envelope.type === "response" &&
    envelope.command === "prompt" &&
    !envelope.ok
  ) {
    failPrompt(
      state,
      envelope.request_id,
      envelope.error?.code === "turn_cancelled" ? "cancelled" : "failed",
    );
  } else if (envelope.type === "server.error") {
    state.connection = "error";
    state.error = envelope.error?.message || "Web server error";
  } else if (envelope.type === "server.stopping") {
    state.connection = "disconnected";
  }
  return state;
}

export function applyRuntimeEvent(state, event, payload) {
  if (event === "log.entry") {
    state.logs.push(payload);
    if (state.logs.length > 1000) state.logs.splice(0, state.logs.length - 1000);
    return;
  }
  if (event === "bridge.prompt.queued") {
    const queued = state.queuedPrompts.find((item) => item.requestId === payload.request_id);
    if (queued) queued.position = payload.position;
    return;
  }
  if (event === "bridge.prompt.started" || event === "chat.turn.start") {
    const queuedIndex = state.queuedPrompts.findIndex(
      (item) => item.requestId === payload.request_id,
    );
    const queued =
      queuedIndex >= 0 ? state.queuedPrompts.splice(queuedIndex, 1)[0] : null;
    if (!state.activeTurn) {
      state.liveAgents = new Map();
      state.activeTurn = {
        turnId: payload.turn_id || "",
        requestId: payload.request_id || queued?.requestId || "",
        userPrompt: queued?.prompt || "",
        startedAt: payload.ts || new Date().toISOString(),
        blocks: [],
        usage: {},
      };
    } else if (payload.turn_id) {
      state.activeTurn.turnId = payload.turn_id;
    }
    return;
  }
  if (event === "bridge.turn.cancelled") {
    failPrompt(state, payload.request_id, "cancelled", payload.ts);
    return;
  }
  if (event === "chat.block.add") {
    const turn = ensureActiveTurn(state, payload);
    const block = payload.block || {};
    const existing = turn.blocks.find((item) => item.id === block.id);
    if (existing) {
      // A result may arrive first after reconnecting or when an earlier add
      // failed. Fill the placeholder without rolling a settled block back to
      // the start event's "running" state.
      const current = { ...existing };
      Object.assign(existing, block, current);
    } else {
      turn.blocks.push({ ...block });
    }
    return;
  }
  if (event === "chat.block.delta") {
    const turn = ensureActiveTurn(state, payload);
    const block = findOrCreateBlock(turn, payload.block_id, "text");
    block.content = String(block.content || "") + String(payload.delta || "");
    return;
  }
  if (event === "chat.block.update") {
    const turn = ensureActiveTurn(state, payload);
    const block = findOrCreateBlock(turn, payload.block_id, "tool_call");
    Object.assign(block, payload.patch || {});
    return;
  }
  if (event === "chat.usage") {
    ensureActiveTurn(state, payload).usage = { ...(payload.usage || {}) };
    return;
  }
  if (event === "chat.worker.lifecycle") {
    const runId = String(payload.run_id || "");
    if (!runId) return;
    const current = state.liveAgents.get(runId) || { run_id: runId };
    const status = String(payload.status || payload.phase || "running");
    state.liveAgents.set(runId, {
      ...current,
      ...payload,
      status,
      settled: SETTLED_AGENT_STATES.has(status),
      started_at: current.started_at || payload.ts || new Date().toISOString(),
      updated_at: payload.ts || new Date().toISOString(),
    });
    return;
  }
  if (event === "chat.turn.end") {
    const turn = ensureActiveTurn(state, payload);
    state.history.push({
      turn_id: payload.turn_id || turn.turnId,
      request_id: turn.requestId,
      created_at: payload.ts || new Date().toISOString(),
      user_prompt: turn.userPrompt,
      final_content: payload.final || "",
      blocks: Array.isArray(payload.blocks) ? payload.blocks : turn.blocks,
      usage: turn.usage,
      duration_ms: payload.duration_ms,
    });
    state.activeTurn = null;
  }
}

export function failPrompt(
  state,
  requestId,
  status = "failed",
  completedAt = new Date().toISOString(),
) {
  state.queuedPrompts = state.queuedPrompts.filter(
    (item) => item.requestId !== requestId,
  );
  const active = state.activeTurn;
  if (
    !active ||
    (active.requestId && requestId && active.requestId !== requestId)
  ) {
    return state;
  }
  state.activeTurn = null;
  for (const [runId, agent] of state.liveAgents) {
    if (agent.settled) continue;
    const startedAt = new Date(agent.started_at || completedAt).getTime();
    const finishedAt = new Date(completedAt).getTime();
    state.liveAgents.set(runId, {
      ...agent,
      status,
      settled: true,
      duration_ms:
        Number.isFinite(startedAt) && Number.isFinite(finishedAt)
          ? Math.max(0, finishedAt - startedAt)
          : agent.duration_ms,
      updated_at: completedAt,
    });
  }
  return state;
}

function runtimeEventBelongsToSession(state, payload) {
  const eventSessionId = String(payload.session_id || "");
  return !eventSessionId || !state.sessionId || eventSessionId === state.sessionId;
}

function ensureActiveTurn(state, payload) {
  if (!state.activeTurn) {
    state.activeTurn = {
      turnId: payload.turn_id || "",
      requestId: "",
      userPrompt: "",
      startedAt: payload.ts || new Date().toISOString(),
      blocks: [],
      usage: {},
    };
  }
  return state.activeTurn;
}

function findOrCreateBlock(turn, blockId, type) {
  let block = turn.blocks.find((item) => item.id === blockId);
  if (!block) {
    block = { id: blockId, type, content: "" };
    turn.blocks.push(block);
  }
  return block;
}

export function projectTurnBlocks(blocks = [], finalContent = "", live = false) {
  const textBlocks = blocks.filter(
    (block) => block.type === "text" && String(block.content || "").trim(),
  );
  const canonical = String(finalContent || "");
  const finalBlock = live
    ? null
    : [...textBlocks].reverse().find((block) => block.role === "final") ||
      [...textBlocks]
        .reverse()
        .find((block) => canonical && String(block.content || "").trim() === canonical.trim());
  return {
    processBlocks: blocks.filter(
      (block) =>
        block.type === "tool_call" ||
        (["reasoning", "text"].includes(block.type) &&
          block !== finalBlock &&
          String(block.content || "").trim()),
    ),
    finalText: live ? "" : canonical || finalBlock?.content || "",
  };
}

export function tokenTotal(usage = {}) {
  return Number(
    usage.total_tokens ??
      usage.total ??
      (Number(usage.input_tokens || 0) + Number(usage.output_tokens || 0)),
  );
}

export function formatCompactNumber(value) {
  const number = Number(value || 0);
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(1)}m`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(1)}k`;
  return String(Math.round(number));
}

export function formatDuration(durationMs) {
  const seconds = Math.max(0, Math.round(Number(durationMs || 0) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}
