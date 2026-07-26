import { describe, expect, test } from "bun:test";

import {
  applyEnvelope,
  applyRuntimeEvent,
  createState,
  formatCompactNumber,
  formatDuration,
  hydrateSnapshot,
  projectTurnBlocks,
  queuePrompt,
  tokenTotal,
} from "./state.js";

describe("Harness-only Web state", () => {
  test("hydrates only conversation state", () => {
    const state = createState();
    hydrateSnapshot(state, {
      workspace: "/repo",
      model: "anthropic/test",
      session_id: "session-1",
      history: [{ turn_id: "turn-1" }],
    });

    expect(state.workspace).toBe("/repo");
    expect(state.history).toHaveLength(1);
    expect(state.liveAgents.size).toBe(0);
    expect("flows" in state).toBe(false);
    expect("workers" in state).toBe(false);
  });

  test("tracks ephemeral agent lifecycle inside the current turn", () => {
    const state = createState();
    queuePrompt(state, "request-1", "Implement it");
    applyRuntimeEvent(state, "bridge.prompt.started", {
      request_id: "request-1",
      turn_id: "turn-1",
    });
    applyRuntimeEvent(state, "chat.worker.lifecycle", {
      run_id: "run-1",
      worker_type_id: "builder",
      status: "running",
      objective: "Implement it",
    });
    applyRuntimeEvent(state, "chat.worker.lifecycle", {
      run_id: "run-1",
      worker_type_id: "builder",
      status: "completed",
      summary: "Done",
    });

    const agent = state.liveAgents.get("run-1");
    expect(agent.summary).toBe("Done");
    expect(agent.settled).toBe(true);
  });

  test("turn end stores the response without discarding agent result metadata", () => {
    const state = createState();
    applyRuntimeEvent(state, "chat.turn.start", { turn_id: "turn-1" });
    applyRuntimeEvent(state, "chat.worker.lifecycle", {
      run_id: "run-1",
      status: "completed",
      worker_type_id: "reviewer",
    });
    applyRuntimeEvent(state, "chat.turn.end", {
      turn_id: "turn-1",
      final: "Complete",
      blocks: [{ id: "answer", type: "text", role: "final", content: "Complete" }],
    });

    expect(state.activeTurn).toBeNull();
    expect(state.history.at(-1).final_content).toBe("Complete");
    expect(state.liveAgents.get("run-1").settled).toBe(true);
  });

  test("ready envelope hydrates the session", () => {
    const state = createState();
    applyEnvelope(state, {
      type: "ready",
      payload: { session_id: "session-1", history: [] },
    });
    expect(state.connection).toBe("connected");
    expect(state.sessionId).toBe("session-1");
  });
});

test("projects the canonical answer separately from process blocks", () => {
  const result = projectTurnBlocks(
    [
      { id: "r", type: "reasoning", content: "Inspecting" },
      { id: "t", type: "tool_call", name: "read", status: "done" },
      { id: "a", type: "text", role: "final", content: "Done" },
    ],
    "Done",
  );
  expect(result.finalText).toBe("Done");
  expect(result.processBlocks.map((block) => block.id)).toEqual(["r", "t"]);
});

test("format helpers stay compact", () => {
  expect(tokenTotal({ input_tokens: 10, output_tokens: 5 })).toBe(15);
  expect(formatCompactNumber(1200)).toBe("1.2k");
  expect(formatDuration(65000)).toBe("1m 05s");
});
