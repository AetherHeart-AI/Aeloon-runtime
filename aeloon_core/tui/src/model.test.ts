import { describe, expect, test } from "bun:test"
import {
  applyCommandResult,
  applyEnvelope,
  appendUserPrompt,
  createAppState,
  hydrateReady,
  markTurnCancelled,
  setVerbosity,
  setView,
  visibleMasterItems,
  visibleMasterTurns,
  visibleWorkerItems,
  waitingSummary,
} from "./model"

const event = (name: string, payload: Record<string, unknown>) =>
  ({ type: "event", event: name, payload }) as const

describe("TUI event projection", () => {
  test("keeps compact Master quiet and aggregates read floods", () => {
    const state = createAppState()

    applyEnvelope(state, event("chat.turn.start", { turn_id: "never-show-this" }))
    applyEnvelope(
      state,
      event("chat.block.add", {
        block: { id: "reasoning-private", type: "reasoning", content: "secret chain" },
      }),
    )
    applyEnvelope(state, event("chat.status", { text: "internal status" }))
    applyEnvelope(
      state,
      event("log.entry", {
        level: "INFO",
        message: "gateway request",
        source: "provider",
      }),
    )
    for (const [id, name] of [
      ["read-1", "read"],
      ["grep-1", "grep"],
      ["read-2", "read"],
    ]) {
      applyEnvelope(
        state,
        event("chat.block.add", {
          block: { id, type: "tool_call", name, arguments: { path: "private.txt" } },
        }),
      )
      applyEnvelope(
        state,
        event("chat.block.update", {
          block_id: id,
          patch: { status: "done", result: "private file contents" },
        }),
      )
    }
    applyEnvelope(
      state,
      event("chat.block.add", {
        block: {
          id: "write-1",
          type: "tool_call",
          name: "write",
          arguments: { path: "src/app.ts", content: "sensitive implementation" },
        },
      }),
    )

    const rendered = visibleMasterItems(state)
    expect(rendered.map((item) => item.kind)).toEqual(["thinking", "aggregate", "tool"])
    expect(rendered[0]?.collapsed).toBeTrue()
    expect(rendered[1]?.body).toBe("read ×2 · grep ×1")
    expect(rendered[2]?.primary).toBe("src/app.ts · 24 chars")
    expect(JSON.stringify(rendered)).not.toContain("gateway request")
    expect(JSON.stringify(rendered)).not.toContain("never-show-this")
    expect(JSON.stringify(rendered)).not.toContain("private file contents")
    expect(JSON.stringify(rendered)).not.toContain("sensitive implementation")
  })

  test("verbose reveals low-signal rows, arguments, logs, and folded reasoning", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.block.add", {
        block: {
          id: "read-1",
          type: "tool_call",
          name: "read",
          arguments: { path: "README.md", offset: 20 },
        },
      }),
    )
    applyEnvelope(
      state,
      event("chat.block.add", {
        block: { id: "reasoning-1", type: "reasoning", content: "raw thought" },
      }),
    )
    applyEnvelope(
      state,
      event("log.entry", { level: "DEBUG", message: "HTTP 200", source: "gateway" }),
    )

    setVerbosity(state, "verbose")
    const rendered = visibleMasterItems(state)
    expect(rendered.map((item) => item.kind)).toEqual(["tool", "thinking", "log"])
    expect(rendered[0]?.rawDetail).toContain('"offset": 20')
    expect(rendered[1]?.body).toBe("raw thought")
    expect(rendered[1]?.collapsed).toBeTrue()
    expect(rendered[2]?.body).toBe("HTTP 200")
  })

  test("promotes a low-signal failure out of its aggregate", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.block.add", {
        block: { id: "read-1", type: "tool_call", name: "read", arguments: { path: "x" } },
      }),
    )
    applyEnvelope(
      state,
      event("chat.block.update", {
        block_id: "read-1",
        patch: { status: "error", result: "Error: permission denied" },
      }),
    )

    const rendered = visibleMasterItems(state)
    expect(rendered.map((item) => item.kind)).toEqual(["aggregate", "tool"])
    expect(rendered[0]?.body).toBe("Routine checks completed")
    expect(rendered[1]?.status).toBe("failed")
    expect(rendered[1]?.body).toContain("permission denied")
    applyEnvelope(state, event("chat.turn.end", { duration_ms: 12 }))
    expect(visibleMasterTurns(state)[0]?.collapsed).toBeFalse()
  })

  test("preserves thinking and tool insertion order while extracting only the final answer", () => {
    const state = createAppState()
    appendUserPrompt(state, "ship the TUI")
    applyEnvelope(state, event("chat.turn.start", {}))
    applyEnvelope(state, event("chat.block.add", {
      block: { content: "checking before execution", id: "reasoning-1", type: "reasoning" },
    }))
    applyEnvelope(state, event("chat.block.add", {
      block: { arguments: { command: "bun test" }, id: "exec-1", name: "exec", type: "tool_call" },
    }))
    applyEnvelope(state, event("chat.block.update", {
      block_id: "exec-1",
      patch: { duration_ms: 34, result: "ok\nExit code: 0", status: "done" },
    }))
    applyEnvelope(state, event("chat.block.add", {
      block: { content: "checking the result", id: "reasoning-2", type: "reasoning" },
    }))
    applyEnvelope(state, event("chat.block.add", {
      block: { content: "Done.", id: "answer-1", type: "text" },
    }))
    applyEnvelope(state, event("chat.turn.end", { duration_ms: 80 }))

    const turn = visibleMasterTurns(state)[0]
    expect(turn?.collapsed).toBeTrue()
    expect(turn?.processSummary).toContain("1 tool · 80ms")
    expect(turn?.process.map((item) => [item.kind, item.body])).toEqual([
      ["thinking", "checking before execution"],
      ["tool", "exit 0 · 15 chars / 2 lines · 34ms"],
      ["thinking", "checking the result"],
    ])
    expect(turn?.answer?.body).toBe("Done.")
    expect(turn?.process[1]).toMatchObject({ verb: "RAN", primary: "bun test", metrics: "exit 0 · 15 chars / 2 lines · 34ms" })
  })

  test("an empty later turn never promotes an earlier cancelled narration to final answer", () => {
    const state = createAppState()
    appendUserPrompt(state, "first")
    applyEnvelope(state, event("chat.turn.start", {}))
    applyEnvelope(state, event("chat.block.add", {
      block: { content: "unfinished", id: "old-text", type: "text" },
    }))
    markTurnCancelled(state)
    appendUserPrompt(state, "second")
    applyEnvelope(state, event("chat.turn.start", {}))
    applyEnvelope(state, event("chat.turn.end", {}))

    const turns = visibleMasterTurns(state)
    expect(turns[0]?.answer).toBeUndefined()
    expect(turns[0]?.process.some((item) => item.body === "unfinished")).toBeTrue()
    expect(turns.at(-1)?.answer).toBeUndefined()
  })

  test("Worker unread never changes view or focus and clears on inspection", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "running",
        profile_id: "coding",
        run_id: "run-private",
        worker_id: "ab12ffff",
      }),
    )
    applyEnvelope(
      state,
      event("chat.worker.activity", {
        current_step: "Implement the reducer",
        detail_source: "worker_declared",
        phase: "working_step",
        profile_id: "coding",
        revision: 1,
        run_id: "run-private",
        todo_completed: 2,
        todo_total: 5,
        worker_id: "ab12ffff",
      }),
    )

    expect(state.view).toEqual({ kind: "master" })
    expect(state.focus).toBe("composer")
    expect(state.workers.ab12ffff?.label).toBe("coding#ab12")
    expect(state.workers.ab12ffff?.unread).toBe(2)
    expect(waitingSummary(state)).toContain("Waiting on 1 Worker")

    setView(state, { kind: "worker", workerId: "ab12ffff" })
    expect(state.workers.ab12ffff?.unread).toBe(0)

    applyEnvelope(
      state,
      event("chat.worker.heartbeat", {
        elapsed_ms: 8_000,
        profile_id: "coding",
        status: "running",
        worker_id: "ab12ffff",
      }),
    )
    expect(state.workers.ab12ffff?.unread).toBe(0)
    expect(visibleWorkerItems(state, "ab12ffff").map((item) => item.kind)).toEqual([
      "lifecycle",
      "step",
    ])
  })

  test("shows a queued Worker dispatch without exposing control ids", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "created",
        profile_id: "coding",
        run_id: "run-private-control-id",
        status: "queued",
        worker_id: "ab12ffff",
      }),
    )

    expect(state.workers.ab12ffff?.status).toBe("queued")
    expect(visibleMasterItems(state)[0]?.title).toBe("WORKER DISPATCHED")
    expect(visibleMasterItems(state)[0]?.workerLabel).toBe("coding#ab12")
    expect(JSON.stringify(visibleMasterItems(state))).not.toContain("run-private-control-id")
  })

  test("Worker detail aggregates routine tools and surfaces mutations and failures", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "running",
        profile_id: "coding",
        worker_id: "abcd1234",
      }),
    )
    for (const name of ["read", "grep", "read", "write", "exec"]) {
      applyEnvelope(
        state,
        event("chat.worker.tool.result", {
          duration_ms: 20,
          label: "coding#abcd",
          metrics: name === "exec"
            ? { command: "bun test src/model.test.ts", exit_code: 1 }
            : { result_chars: 10, result_lines: 2 },
          profile_id: "coding",
          status: name === "exec" ? "error" : "done",
          tool_name: name,
          worker_id: "abcd1234",
        }),
      )
    }

    const compact = visibleWorkerItems(state, "abcd1234")
    expect(compact.map((item) => item.kind)).toEqual([
      "lifecycle",
      "aggregate",
      "tool",
      "tool",
    ])
    expect(compact.find((item) => item.toolName === "exec")).toMatchObject({
      metrics: "exit 1 · 20ms",
      primary: "bun test src/model.test.ts",
      verb: "RAN",
    })
    expect(compact[1]?.body).toBe("read ×2 · grep ×1")
    expect(compact.at(-1)?.status).toBe("failed")

    setVerbosity(state, "verbose")
    expect(visibleWorkerItems(state, "abcd1234").filter((item) => item.kind === "tool")).toHaveLength(5)
  })

  test("profile delegates stay in Master and do not masquerade as durable Workers", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.profile.delegate.start", {
        agent_id: "reviewer",
        branch_id: "internal-uuid",
        label: "review#1",
        task: "Review the patch",
      }),
    )

    expect(state.workerOrder).toEqual([])
    expect(visibleMasterItems(state)[0]?.title).toBe("DELEGATED")
    expect(JSON.stringify(visibleMasterItems(state))).not.toContain("internal-uuid")
  })

  test("hydrates the operator-only Worker journal into a readable workbench", () => {
    const state = createAppState()
    applyCommandResult(state, "inspect_worker", {
      current_step: "Run the focused regression suite",
      phase: "using_tool",
      phases: ["planning", "using_tool"],
      profile_id: "coding",
      runs: [
        {
          goal: "Persist only safe Worker progress",
          run_id: "run-private-control-id",
          status: "running",
          worker_id: "ab12ffff",
        },
      ],
      status: "running",
      timeline_available: true,
      timeline: [
        { kind: "lifecycle", phase: "running", status: "running" },
        {
          current_step: "Run the focused regression suite",
          kind: "phase",
          phase: "working_step",
          todo_completed: 3,
          todo_total: 4,
        },
        { kind: "tools", signal: "low", status: "done", tool_counts: { read: 4, grep: 2 } },
        {
          duration_ms: 25,
          kind: "tool",
          metrics: { new_chars: 420, old_chars: 300, resource: "worker_ui.py" },
          signal: "high",
          status: "done",
          tool_name: "edit",
        },
        { action: "retry", event: "tests need one repair", kind: "guard", source: "guard" },
      ],
      todo_completed: 3,
      todo_total: 4,
      worker_id: "ab12ffff",
    })

    const worker = state.workers.ab12ffff
    expect(state.view).toEqual({ kind: "master" })
    expect(worker?.goal).toBe("Persist only safe Worker progress")
    expect(worker?.phase).toBe("executing")
    expect(worker?.currentStep).toBe("Run the focused regression suite")
    expect(worker?.todoCompleted).toBe(3)
    expect(worker?.todoTotal).toBe(4)
    expect(visibleWorkerItems(state, "ab12ffff").map((item) => item.kind)).toEqual([
      "lifecycle",
      "step",
      "aggregate",
      "tool",
      "guard",
    ])
    expect(visibleWorkerItems(state, "ab12ffff")[2]?.body).toBe("read ×4 · grep ×2")
    expect(JSON.stringify(visibleWorkerItems(state, "ab12ffff"))).not.toContain(
      "run-private-control-id",
    )
  })

  test("ignores late events from an inactive session", () => {
    const state = createAppState()
    hydrateReady(state, {
      history: [],
      model: "test",
      session_id: "session-b",
      workers: [],
      workspace: "/workspace",
    })

    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "failed",
        profile_id: "coding",
        session_id: "session-a",
        worker_id: "old-worker",
      }),
    )

    expect(state.workerOrder).toEqual([])
    expect(state.masterTimeline).toEqual([])
  })

  test("resets run-scoped Worker state when a reusable Worker starts a new run", () => {
    const state = createAppState()
    applyCommandResult(state, "inspect_worker", {
      current_step: "Old step",
      phase: "testing",
      profile_id: "coding",
      runs: [
        {
          duration_ms: 500,
          goal: "Old goal",
          run_id: "run-one",
          status: "completed",
          summary: "Old result",
          usage: { total_tokens: 10 },
          worker_id: "worker-one",
        },
      ],
      timeline: [{ kind: "lifecycle", phase: "completed", status: "completed" }],
      timeline_available: true,
      todo_completed: 2,
      todo_total: 2,
      worker_id: "worker-one",
    })

    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "created",
        profile_id: "coding",
        run_id: "run-two",
        status: "queued",
        worker_id: "worker-one",
      }),
    )

    const worker = state.workers["worker-one"]
    expect(worker?.runId).toBe("run-two")
    expect(worker?.report).toBeUndefined()
    expect(worker?.durationMs).toBeUndefined()
    expect(worker?.currentStep).toBeUndefined()
    expect(worker?.todoCompleted).toBeUndefined()
    expect(worker?.todoTotal).toBeUndefined()
    expect(worker?.usage).toEqual({})
    expect(worker?.lastRevision).toEqual({})
    expect(worker?.timeline.some((item) => item.body === "Old result")).toBeFalse()
  })

  test("shows tokens from persisted nested usage ledgers", () => {
    const state = createAppState()
    hydrateReady(state, {
      history: [
        {
          final_content: "done",
          tools_used: [],
          usage: {
            totals: {
              completion_tokens: 3,
              prompt_tokens: 7,
              total_tokens: 10,
            },
          },
          user_prompt: "hello",
        },
      ],
      model: "test",
      session_id: "session-a",
      workers: [],
      workspace: "/workspace",
    })

    expect(state.masterTimeline.at(-1)?.body).toContain("tokens 7 in / 3 out")
  })

  test("does not regress a running Worker when its spawn response arrives late", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "running",
        profile_id: "coding",
        run_id: "run-one",
        worker_id: "worker-one",
      }),
    )

    applyCommandResult(state, "spawn_worker", {
      created: true,
      profile_id: "coding",
      run_id: "run-one",
      worker_id: "worker-one",
    })

    expect(state.workers["worker-one"]?.status).toBe("running")
  })

  test("settles streaming blocks when a turn is cancelled", () => {
    const state = createAppState()
    applyEnvelope(state, event("chat.turn.start", {}))
    applyEnvelope(
      state,
      event("chat.block.add", {
        block: { content: "partial answer", id: "text-one", type: "text" },
      }),
    )
    applyEnvelope(
      state,
      event("chat.block.add", {
        block: { arguments: { cmd: "test" }, id: "tool-one", name: "exec", type: "tool_call" },
      }),
    )

    markTurnCancelled(state)
    applyEnvelope(state, event("chat.turn.start", {}))
    applyEnvelope(state, event("chat.turn.end", {}))

    const oldItems = state.masterTimeline.filter((item) =>
      ["partial answer", "exec"].some((value) =>
        item.body?.includes(value) || item.toolName === value,
      ),
    )
    expect(oldItems.map((item) => item.status)).toEqual(["cancelled", "cancelled"])
    expect(state.pendingBlocks).toEqual({})
  })

  test("uses a terminal phase even when duration is unavailable", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "cancelled",
        profile_id: "coding",
        run_id: "run-one",
        worker_id: "worker-one",
      }),
    )

    expect(state.workers["worker-one"]?.status).toBe("cancelled")
    expect(state.workers["worker-one"]?.phase).toBe("cancelled")
  })

  test("ignores late events and snapshots from an older Worker run", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "running",
        profile_id: "coding",
        run_id: "run-one",
        run_sequence: 1,
        worker_id: "worker-one",
      }),
    )
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "running",
        profile_id: "coding",
        run_id: "run-two",
        run_sequence: 2,
        worker_id: "worker-one",
      }),
    )
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "completed",
        profile_id: "coding",
        run_id: "run-one",
        run_sequence: 1,
        worker_id: "worker-one",
      }),
    )
    applyEnvelope(
      state,
      event("chat.worker.activity", {
        current_step: "stale step",
        profile_id: "coding",
        revision: 99,
        run_id: "run-one",
        run_sequence: 1,
        worker_id: "worker-one",
      }),
    )
    applyCommandResult(state, "inspect_worker", {
      phase: "completed",
      profile_id: "coding",
      runs: [
        {
          run_id: "run-one",
          run_sequence: 1,
          status: "completed",
          summary: "stale report",
          worker_id: "worker-one",
        },
      ],
      status: "completed",
      timeline: [],
      timeline_available: true,
      worker_id: "worker-one",
    })

    const worker = state.workers["worker-one"]
    expect(worker?.runId).toBe("run-two")
    expect(worker?.runSequence).toBe(2)
    expect(worker?.status).toBe("running")
    expect(worker?.currentStep).toBeUndefined()
    expect(worker?.report).toBeUndefined()
  })

  test("does not regress a terminal run to a late running lifecycle", () => {
    const state = createAppState()
    for (const phase of ["running", "completed", "running"]) {
      applyEnvelope(
        state,
        event("chat.worker.lifecycle", {
          phase,
          profile_id: "coding",
          run_id: "run-one",
          run_sequence: 1,
          worker_id: "worker-one",
        }),
      )
    }

    expect(state.workers["worker-one"]?.status).toBe("completed")
    expect(state.workers["worker-one"]?.phase).toBe("completed")
  })

  test("projects Worker recovery as a semantic action on the new run", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "partial",
        profile_id: "coding",
        run_id: "run-one",
        run_sequence: 1,
        worker_id: "worker-one",
      }),
    )

    applyCommandResult(state, "resume_worker", {
      action: "continued",
      created: true,
      goal: "finish the remaining tests",
      run_id: "run-two",
      run_sequence: 2,
      source_status: "partial",
      status: "queued",
      worker_id: "worker-one",
    })

    expect(state.workers["worker-one"]?.runSequence).toBe(2)
    expect(state.workers["worker-one"]?.status).toBe("queued")
    expect(state.masterTimeline.at(-1)?.title).toBe("WORKER RECOVERY")
    expect(state.masterTimeline.at(-1)?.body).toContain("Continuation scheduled")
  })
})
