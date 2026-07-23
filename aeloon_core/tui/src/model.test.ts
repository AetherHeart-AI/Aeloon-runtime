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

  test("summarizes native write and str_replace arguments", () => {
    const state = createAppState()
    applyEnvelope(state, event("chat.block.add", {
      block: {
        arguments: { content: "你好", path: "notes.txt" },
        id: "write-chunk",
        name: "write",
        type: "tool_call",
      },
    }))
    applyEnvelope(state, event("chat.block.add", {
      block: {
        arguments: {
          new_str: "after\n",
          old_str: "before\n",
          path: "notes.txt",
          replace_all: true,
        },
        id: "replace-one",
        name: "str_replace",
        type: "tool_call",
      },
    }))

    const rendered = visibleMasterItems(state).filter((item) => item.kind === "tool")
    expect(rendered[0]).toMatchObject({
      primary: "notes.txt · 2 chars",
      verb: "WROTE",
    })
    expect(rendered[1]).toMatchObject({
      primary: "notes.txt · 7 → 6 chars · all matches",
      verb: "REPLACED",
    })
    expect(JSON.stringify(rendered)).not.toContain("你好")
    expect(JSON.stringify(rendered)).not.toContain("before")
    expect(JSON.stringify(rendered)).not.toContain("after")
  })

  test("keeps legacy edit history readable without exposing its file body", () => {
    const state = createAppState()
    applyEnvelope(state, event("chat.block.add", {
      block: {
        arguments: {
          new_text: "private replacement body",
          old_text: "private original body",
          path: "legacy.txt",
        },
        id: "legacy-edit",
        name: "edit",
        type: "tool_call",
      },
    }))

    const rendered = visibleMasterItems(state).filter((item) => item.kind === "tool")
    expect(rendered[0]?.primary).toBe("legacy.txt")
    expect(JSON.stringify(rendered)).not.toContain("private original body")
    expect(JSON.stringify(rendered)).not.toContain("private replacement body")
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
    expect(rendered.map((item) => item.kind)).toEqual(["tool"])
    expect(rendered[0]?.status).toBe("failed")
    expect(rendered[0]?.collapsed).toBeTrue()
    expect(rendered[0]?.body).toContain("permission denied")
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
    expect(turn?.process[1]).toMatchObject({
      metrics: "exit 0 · 15 chars / 2 lines · 34ms",
      primary: "bun test",
      resultDetail: "ok",
      resultPreview: "ok",
      verb: "RAN",
    })
  })

  test("uses turn-end final content instead of promoting earlier narration", () => {
    const state = createAppState()
    appendUserPrompt(state, "fix the reported issue")
    applyEnvelope(state, event("chat.turn.start", {}))
    applyEnvelope(state, event("chat.block.add", {
      block: { content: "I will start a fresh builder.", id: "narration", type: "text" },
    }))
    applyEnvelope(state, event("chat.turn.end", {
      duration_ms: 42,
      final: "The issue is fixed and verified.",
    }))

    const turn = visibleMasterTurns(state)[0]
    expect(turn?.answer?.body).toBe("The issue is fixed and verified.")
    expect(turn?.process.some((item) => item.body === "I will start a fresh builder.")).toBeTrue()
  })

  test("keeps command output through the detail bound and exposes a short preview", () => {
    const state = createAppState()
    const output = `first line\n${"x".repeat(17_000)}\nlast line\nExit code: 0`
    applyEnvelope(state, event("chat.block.add", {
      block: {
        arguments: { command: "python generate.py" },
        id: "exec-long",
        name: "exec",
        type: "tool_call",
      },
    }))
    applyEnvelope(state, event("chat.block.update", {
      block_id: "exec-long",
      patch: { result: output, status: "done" },
    }))

    const tool = visibleMasterItems(state).find((item) => item.toolName === "exec")
    expect(tool?.resultPreview).toStartWith("first line\n")
    expect(tool?.resultPreview).toEndWith("\n…")
    expect(tool?.resultDetail).toStartWith("first line\n")
    expect(tool?.resultDetail).toContain("chars hidden")
    expect(tool?.resultDetail).toEndWith("last line")
    expect((tool?.resultDetail?.length ?? 0) > 1_200).toBeTrue()
    expect(tool?.resultDetail?.length).toBeLessThanOrEqual(16_000)
    expect(tool?.resultPreview?.length).toBeLessThanOrEqual(360)
  })

  test("keeps long failures folded behind their visible preview", () => {
    const state = createAppState()
    const output = `Error: focused tests failed\n${"x".repeat(17_000)}\nfinal failure context`
    applyEnvelope(state, event("chat.block.add", {
      block: {
        arguments: { command: "pytest -q" },
        id: "exec-failure",
        name: "exec",
        type: "tool_call",
      },
    }))
    applyEnvelope(state, event("chat.block.update", {
      block_id: "exec-failure",
      patch: { result: output, status: "error" },
    }))

    const tool = visibleMasterItems(state).find((item) => item.toolName === "exec")
    expect(tool?.collapsed).toBeTrue()
    expect(tool?.resultPreview).toStartWith("Error: focused tests failed")
    expect(tool?.resultPreview).toEndWith("\n…")
    expect(tool?.resultDetail).toContain("chars hidden")
    expect(tool?.resultDetail).toEndWith("final failure context")
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
        worker_type_id: "coding",
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
        worker_type_id: "coding",
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
        worker_type_id: "coding",
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
        worker_type_id: "coding",
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

  test("Worker compact hides routine tools and surfaces mutations, output, and failures", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "running",
        worker_type_id: "coding",
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
            ? {
                command: "bun test src/model.test.ts",
                exit_code: 1,
                result_preview: "one regression failed\nExit code: 1",
              }
            : name === "write"
              ? { input_bytes: 10, input_chars: 10 }
              : { result_chars: 10, result_lines: 2 },
          worker_type_id: "coding",
          status: name === "exec" ? "error" : "done",
          tool_name: name,
          worker_id: "abcd1234",
        }),
      )
    }

    const compact = visibleWorkerItems(state, "abcd1234")
    expect(compact.map((item) => item.kind)).toEqual([
      "lifecycle",
      "tool",
      "tool",
    ])
    expect(compact.find((item) => item.toolName === "exec")).toMatchObject({
      metrics: "exit 1 · 20ms",
      primary: "bun test src/model.test.ts",
      resultDetail: "one regression failed",
      resultPreview: "one regression failed",
      collapsed: true,
      verb: "RAN",
    })
    expect(compact.find((item) => item.toolName === "write")?.metrics)
      .toBe("10 chars written · 20ms")
    expect(compact.at(-1)?.status).toBe("failed")
    expect(state.workers.abcd1234?.unread).toBe(3)

    setVerbosity(state, "verbose")
    expect(visibleWorkerItems(state, "abcd1234").filter((item) => item.kind === "tool")).toHaveLength(5)
  })

  test("hidden routine tools stay unread-silent while their failures remain visible", () => {
    const state = createAppState()
    applyEnvelope(state, event("chat.worker.lifecycle", {
      phase: "running",
      worker_type_id: "coding",
      worker_id: "abcd1234",
    }))
    const lifecycleUnread = state.workers.abcd1234?.unread
    applyEnvelope(state, event("chat.worker.tool.result", {
      metrics: { result_chars: 12, result_lines: 1 },
      worker_type_id: "coding",
      status: "done",
      tool_name: "read",
      worker_id: "abcd1234",
    }))
    expect(state.workers.abcd1234?.unread).toBe(lifecycleUnread)
    expect(visibleWorkerItems(state, "abcd1234").some((item) => item.toolName === "read")).toBeFalse()

    applyEnvelope(state, event("chat.worker.tool.result", {
      metrics: { result_preview: "Error: permission denied" },
      worker_type_id: "coding",
      status: "error",
      tool_name: "read",
      worker_id: "abcd1234",
    }))
    expect(state.workers.abcd1234?.unread).toBe((lifecycleUnread ?? 0) + 1)
    expect(visibleWorkerItems(state, "abcd1234").find((item) => item.toolName === "read"))
      .toMatchObject({ resultPreview: "Error: permission denied", status: "failed" })
  })

  test("hydrates the operator-only Worker journal into a readable workbench", () => {
    const state = createAppState()
    applyCommandResult(state, "inspect_worker", {
      current_step: "Run the focused regression suite",
      phase: "using_tool",
      phases: ["planning", "using_tool"],
      worker_type_id: "coding",
      runs: [
        {
          objective: "Persist only safe Worker progress",
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
          tool_name: "str_replace",
        },
      ],
      todo_completed: 3,
      todo_total: 4,
      worker_id: "ab12ffff",
    })

    const worker = state.workers.ab12ffff
    expect(state.view).toEqual({ kind: "master" })
    expect(worker?.objective).toBe("Persist only safe Worker progress")
    expect(worker?.phase).toBe("executing")
    expect(worker?.currentStep).toBe("Run the focused regression suite")
    expect(worker?.todoCompleted).toBe(3)
    expect(worker?.todoTotal).toBe(4)
    expect(visibleWorkerItems(state, "ab12ffff").map((item) => item.kind)).toEqual([
      "lifecycle",
      "step",
      "tool",
    ])
    expect(JSON.stringify(visibleWorkerItems(state, "ab12ffff"))).not.toContain(
      "run-private-control-id",
    )
    setVerbosity(state, "verbose")
    const verbose = visibleWorkerItems(state, "ab12ffff")
    expect(verbose.map((item) => item.kind)).toEqual([
      "lifecycle",
      "step",
      "aggregate",
      "tool",
    ])
    expect(verbose[2]?.body).toBe("read ×4 · grep ×2")
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
        worker_type_id: "coding",
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
      worker_type_id: "coding",
      runs: [
        {
          duration_ms: 500,
          objective: "Old objective",
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
        worker_type_id: "coding",
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
        worker_type_id: "coding",
        run_id: "run-one",
        worker_id: "worker-one",
      }),
    )

    applyCommandResult(state, "spawn_worker", {
      created: true,
      worker_type_id: "coding",
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
        worker_type_id: "coding",
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
        worker_type_id: "coding",
        run_id: "run-one",
        run_sequence: 1,
        worker_id: "worker-one",
      }),
    )
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "running",
        worker_type_id: "coding",
        run_id: "run-two",
        run_sequence: 2,
        worker_id: "worker-one",
      }),
    )
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "completed",
        worker_type_id: "coding",
        run_id: "run-one",
        run_sequence: 1,
        worker_id: "worker-one",
      }),
    )
    applyEnvelope(
      state,
      event("chat.worker.activity", {
        current_step: "stale step",
        worker_type_id: "coding",
        revision: 99,
        run_id: "run-one",
        run_sequence: 1,
        worker_id: "worker-one",
      }),
    )
    applyCommandResult(state, "inspect_worker", {
      phase: "completed",
      worker_type_id: "coding",
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
          worker_type_id: "coding",
          run_id: "run-one",
          run_sequence: 1,
          worker_id: "worker-one",
        }),
      )
    }

    expect(state.workers["worker-one"]?.status).toBe("completed")
    expect(state.workers["worker-one"]?.phase).toBe("completed")
  })

  test("projects Worker resume as a continuation on the new run", () => {
    const state = createAppState()
    applyEnvelope(
      state,
      event("chat.worker.lifecycle", {
        phase: "waiting_for_context",
        worker_type_id: "coding",
        run_id: "run-one",
        run_sequence: 1,
        worker_id: "worker-one",
      }),
    )

    applyCommandResult(state, "resume_worker", {
      action: "continued",
      created: true,
      objective: "finish the remaining tests",
      run_id: "run-two",
      run_sequence: 2,
      source_status: "waiting_for_context",
      status: "queued",
      worker_id: "worker-one",
    })

    expect(state.workers["worker-one"]?.runSequence).toBe(2)
    expect(state.workers["worker-one"]?.status).toBe("queued")
    expect(state.masterTimeline.at(-1)?.title).toBe("WORKER RESUMED")
    expect(state.masterTimeline.at(-1)?.body).toContain("Continuation scheduled")
  })
})
