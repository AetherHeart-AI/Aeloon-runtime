import { expect, test } from "bun:test"
import type { TextareaRenderable } from "@opentui/core"
import { testRender } from "@opentui/solid"
import { createSignal } from "solid-js"
import { App, type RuntimeBridge } from "./app"
import { BridgeRequestError, type BridgeClientOptions } from "./bridge-client"
import type { BridgeEnvelope, JsonObject, ReadySnapshot } from "./protocol"

const snapshot: ReadySnapshot = {
  history: [
    {
      final_content: "I have mapped the existing event flow and started the refactor.",
      tools_used: ["read", "grep"],
      usage: { completion_tokens: 84, prompt_tokens: 210, total_tokens: 294 },
      user_prompt: "Refactor the terminal UI into a quiet operator console.",
    },
  ],
  model: "gpt-5",
  session_id: "session-internal-uuid",
  workers: [
    {
      definition: {
        description: "Build and verify complete workspace changes.",
        digest: "sha256:builder-v2",
        id: "coding",
        source: "builtin",
      },
      latest_run: {
        objective: "Build the safe Worker detail projection and verify its tests.",
        run_id: "run-internal-uuid",
        run_sequence: 1,
        status: "running",
      },
      worker_type_id: "coding",
      status: "running",
      worker_id: "ab12-worker-internal-uuid",
    },
  ],
  workspace: "/Users/demo/aeloon-core",
}

const events: BridgeEnvelope[] = [
  { type: "event", event: "chat.turn.start", payload: { turn_id: "turn-internal-uuid" } },
  {
    type: "event",
    event: "chat.block.add",
    payload: {
      block: {
        arguments: { path: "/private/source/one.py" },
        id: "read-one-internal-uuid",
        name: "read",
        type: "tool_call",
      },
    },
  },
  {
    type: "event",
    event: "chat.block.update",
    payload: {
      block_id: "read-one-internal-uuid",
      patch: { result: "private file contents", status: "done" },
    },
  },
  {
    type: "event",
    event: "chat.block.add",
    payload: {
      block: {
        content: "reasoning dump must never render",
        id: "reasoning-internal-uuid",
        type: "reasoning",
      },
    },
  },
  {
    type: "event",
    event: "chat.block.add",
    payload: {
      block: {
        arguments: { content: "safe change", path: "aeloon_core/tui/src/app.tsx" },
        id: "write-internal-uuid",
        name: "write",
        type: "tool_call",
      },
    },
  },
  {
    type: "event",
    event: "chat.block.update",
    payload: {
      block_id: "write-internal-uuid",
      patch: { duration_ms: 42, result: "ok", status: "done" },
    },
  },
  {
    type: "event",
    event: "chat.worker.lifecycle",
    payload: {
      phase: "running",
      worker_type_id: "coding",
      run_id: "run-internal-uuid",
      status: "running",
      worker_id: "ab12-worker-internal-uuid",
    },
  },
  {
    type: "event",
    event: "chat.worker.activity",
    payload: {
      current_step: "Wire the operator-safe journal",
      label: "coding#ab12",
      phase: "using_tool",
      worker_type_id: "coding",
      revision: 1,
      run_id: "run-internal-uuid",
      todo_completed: 2,
      todo_total: 5,
      tool_names: ["read"],
      worker_id: "ab12-worker-internal-uuid",
    },
  },
  {
    type: "event",
    event: "chat.worker.tool.result",
    payload: {
      duration_ms: 15,
      metrics: { resource: "worker_sessions.py", result_chars: 840, result_lines: 31 },
      worker_type_id: "coding",
      status: "done",
      tool_name: "read",
      worker_id: "ab12-worker-internal-uuid",
    },
  },
  {
    type: "event",
    event: "chat.worker.tool.result",
    payload: {
      duration_ms: 41,
      metrics: {
        command: "bun test src/model.test.ts",
        exit_code: 0,
        result_preview: "worker tests passed\nExit code: 0",
      },
      worker_type_id: "coding",
      status: "done",
      tool_name: "exec",
      worker_id: "ab12-worker-internal-uuid",
    },
  },
  {
    type: "event",
    event: "chat.worker.tool.result",
    payload: {
      duration_ms: 33,
      metrics: { new_chars: 310, old_chars: 240, resource: "worker_ui_journal.py" },
      worker_type_id: "coding",
      status: "done",
      tool_name: "str_replace",
      worker_id: "ab12-worker-internal-uuid",
    },
  },
  {
    type: "event",
    event: "chat.guard.decision",
    payload: { action: "retry", event: "test failed once", source: "quality_guard" },
  },
  {
    type: "event",
    event: "log.entry",
    payload: { level: "DEBUG", message: "gateway HTTP detail", source: "gateway" },
  },
]

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

test("OpenTUI test renderer applies reactive updates", async () => {
  let update: ((value: string) => void) | undefined
  const setup = await testRender(() => {
    const [value, setValue] = createSignal("before")
    update = setValue
    return <text>{value()}</text>
  })
  update?.("after")
  await setup.flush()
  expect(setup.captureCharFrame()).toContain("after")
  setup.renderer.destroy()
})

test("compact Master renders the command deck without implementation noise", async () => {
  const setup = await testRender(
    () => <App connect={false} initialEnvelopes={events} initialSnapshot={snapshot} />,
    { height: 38, width: 110 },
  )
  await setup.flush()
  const frame = setup.captureCharFrame()

  expect(frame).toContain("AELOON")
  expect(frame).toContain("aeloon-core")
  expect(frame).toContain("MASTER")
  expect(frame).toContain("coding#ab12")
  expect(frame).toContain("2/5")
  expect(frame).toContain("GUARD")
  expect(frame).toContain("WROTE")
  expect(frame).toContain("REPLACED")
  expect(frame).toContain("running")
  expect(frame).toContain("compact")
  expect(frame).toContain("Ask Aeloon to work in this workspace")
  expect(frame).not.toContain("session-internal-uuid")
  expect(frame).not.toContain("turn-internal-uuid")
  expect(frame).not.toContain("reasoning dump")
  expect(frame).not.toContain("gateway HTTP detail")
  expect(frame).not.toContain("private file contents")
  expect(frame).not.toContain("safe change")

  setup.renderer.destroy()
})

test("completed turns collapse process rows and t restores the live timeline", async () => {
  const completedEvents: BridgeEnvelope[] = [
    { type: "event", event: "chat.turn.start", payload: {} },
    {
      type: "event",
      event: "chat.block.add",
      payload: { block: { content: "plan the check", id: "think-before", type: "reasoning" } },
    },
    {
      type: "event",
      event: "chat.block.add",
      payload: { block: { arguments: { command: "bun test" }, id: "exec-fold", name: "exec", type: "tool_call" } },
    },
    {
      type: "event",
      event: "chat.block.update",
      payload: { block_id: "exec-fold", patch: { duration_ms: 34, result: "ok\nExit code: 0", status: "done" } },
    },
    {
      type: "event",
      event: "chat.block.add",
      payload: { block: { content: "inspect the result", id: "think-after", type: "reasoning" } },
    },
    {
      type: "event",
      event: "chat.block.add",
      payload: { block: { content: "The focused tests pass.", id: "answer-fold", type: "text" } },
    },
    { type: "event", event: "chat.turn.end", payload: { duration_ms: 81 } },
  ]
  const setup = await testRender(
    () => <App connect={false} initialEnvelopes={completedEvents} initialSnapshot={{ ...snapshot, history: [] }} />,
    { height: 30, width: 100 },
  )
  await setup.flush()
  let frame = setup.captureCharFrame()
  expect(frame).toContain("▸ PROCESS · 1 tool · 81ms")
  expect(frame).toContain("The focused tests pass.")
  expect(frame).not.toContain("RAN bun test")

  setup.mockInput.pressEscape()
  await setup.flush()
  setup.mockInput.pressTab()
  setup.mockInput.pressTab()
  await setup.flush()
  setup.mockInput.pressKey("t")
  await setup.flush()
  frame = setup.captureCharFrame()
  expect(frame).toContain("▾ PROCESS · 1 tool · 81ms")
  expect(frame).toContain("RAN bun test · exit 0 · 15 chars / 2 lines · 34ms")
  expect(frame).toContain("ok")
  expect(frame.indexOf("thinking")).toBeLessThan(frame.indexOf("RAN bun test"))
  expect(frame.indexOf("RAN bun test")).toBeLessThan(frame.lastIndexOf("thinking"))
  setup.renderer.destroy()
})

test("status and shortcuts stay legible at eighty columns", async () => {
  const setup = await testRender(
    () => (
      <App
        connect={false}
        initialSnapshot={{
          history: [],
          model: "gpt-5",
          session_id: "private-session-id",
          workers: [],
          workspace: "/Users/demo/aeloon-core",
        }}
      />
    ),
    { height: 24, width: 80 },
  )
  await setup.flush()
  const frame = setup.captureCharFrame()
  expect(frame).toContain("idle · compact · pin:on · focus:composer")
  expect(frame).toContain("Tab views · Esc focus · /help")
  expect(frame).not.toContain("composTab")
  setup.renderer.destroy()
})

test("Tab opens the Worker workbench without stealing composer focus on updates", async () => {
  const setup = await testRender(
    () => <App connect={false} initialEnvelopes={events} initialSnapshot={snapshot} />,
    { height: 40, width: 110 },
  )
  expect(setup.renderer.root.findDescendantById("composer")?.focused).toBe(true)
  setup.mockInput.pressEscape()
  await setup.flush()
  expect(setup.captureCharFrame()).toContain("focus:transcript")
  setup.mockInput.pressTab()
  await setup.flush()
  const frame = setup.captureCharFrame()

  expect(frame).toContain("Build the safe Worker detail projection")
  expect(frame).toContain("Build and verify complete workspace changes")
  expect(frame).toContain("sha256:builder-v2")
  expect(frame).toContain("PHASE")
  expect(frame).toContain("TODO")
  expect(frame).toContain("Wire the operator-safe journal")
  expect(frame).not.toContain("ROUTINE ACTIVITY")
  expect(frame).toContain("worker_ui_journal.py")
  expect(frame).toContain("worker tests passed")
  expect(frame).not.toContain("run-internal-uuid")

  setup.renderer.destroy()
})

test("terminal Worker refreshes its report without taking focus from Master", async () => {
  let options: BridgeClientOptions | undefined
  const requests: string[] = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: async (command) => {
        requests.push(command)
        if (command !== "inspect_worker") return {}
        return {
          current_step: "Verify focused tests",
          phase: "completed",
          phases: ["planning", "testing", "completed"],
          worker_type_id: "coding",
          runs: [
            {
              duration_ms: 4_200,
              objective: "Build the safe Worker detail projection and verify its tests.",
              run_id: "run-internal-uuid",
              status: "completed",
              summary: "Safe journal implemented; focused tests pass.",
              worker_id: "ab12-worker-internal-uuid",
            },
          ],
          status: "completed",
          timeline: [],
          timeline_available: false,
          worker_id: "ab12-worker-internal-uuid",
        }
      },
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 40,
    width: 110,
  })
  await setup.flush()
  setup.mockInput.pressEscape()
  await Bun.sleep(20)
  await setup.flush()
  expect(setup.captureCharFrame()).toContain("focus:transcript")
  setup.mockInput.pressEscape()
  await Bun.sleep(20)
  await setup.flush()
  expect(setup.renderer.root.findDescendantById("composer")?.focused).toBe(true)

  options?.onEnvelope({
    type: "event",
    event: "chat.worker.lifecycle",
    payload: {
      duration_ms: 4_200,
      phase: "completed",
      worker_type_id: "coding",
      run_id: "run-internal-uuid",
      status: "completed",
      worker_id: "ab12-worker-internal-uuid",
    },
  })
  await setup.waitFor(() => requests.includes("inspect_worker"))
  await Bun.sleep(0)
  await Bun.sleep(0)
  await setup.flush()

  expect(setup.renderer.root.findDescendantById("composer")?.focused).toBe(true)
  expect(setup.captureCharFrame()).toContain("gpt-5 · MASTER")
  expect(setup.captureCharFrame()).not.toContain("Safe journal implemented")

  setup.mockInput.pressEscape()
  await Bun.sleep(20)
  await setup.flush()
  expect(setup.captureCharFrame()).toContain("focus:transcript")
  setup.mockInput.pressTab()
  await setup.flush()
  const frame = setup.captureCharFrame()
  expect(frame).toContain("RESULT · completed")
  expect(frame).toContain("Safe journal implemented; focused tests pass.")
  expect(frame).not.toContain("[>] Verify focused tests")

  setup.renderer.destroy()
})

test("composer accepts a second prompt while the first turn is running", async () => {
  const pending: Array<() => void> = []
  const requests: Array<{ command: string; payload?: JsonObject }> = []
  let options: BridgeClientOptions | undefined
  let starts = 0
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: (command, payload) => {
        requests.push({ command, payload })
        if (command !== "prompt") return Promise.resolve({})
        return new Promise<void>((resolve) => pending.push(resolve))
      },
      start: () => {
        starts += 1
        options?.onEnvelope({ type: "ready", payload: snapshot })
      },
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 34,
    width: 100,
  })
  await setup.flush()
  expect(options).toBeDefined()
  expect(starts).toBe(1)
  expect(setup.renderer.root.findDescendantById("composer")?.focused).toBe(true)

  await setup.mockInput.typeText("first prompt")
  setup.mockInput.pressEnter()
  await setup.mockInput.typeText("second prompt")
  setup.mockInput.pressEnter()
  await setup.flush()

  expect(requests.filter((item) => item.command === "prompt")).toHaveLength(1)
  expect(setup.captureCharFrame()).toContain("queued (1)")
  options?.onEnvelope({
    type: "event",
    event: "chat.block.add",
    payload: {
      block: {
        content: "answer to first",
        id: "answer-one",
        type: "text",
      },
    },
  })
  await setup.flush()
  const queuedFrame = setup.captureCharFrame()
  expect(queuedFrame.indexOf("first prompt")).toBeLessThan(
    queuedFrame.indexOf("answer to first"),
  )
  expect(queuedFrame.indexOf("answer to first")).toBeLessThan(
    queuedFrame.indexOf("second prompt"),
  )

  pending.shift()?.()
  await setup.waitFor(() => requests.filter((item) => item.command === "prompt").length === 2)
  pending.shift()?.()
  await setup.flush()

  setup.renderer.destroy()
})

test("late Worker inspection hydrates detail without stealing a newer Master selection", async () => {
  let options: BridgeClientOptions | undefined
  const inspection = deferred<unknown>()
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: (command) =>
        command === "inspect_worker" ? inspection.promise : Promise.resolve({}),
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 36,
    width: 100,
  })
  await setup.flush()
  setup.mockInput.pressEscape()
  await Bun.sleep(20)
  setup.mockInput.pressTab()
  await setup.flush()
  expect(setup.captureCharFrame()).toContain("gpt-5 · coding#ab12")

  setup.mockInput.pressKey("m")
  inspection.resolve({
    worker_type_id: "coding",
    runs: [],
    status: "running",
    timeline: [],
    timeline_available: true,
    worker_id: "ab12-worker-internal-uuid",
  })
  await Bun.sleep(0)
  await setup.flush()

  expect(setup.captureCharFrame()).toContain("gpt-5 · MASTER")
  expect(setup.captureCharFrame()).toContain("focus:transcript")
  setup.renderer.destroy()
})

test("startup submissions stay queued until ready and retain their user row", async () => {
  let options: BridgeClientOptions | undefined
  const requests: Array<{ command: string; payload?: JsonObject }> = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: async (command, payload) => {
        requests.push({ command, payload })
        return {}
      },
      start: () => undefined,
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 28,
    width: 100,
  })
  await setup.flush()
  await setup.mockInput.typeText("early prompt")
  setup.mockInput.pressEnter()
  await setup.flush()

  expect(requests).toEqual([])
  expect(setup.captureCharFrame()).toContain("queued (1)")

  options?.onEnvelope({ type: "ready", payload: snapshot })
  await setup.waitFor(() => requests.some((item) => item.command === "prompt"))
  await setup.flush()

  expect(setup.captureCharFrame()).toContain("early prompt")
  expect(setup.captureCharFrame()).not.toContain("queued (1)")
  setup.renderer.destroy()
})

test("operator cancellation is a cancelled turn and the FIFO continues", async () => {
  let options: BridgeClientOptions | undefined
  const promptGates = [deferred(), deferred()]
  const promptRequests: string[] = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: (command, payload) => {
        if (command === "cancel_turn") return Promise.resolve({ cancelled: true })
        if (command !== "prompt") return Promise.resolve({})
        promptRequests.push(String(payload?.prompt ?? ""))
        return promptGates[promptRequests.length - 1]?.promise ?? Promise.resolve()
      },
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    exitOnCtrlC: false,
    height: 32,
    width: 100,
  })
  await setup.flush()
  await setup.mockInput.typeText("cancel me")
  setup.mockInput.pressEnter()
  await setup.mockInput.typeText("run next")
  setup.mockInput.pressEnter()
  setup.mockInput.pressCtrlC()
  await Bun.sleep(0)
  promptGates[0]?.reject(
    new BridgeRequestError("prompt", "turn cancelled", "turn_cancelled"),
  )
  await Bun.sleep(0)
  await Bun.sleep(0)
  await Bun.sleep(0)
  await setup.flush()

  const frame = setup.captureCharFrame()
  expect(frame).toContain("TURN CANCELLED")
  expect(frame).not.toContain("TURN FAILED")
  expect(promptRequests).toEqual(["cancel me", "run next"])
  promptGates[1]?.resolve()
  setup.renderer.destroy()
})

test("composer history traverses multiple entries and returns to the draft", async () => {
  let options: BridgeClientOptions | undefined
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: async () => ({}),
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 28,
    width: 100,
  })
  await setup.flush()
  for (const value of ["first", "second", "third"]) {
    await setup.mockInput.typeText(value)
    setup.mockInput.pressEnter()
    await Bun.sleep(0)
  }
  await setup.mockInput.typeText("draft")
  const composer = setup.renderer.root.findDescendantById(
    "composer",
  ) as TextareaRenderable

  setup.mockInput.pressArrow("up")
  expect(composer.plainText).toBe("third")
  setup.mockInput.pressArrow("up")
  expect(composer.plainText).toBe("second")
  setup.mockInput.pressArrow("down")
  expect(composer.plainText).toBe("third")
  setup.mockInput.pressArrow("down")
  expect(composer.plainText).toBe("draft")

  setup.renderer.destroy()
})

test("manual spawn starts an attached in-process Worker", async () => {
  let options: BridgeClientOptions | undefined
  const requests: Array<{ command: string; payload?: JsonObject }> = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: async (command, payload) => {
        requests.push({ command, payload })
        return {
          created: true,
          worker_type_id: "coding",
          run_id: "private-run",
          worker_id: "worker-manual",
        }
      },
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 28,
    width: 100,
  })
  await setup.flush()
  await setup.mockInput.typeText("/spawn coding fix the queue")
  setup.mockInput.pressEnter()
  await Bun.sleep(0)
  await setup.flush()

  const request = requests.find((item) => item.command === "spawn_worker")
  expect(request?.payload).toMatchObject({
    objective: "fix the queue",
    session_id: "session-internal-uuid",
    worker_type_id: "coding",
  })
  setup.renderer.destroy()
})

test("a terminal Worker event from another session cannot trigger detail hydration", async () => {
  let options: BridgeClientOptions | undefined
  const requests: string[] = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: async (command) => {
        requests.push(command)
        return {}
      },
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 28,
    width: 100,
  })
  await setup.flush()
  options?.onEnvelope({
    type: "event",
    event: "chat.worker.lifecycle",
    payload: {
      phase: "failed",
      worker_type_id: "coding",
      session_id: "another-session",
      worker_id: "foreign-worker",
    },
  })
  await Bun.sleep(0)
  await setup.flush()

  expect(requests).not.toContain("inspect_worker")
  expect(setup.captureCharFrame()).not.toContain("foreign-worker")
  setup.renderer.destroy()
})

test("a late Worker detail response cannot repopulate a newly selected session", async () => {
  let options: BridgeClientOptions | undefined
  const inspection = deferred<unknown>()
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: (command) =>
        command === "inspect_worker" ? inspection.promise : Promise.resolve({}),
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 28,
    width: 100,
  })
  await setup.flush()
  setup.mockInput.pressEscape()
  await Bun.sleep(20)
  setup.mockInput.pressTab()
  await setup.flush()

  options?.onEnvelope({
    type: "ready",
    payload: {
      history: [],
      model: "gpt-5",
      session_id: "session-b",
      workers: [],
      workspace: "/Users/demo/aeloon-core",
    },
  })
  inspection.resolve({
    worker_type_id: "coding",
    runs: [],
    status: "running",
    timeline: [],
    timeline_available: true,
    worker_id: "ab12-worker-internal-uuid",
  })
  await Bun.sleep(0)
  await Bun.sleep(0)
  await setup.flush()

  expect(setup.captureCharFrame()).not.toContain("coding#ab12")
  expect(setup.captureCharFrame()).toContain("gpt-5 · MASTER")
  setup.renderer.destroy()
})

test("a late inspection cannot roll a Worker back to its previous run", async () => {
  let options: BridgeClientOptions | undefined
  const inspection = deferred<unknown>()
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: (command) =>
        command === "inspect_worker" ? inspection.promise : Promise.resolve({}),
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 34,
    width: 100,
  })
  await setup.flush()
  setup.mockInput.pressEscape()
  await Bun.sleep(20)
  setup.mockInput.pressTab()
  await setup.flush()

  options?.onEnvelope({
    type: "event",
    event: "chat.worker.lifecycle",
    payload: {
      phase: "running",
      worker_type_id: "coding",
      run_id: "run-two",
      run_sequence: 2,
      worker_id: "ab12-worker-internal-uuid",
    },
  })
  inspection.resolve({
    phase: "completed",
    worker_type_id: "coding",
    runs: [
      {
        run_id: "run-internal-uuid",
        status: "completed",
        summary: "stale report must not appear",
        worker_id: "ab12-worker-internal-uuid",
      },
    ],
    status: "completed",
    timeline: [],
    timeline_available: true,
    worker_id: "ab12-worker-internal-uuid",
  })
  await Bun.sleep(0)
  await Bun.sleep(0)
  await setup.flush()

  const frame = setup.captureCharFrame()
  expect(frame).toContain("running")
  expect(frame).not.toContain("stale report")
  setup.renderer.destroy()
})

test("session changes pause prompts until the new snapshot is installed", async () => {
  let options: BridgeClientOptions | undefined
  const sessionChange = deferred<unknown>()
  const requests: Array<{ command: string; payload?: JsonObject }> = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: (command, payload) => {
        requests.push({ command, payload })
        return command === "new_session" ? sessionChange.promise : Promise.resolve({})
      },
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 30,
    width: 100,
  })
  await setup.flush()
  await setup.mockInput.typeText("/new")
  setup.mockInput.pressEnter()
  await setup.waitFor(() => requests.some((item) => item.command === "new_session"))
  await setup.mockInput.typeText("belongs to session b")
  setup.mockInput.pressEnter()
  await setup.flush()

  expect(requests.some((item) => item.command === "prompt")).toBeFalse()
  expect(setup.captureCharFrame()).toContain("switching session")
  sessionChange.resolve({
    history: [],
    model: "gpt-5",
    session_id: "session-b",
    workers: [],
    workspace: "/Users/demo/aeloon-core",
  })
  await setup.waitFor(() => requests.some((item) => item.command === "prompt"))
  await setup.flush()

  expect(requests.find((item) => item.command === "prompt")?.payload).toEqual({
    prompt: "belongs to session b",
  })
  expect(setup.captureCharFrame()).toContain("belongs to session b")
  setup.renderer.destroy()
})

test("a failed session change keeps queued prompts held until a later switch succeeds", async () => {
  let options: BridgeClientOptions | undefined
  const failedChange = deferred<unknown>()
  const successfulChange = deferred<unknown>()
  const requests: Array<{ command: string; payload?: JsonObject }> = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: (command, payload) => {
        requests.push({ command, payload })
        if (command === "resume_session") return failedChange.promise
        if (command === "new_session") return successfulChange.promise
        return Promise.resolve({})
      },
      start: () => options?.onEnvelope({ type: "ready", payload: snapshot }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 32,
    width: 100,
  })
  await setup.flush()
  await setup.mockInput.typeText("/resume missing-session")
  setup.mockInput.pressEnter()
  await setup.waitFor(() => requests.some((item) => item.command === "resume_session"))
  await setup.mockInput.typeText("must wait for the intended session")
  setup.mockInput.pressEnter()
  failedChange.reject(new BridgeRequestError("resume_session", "session not found"))
  await Bun.sleep(0)
  await setup.flush()

  expect(requests.some((item) => item.command === "prompt")).toBeFalse()
  expect(setup.captureCharFrame()).toContain("SESSION HELD")
  expect(setup.captureCharFrame()).toContain("queued (1)")

  await setup.mockInput.typeText("/new")
  setup.mockInput.pressEnter()
  await setup.waitFor(() => requests.some((item) => item.command === "new_session"))
  successfulChange.resolve({
    history: [],
    model: "gpt-5",
    session_id: "session-c",
    workers: [],
    workspace: "/Users/demo/aeloon-core",
  })
  await setup.waitFor(() => requests.some((item) => item.command === "prompt"))

  expect(requests.find((item) => item.command === "prompt")?.payload).toEqual({
    prompt: "must wait for the intended session",
  })
  setup.renderer.destroy()
})

test("a bridge failure is visible and retains later composer input", async () => {
  let options: BridgeClientOptions | undefined
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: async () => {
        throw new Error("bridge is down")
      },
      start: () => options?.onFatal?.("Runtime bridge exited before ready."),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 28,
    width: 100,
  })
  await setup.flush()
  await setup.mockInput.typeText("keep this prompt")
  setup.mockInput.pressEnter()
  await setup.flush()

  const frame = setup.captureCharFrame()
  expect(frame).toContain("RUNTIME FAILED")
  expect(frame).toContain("failed · queued (1)")
  setup.renderer.destroy()
})

test("Worker resume pre-fills the composer and sends an explicit response", async () => {
  let options: BridgeClientOptions | undefined
  const requests: Array<{ command: string; payload?: JsonObject }> = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: async (command, payload) => {
        requests.push({ command, payload })
        return {}
      },
      start: () =>
        options?.onEnvelope({
          type: "ready",
          payload: waitingSnapshot(),
        }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 30,
    width: 100,
  })
  await setup.flush()
  setup.mockInput.pressEscape()
  await Bun.sleep(20)
  setup.mockInput.pressTab()
  await setup.flush()
  setup.mockInput.pressKey("r")
  await setup.flush()

  const composer = setup.renderer.root.findDescendantById(
    "composer",
  ) as TextareaRenderable
  expect(composer.focused).toBeTrue()
  expect(composer.plainText).toBe("/resume-worker ")
  expect(setup.captureCharFrame()).toContain("Which focused tests should I rerun?")

  await setup.mockInput.typeText("retry focused tests")
  setup.mockInput.pressEnter()
  await setup.waitFor(() => requests.some((item) => item.command === "resume_worker"))

  expect(requests.find((item) => item.command === "resume_worker")?.payload).toMatchObject({
    response: "retry focused tests",
    run_id: "run-internal-uuid",
  })
  setup.renderer.destroy()
})

test("Worker resume never treats a response prefix as another Worker selector", async () => {
  let options: BridgeClientOptions | undefined
  const requests: Array<{ command: string; payload?: JsonObject }> = []
  const bridgeFactory = (value: BridgeClientOptions): RuntimeBridge => {
    options = value
    return {
      close: async () => undefined,
      request: async (command, payload) => {
        requests.push({ command, payload })
        return {}
      },
      start: () =>
        options?.onEnvelope({
          type: "ready",
          payload: {
            ...waitingSnapshot(),
            workers: [
              ...(waitingSnapshot().workers ?? []),
              {
                latest_run: {
                  objective: "A different Worker whose label matches the response prefix.",
                  run_id: "other-run",
                  run_sequence: 3,
                  status: "failed",
                },
                worker_type_id: "coding-other",
                status: "failed",
                worker_id: "coding-other-worker",
              },
            ],
          },
        }),
    }
  }
  const setup = await testRender(() => <App bridgeFactory={bridgeFactory} />, {
    height: 32,
    width: 100,
  })
  await setup.flush()
  setup.mockInput.pressEscape()
  await Bun.sleep(20)
  setup.mockInput.pressTab()
  await setup.flush()
  setup.mockInput.pressKey("r")
  await setup.mockInput.typeText("coding-other fix focused tests")
  setup.mockInput.pressEnter()
  await setup.waitFor(() => requests.some((item) => item.command === "resume_worker"))

  expect(requests.find((item) => item.command === "resume_worker")?.payload).toMatchObject({
    response: "coding-other fix focused tests",
    run_id: "run-internal-uuid",
  })
  setup.renderer.destroy()
})

function waitingSnapshot(): ReadySnapshot {
  return {
    ...snapshot,
    workers: (snapshot.workers ?? []).map((worker) => ({
      ...worker,
      latest_run: {
        ...(worker.latest_run ?? {}),
        status: "waiting_for_context",
        waiting_question: "Which focused tests should I rerun?",
      },
      status: "idle",
    })),
  }
}
