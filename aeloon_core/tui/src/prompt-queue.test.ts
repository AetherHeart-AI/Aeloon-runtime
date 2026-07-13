import { describe, expect, test } from "bun:test"
import { PromptQueue } from "./prompt-queue"

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

describe("PromptQueue", () => {
  test("dispatches one prompt at a time in FIFO order", async () => {
    const calls: string[] = []
    const queuedSnapshots: string[][] = []
    const gates = [deferred(), deferred(), deferred()]
    const queue = new PromptQueue(
      async (prompt) => {
        calls.push(prompt)
        await gates[calls.length - 1]?.promise
      },
      { onChanged: (items) => queuedSnapshots.push(items) },
    )

    queue.submit("one")
    queue.submit("two")
    queue.submit("three")
    expect(calls).toEqual(["one"])
    expect(queue.queued).toEqual(["two", "three"])

    gates[0]?.resolve()
    await Bun.sleep(0)
    expect(calls).toEqual(["one", "two"])
    expect(queue.queued).toEqual(["three"])

    gates[1]?.resolve()
    await Bun.sleep(0)
    expect(calls).toEqual(["one", "two", "three"])
    gates[2]?.resolve()
    await Bun.sleep(0)
    expect(queue.active).toBeUndefined()
    expect(queuedSnapshots.at(-1)).toEqual([])
  })

  test("continues after a failed turn", async () => {
    const calls: string[] = []
    const errors: string[] = []
    const gate = deferred()
    const queue = new PromptQueue(
      async (prompt) => {
        calls.push(prompt)
        if (prompt === "first") await gate.promise
      },
      { onError: (prompt) => errors.push(prompt) },
    )

    queue.submit("first")
    queue.submit("second")
    gate.reject(new Error("boom"))
    await Bun.sleep(0)
    await Bun.sleep(0)

    expect(errors).toEqual(["first"])
    expect(calls).toEqual(["first", "second"])
  })

  test("preserves multiline paste whitespace", async () => {
    const calls: string[] = []
    const queue = new PromptQueue(async (prompt) => {
      calls.push(prompt)
    })

    const prompt = "  inspect this:\n    indented example\n"
    queue.submit(prompt)
    await Bun.sleep(0)

    expect(calls).toEqual([prompt])
  })

  test("accepts prompts while paused and dispatches them in FIFO order after resume", async () => {
    const calls: string[] = []
    const accepted: Array<[string, boolean]> = []
    const queue = new PromptQueue(
      async (prompt) => {
        calls.push(prompt)
      },
      { onAccepted: (prompt, queued) => accepted.push([prompt, queued]) },
      true,
    )

    queue.submit("one")
    queue.submit("two")
    expect(calls).toEqual([])
    expect(queue.queued).toEqual(["one", "two"])

    queue.resume()
    await Bun.sleep(0)
    await Bun.sleep(0)

    expect(calls).toEqual(["one", "two"])
    expect(accepted).toEqual([
      ["one", true],
      ["two", true],
    ])
  })

  test("retains queued prompts when the runtime transport fails", async () => {
    const calls: string[] = []
    const gate = deferred()
    const queue = new PromptQueue(async (prompt) => {
      calls.push(prompt)
      await gate.promise
    })

    queue.submit("active")
    queue.submit("keep one")
    queue.fail(new Error("bridge closed"))
    queue.resume()
    queue.submit("keep two")
    gate.reject(new Error("bridge closed"))
    await Bun.sleep(0)

    expect(calls).toEqual(["active"])
    expect(queue.queued).toEqual(["keep one", "keep two"])
  })
})
