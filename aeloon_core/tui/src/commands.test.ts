import { describe, expect, test } from "bun:test"
import { COMMANDS, commandSuggestions, parseCommand } from "./commands"

describe("slash commands", () => {
  test("suggests commands from a prefix", () => {
    expect(commandSuggestions("/ver").map((item) => item.name)).toEqual(["verbosity"])
    expect(commandSuggestions("hello")).toEqual([])
  })

  test("exposes the v2 Worker type catalog", () => {
    expect(COMMANDS.map((item) => item.name)).toContain("worker-types")
  })

  test("preserves a quoted multiline-friendly task", () => {
    expect(parseCommand('/spawn coding "fix the queue"')).toEqual({
      args: ["coding", "fix the queue"],
      name: "spawn",
      rawArgs: 'coding "fix the queue"',
    })
  })
})
