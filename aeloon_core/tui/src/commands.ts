export interface CommandDefinition {
  description: string
  name: string
  usage: string
}

export interface ParsedCommand {
  args: string[]
  name: string
  rawArgs: string
}

export const COMMANDS: CommandDefinition[] = [
  { name: "help", usage: "/help", description: "Show commands and shortcuts" },
  { name: "workers", usage: "/workers", description: "Refresh the Worker strip" },
  { name: "worker", usage: "/worker <label>", description: "Open one Worker detail view" },
  {
    name: "verbosity",
    usage: "/verbosity [compact|verbose]",
    description: "Change transcript density",
  },
  { name: "logs", usage: "/logs [on|off|detail]", description: "Open gateway diagnostics" },
  { name: "new", usage: "/new", description: "Start a fresh session" },
  { name: "sessions", usage: "/sessions", description: "List resumable sessions" },
  { name: "resume", usage: "/resume <session>", description: "Resume a saved session" },
  {
    name: "worker-types",
    usage: "/worker-types",
    description: "List available Worker types",
  },
  {
    name: "spawn",
    usage: "/spawn <worker-type> <objective>",
    description: "Spawn a Worker session",
  },
  { name: "cancel", usage: "/cancel [run]", description: "Cancel the selected Worker" },
  {
    name: "resume-worker",
    usage: "/resume-worker <response…>",
    description: "Answer the selected waiting Worker",
  },
  { name: "cancel-turn", usage: "/cancel-turn", description: "Cancel the active Master turn" },
  { name: "master", usage: "/master", description: "Return to the Master transcript" },
  { name: "clear", usage: "/clear", description: "Clear the visible Master transcript" },
  { name: "quit", usage: "/quit", description: "Exit Aeloon" },
]

export function parseCommand(value: string): ParsedCommand | undefined {
  const clean = value.trim()
  if (!clean.startsWith("/")) return undefined
  const withoutSlash = clean.slice(1)
  const firstSpace = withoutSlash.search(/\s/)
  const name = (firstSpace < 0 ? withoutSlash : withoutSlash.slice(0, firstSpace)).toLowerCase()
  const rawArgs = firstSpace < 0 ? "" : withoutSlash.slice(firstSpace).trim()
  return { args: splitArguments(rawArgs), name, rawArgs }
}

export function commandSuggestions(value: string): CommandDefinition[] {
  const clean = value.trimStart()
  if (!clean.startsWith("/") || clean.includes(" ") || clean.includes("\n")) return []
  const query = clean.slice(1).toLowerCase()
  const prefixMatches = COMMANDS.filter((command) => command.name.startsWith(query))
  if (prefixMatches.length) return prefixMatches.slice(0, 6)
  return COMMANDS.filter((command) => command.description.toLowerCase().includes(query)).slice(0, 6)
}

function splitArguments(value: string): string[] {
  const result: string[] = []
  let current = ""
  let quote: '"' | "'" | undefined
  let escaped = false
  for (const character of value) {
    if (escaped) {
      current += character
      escaped = false
      continue
    }
    if (character === "\\" && quote === '"') {
      escaped = true
      continue
    }
    if (quote) {
      if (character === quote) quote = undefined
      else current += character
      continue
    }
    if (character === '"' || character === "'") {
      quote = character
      continue
    }
    if (/\s/.test(character)) {
      if (current) result.push(current)
      current = ""
      continue
    }
    current += character
  }
  if (current) result.push(current)
  return result
}
