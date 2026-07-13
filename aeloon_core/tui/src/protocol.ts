export type JsonObject = Record<string, unknown>

export interface SessionRecord extends JsonObject {
  created_at?: string
  final_content?: string | null
  tools_used?: string[]
  usage?: JsonObject
  user_prompt?: string
}

export interface WorkerRunSnapshot extends JsonObject {
  created_at?: string
  duration_ms?: number | null
  goal?: string
  run_id?: string
  run_sequence?: number
  status?: string
  summary?: string | null
  usage?: JsonObject
}

export interface WorkerSnapshot extends JsonObject {
  current_step?: string | null
  latest_run?: WorkerRunSnapshot | null
  label?: string
  phase?: string
  phases?: string[]
  profile?: JsonObject
  profile_id?: string
  run_count?: number
  runs?: WorkerRunSnapshot[]
  status?: string
  timeline?: JsonObject[]
  timeline_available?: boolean
  todo_completed?: number | null
  todo_total?: number | null
  worker_id?: string
}

export interface ReadySnapshot extends JsonObject {
  history?: SessionRecord[]
  model?: string
  session_id?: string
  workers?: WorkerSnapshot[]
  workspace?: string
}

export type BridgeEnvelope =
  | { type: "ready"; payload: ReadySnapshot }
  | { type: "event"; event: string; payload: JsonObject }
  | {
      type: "response"
      request_id: string
      ok: boolean
      result?: unknown
      error?: { code?: string; message: string }
    }

export interface BridgeCommand {
  type: "command"
  command: string
  request_id: string
  payload?: JsonObject
}

export function isBridgeEnvelope(value: unknown): value is BridgeEnvelope {
  if (!value || typeof value !== "object") return false
  const candidate = value as Record<string, unknown>
  if (candidate.type === "ready") return isObject(candidate.payload)
  if (candidate.type === "event") {
    return typeof candidate.event === "string" && isObject(candidate.payload)
  }
  return (
    candidate.type === "response" &&
    typeof candidate.request_id === "string" &&
    typeof candidate.ok === "boolean"
  )
}

export function isObject(value: unknown): value is JsonObject {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}
