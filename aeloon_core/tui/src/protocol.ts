export type JsonObject = Record<string, unknown>

export interface SessionRecord extends JsonObject {
  created_at?: string
  final_content?: string | null
  tools_used?: string[]
  usage?: JsonObject
  user_prompt?: string
}

export interface WorkerRunSnapshot extends JsonObject {
  action?: string
  cancel_requested?: boolean
  created_at?: string
  duration_ms?: number | null
  objective?: string
  run_id?: string
  run_sequence?: number
  source_run_id?: string | null
  status?: string
  summary?: string | null
  usage?: JsonObject
  waiting_question?: string | null
}

export interface WorkerDefinitionSnapshot extends JsonObject {
  description?: string
  digest?: string
  id?: string
  source?: string
}

export interface WorkerSnapshot extends JsonObject {
  current_step?: string | null
  definition?: WorkerDefinitionSnapshot
  latest_run?: WorkerRunSnapshot | null
  label?: string
  phase?: string
  phases?: string[]
  run_count?: number
  runs?: WorkerRunSnapshot[]
  status?: string
  timeline?: JsonObject[]
  timeline_available?: boolean
  todo_completed?: number | null
  todo_total?: number | null
  worker_id?: string
  worker_type_id?: string
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
