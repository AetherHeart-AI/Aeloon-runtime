import type {
  BridgeEnvelope,
  JsonObject,
  ReadySnapshot,
  SessionRecord,
  WorkerRunSnapshot,
  WorkerSnapshot,
} from "./protocol"
import { isObject } from "./protocol"

export type Verbosity = "compact" | "verbose"
export type FocusTarget = "composer" | "transcript"
export type View =
  | { kind: "master" }
  | { kind: "worker"; workerId: string }
  | { kind: "logs" }

export type TimelineKind =
  | "aggregate"
  | "assistant"
  | "error"
  | "guard"
  | "lifecycle"
  | "log"
  | "narration"
  | "step"
  | "summary"
  | "system"
  | "thinking"
  | "tool"
  | "user"

export type TimelineStatus = "running" | "done" | "partial" | "failed" | "cancelled"
export type SignalLevel = "high" | "low"

export interface TimelineItem {
  aggregateCounts?: Record<string, number>
  aggregateId?: string
  body?: string
  collapsed?: boolean
  detail?: string
  id: string
  kind: TimelineKind
  metrics?: string
  primary?: string
  rawDetail?: string
  resultPreview?: string
  signal: SignalLevel
  status?: TimelineStatus
  title: string
  toolName?: string
  ts?: string
  verb?: string
  workerLabel?: string
}

export interface TurnGroup {
  answer?: TimelineItem
  collapsed: boolean
  id: string
  process: TimelineItem[]
  processSummary: string
  summary?: TimelineItem
  user?: TimelineItem
}

export interface GatewayLog {
  detail?: JsonObject
  id: string
  level: string
  message: string
  source: string
  ts?: string
}

export interface WorkerInfo {
  aggregateId?: string
  currentStep?: string
  durationMs?: number
  goal?: string
  id: string
  label: string
  lastRevision: Record<string, number>
  phase: string
  phases: string[]
  profileId: string
  report?: string
  runId?: string
  runSequence?: number
  startedAt?: number
  status: string
  timeline: TimelineItem[]
  todoCompleted?: number
  todoTotal?: number
  unread: number
  usage: JsonObject
}

interface PendingBlock {
  aggregateId?: string
  arguments?: unknown
  itemId?: string
  name?: string
  startedAt?: string
  type: string
  workerLabel?: string
}

export interface AppState {
  bridgeError?: string
  bridgeReady: boolean
  focus: FocusTarget
  gatewayLogs: GatewayLog[]
  lastTurnDurationMs?: number
  logDetail: boolean
  masterAggregateId?: string
  masterTimeline: TimelineItem[]
  model: string
  pendingBlocks: Record<string, PendingBlock>
  pinToBottom: boolean
  queuedPrompts: string[]
  running: boolean
  sequence: number
  sessionId: string
  sessionSwitching: boolean
  turnStartedAt?: number
  turnProcessCollapsed: Record<string, boolean>
  turnToolNames: string[]
  usage: JsonObject
  verbosity: Verbosity
  view: View
  workerOrder: string[]
  workers: Record<string, WorkerInfo>
  workspace: string
}

const LOW_SIGNAL_TOOLS = new Set([
  "discover_profiles",
  "glob",
  "grep",
  "inspect_worker",
  "list_workers",
  "read",
  "skill",
  "webfetch",
  "websearch",
])

const TERMINAL_WORKER_STATES = new Set([
  "archived",
  "cancelled",
  "completed",
  "failed",
  "partial",
  "timed_out",
])

const LOG_LIMIT = 1_000
const RAW_DETAIL_LIMIT = 1_200

export function createAppState(): AppState {
  return {
    bridgeReady: false,
    focus: "composer",
    gatewayLogs: [],
    logDetail: false,
    masterTimeline: [],
    model: "",
    pendingBlocks: {},
    pinToBottom: true,
    queuedPrompts: [],
    running: false,
    sequence: 0,
    sessionId: "",
    sessionSwitching: false,
    turnToolNames: [],
    turnProcessCollapsed: {},
    usage: {},
    verbosity: "compact",
    view: { kind: "master" },
    workerOrder: [],
    workers: {},
    workspace: "",
  }
}

export function applyEnvelope(state: AppState, envelope: BridgeEnvelope): void {
  if (envelope.type === "ready") {
    hydrateReady(state, envelope.payload)
    return
  }
  if (envelope.type === "response") return
  applyEvent(state, envelope.event, envelope.payload)
}

export function hydrateReady(state: AppState, snapshot: ReadySnapshot): void {
  state.bridgeError = undefined
  state.bridgeReady = true
  state.workspace = stringValue(snapshot.workspace)
  state.model = stringValue(snapshot.model)
  state.sessionId = stringValue(snapshot.session_id)
  state.masterTimeline = []
  state.masterAggregateId = undefined
  state.pendingBlocks = {}
  state.running = false
  state.turnStartedAt = undefined
  state.turnToolNames = []
  state.turnProcessCollapsed = {}
  state.usage = {}
  state.gatewayLogs = []
  state.workers = {}
  state.workerOrder = []
  state.view = { kind: "master" }

  for (const record of arrayOfObjects(snapshot.history) as SessionRecord[]) {
    hydrateSessionRecord(state, record)
  }
  for (const worker of arrayOfObjects(snapshot.workers) as WorkerSnapshot[]) {
    upsertWorkerSnapshot(state, worker)
  }
}

export function applyEvent(state: AppState, event: string, payload: JsonObject): void {
  const eventSessionId = stringValue(payload.session_id)
  if (eventSessionId && state.sessionId && eventSessionId !== state.sessionId) return

  if (event === "chat.turn.start") {
    settlePendingBlocks(state, "partial")
    state.running = true
    state.turnStartedAt = Date.now()
    state.lastTurnDurationMs = undefined
    state.turnToolNames = []
    state.usage = {}
    state.masterAggregateId = undefined
    return
  }
  if (event === "chat.status" || event === "chat.worker.heartbeat") {
    if (event === "chat.worker.heartbeat") updateWorkerHeartbeat(state, payload)
    return
  }
  if (event === "chat.block.add") {
    onBlockAdd(state, payload)
    return
  }
  if (event === "chat.block.delta") {
    onBlockDelta(state, payload)
    return
  }
  if (event === "chat.block.update") {
    onBlockUpdate(state, payload)
    return
  }
  if (event === "chat.usage" || event === "chat.llm.response") {
    const usage = objectValue(
      event === "chat.usage"
        ? payload.usage
        : payload.aggregate_usage ?? payload.usage,
    )
    if (usage) state.usage = { ...usage }
    return
  }
  if (event === "chat.guard.decision" || event === "chat.profile.delegate.guard") {
    addGuard(state, payload)
    return
  }
  if (event === "chat.worker.guard") {
    addWorkerGuard(state, payload)
    return
  }
  if (event === "chat.worker.lifecycle") {
    onWorkerLifecycle(state, payload)
    return
  }
  if (event === "chat.worker.activity") {
    onWorkerActivity(state, payload)
    return
  }
  if (event === "chat.worker.tool.result") {
    onWorkerToolResult(state, payload)
    return
  }
  if (event === "log.entry") {
    onLogEntry(state, payload)
    return
  }
  if (event === "chat.turn.end") {
    onTurnEnd(state, payload)
    return
  }
  if (event.startsWith("chat.profile.")) {
    onProfileEvent(state, event, payload)
  }
}

export function appendUserPrompt(state: AppState, prompt: string): void {
  addMasterItem(state, {
    body: prompt,
    kind: "user",
    signal: "high",
    title: "YOU",
  })
}

export function appendSystemNotice(
  state: AppState,
  title: string,
  body: string,
  kind: "system" | "error" = "system",
): void {
  addMasterItem(state, {
    body,
    kind,
    signal: "high",
    status: kind === "error" ? "failed" : undefined,
    title,
  })
}

export function clearMasterTimeline(state: AppState): void {
  state.masterTimeline = []
  state.masterAggregateId = undefined
  state.pendingBlocks = {}
}

export function setQueuedPrompts(state: AppState, prompts: string[]): void {
  state.queuedPrompts = [...prompts]
}

export function markTurnDispatching(state: AppState): void {
  state.running = true
  state.turnStartedAt = Date.now()
}

export function markTurnFailed(state: AppState, message: string): void {
  settlePendingBlocks(state, "failed")
  state.running = false
  state.turnStartedAt = undefined
  addMasterItem(state, {
    body: oneLine(message, 240),
    kind: "error",
    signal: "high",
    status: "failed",
    title: "TURN FAILED",
  })
}

export function markTurnCancelled(state: AppState): void {
  settlePendingBlocks(state, "cancelled")
  state.running = false
  state.turnStartedAt = undefined
  addMasterItem(state, {
    body: "Cancelled by operator",
    kind: "summary",
    signal: "high",
    status: "cancelled",
    title: "TURN CANCELLED",
  })
}

export function markBridgeFailed(state: AppState, message: string): void {
  state.bridgeError = oneLine(message, 180)
  state.bridgeReady = false
  state.running = false
  state.turnStartedAt = undefined
  settlePendingBlocks(state, "failed")
  addMasterItem(state, {
    body: state.bridgeError,
    kind: "error",
    signal: "high",
    status: "failed",
    title: "RUNTIME FAILED",
  })
}

export function setView(state: AppState, view: View): void {
  state.view = view
  if (view.kind === "worker") {
    const worker = state.workers[view.workerId]
    if (worker) worker.unread = 0
  }
}

export function cycleView(state: AppState, direction = 1): void {
  const available: View[] = [
    { kind: "master" },
    ...state.workerOrder.map((workerId) => ({ kind: "worker", workerId }) as View),
  ]
  const current = available.findIndex((candidate) => sameView(candidate, state.view))
  const next = (Math.max(0, current) + direction + available.length) % available.length
  setView(state, available[next] ?? { kind: "master" })
}

export function setVerbosity(state: AppState, verbosity: Verbosity): void {
  state.verbosity = verbosity
}

export function toggleVerbosity(state: AppState): void {
  state.verbosity = state.verbosity === "compact" ? "verbose" : "compact"
}

export function setFocus(state: AppState, focus: FocusTarget): void {
  state.focus = focus
}

export function setPinned(state: AppState, pinned: boolean): void {
  state.pinToBottom = pinned
}

export function visibleMasterItems(state: AppState): TimelineItem[] {
  const visible = state.masterTimeline.filter((item) => {
    if (state.verbosity === "compact") {
      return item.kind === "aggregate" || item.kind === "thinking" || (item.signal === "high" && item.kind !== "log")
    }
    return item.kind !== "aggregate"
  })
  if (state.verbosity === "verbose") return visible
  return visible.map((item) => (item.rawDetail ? { ...item, rawDetail: undefined } : item))
}

export function visibleMasterTurns(state: AppState): TurnGroup[] {
  const groups: TurnGroup[] = []
  let current: TurnGroup | undefined
  for (const item of visibleMasterItems(state)) {
    if (item.kind === "user") {
      current = emptyTurn(item.id, item)
      groups.push(current)
      continue
    }
    if (!current || current.summary) {
      current = emptyTurn(item.id)
      groups.push(current)
    }
    if (item.kind === "summary") {
      current.summary = item
    } else {
      current.process.push(item)
    }
  }
  for (const group of groups) {
    for (let index = group.process.length - 1; index >= 0; index -= 1) {
      if (group.process[index]?.kind !== "assistant") continue
      group.answer = group.process[index]
      group.process = group.process
        .filter((_, candidateIndex) => candidateIndex !== index)
        .map((candidate) => candidate.kind === "assistant"
          ? { ...candidate, kind: "narration", title: "NARRATION" }
          : candidate)
      break
    }
    const failed = group.process.filter((item) => item.status === "failed").length
    const defaultCollapsed = Boolean(group.summary) && failed === 0
    group.collapsed = state.turnProcessCollapsed[group.id] ?? defaultCollapsed
    group.processSummary = formatProcessSummary(group.process, group.summary)
  }
  return groups
}

export function toggleTurnProcess(state: AppState, turnId?: string): void {
  const groups = visibleMasterTurns(state).filter((group) => group.process.length)
  const group = turnId ? groups.find((candidate) => candidate.id === turnId) : groups.at(-1)
  if (!group) return
  state.turnProcessCollapsed[group.id] = !group.collapsed
}

export function toggleTimelineItem(state: AppState, itemId: string): void {
  const item = state.masterTimeline.find((candidate) => candidate.id === itemId)
    ?? Object.values(state.workers).flatMap((worker) => worker.timeline).find((candidate) => candidate.id === itemId)
  if (item) item.collapsed = !item.collapsed
}

export function visibleWorkerItems(state: AppState, workerId: string): TimelineItem[] {
  const worker = state.workers[workerId]
  if (!worker) return []
  const visible = worker.timeline.filter((item) => {
    if (state.verbosity === "compact") {
      return item.kind === "aggregate" || item.signal === "high"
    }
    return item.kind !== "aggregate"
  })
  if (state.verbosity === "verbose") return visible
  return visible.map((item) => (item.rawDetail ? { ...item, rawDetail: undefined } : item))
}

export function runningWorkers(state: AppState): WorkerInfo[] {
  return state.workerOrder
    .map((workerId) => state.workers[workerId])
    .filter((worker): worker is WorkerInfo => Boolean(worker) && !TERMINAL_WORKER_STATES.has(worker.status))
}

export function waitingSummary(state: AppState): string {
  const workers = runningWorkers(state)
  if (!workers.length) return ""
  const labels = workers.slice(0, 3).map((worker) => worker.label)
  const suffix = workers.length > labels.length ? ` +${workers.length - labels.length}` : ""
  return `Waiting on ${workers.length} Worker${workers.length === 1 ? "" : "s"} · ${labels.join(", ")}${suffix}`
}

export function applyCommandResult(state: AppState, command: string, result: unknown): void {
  if (command === "new_session" || command === "resume_session") {
    if (isObject(result)) hydrateReady(state, result as ReadySnapshot)
    return
  }
  if (command === "list_workers") {
    for (const worker of arrayOfObjects(result) as WorkerSnapshot[]) upsertWorkerSnapshot(state, worker)
    addMasterItem(state, {
      body: `${state.workerOrder.length} Worker${state.workerOrder.length === 1 ? "" : "s"} available in the strip.`,
      kind: "system",
      signal: "high",
      title: "WORKERS",
    })
    return
  }
  if (command === "inspect_worker") {
    hydrateWorkerSnapshot(state, result)
    return
  }
  if (command === "spawn_worker") {
    if (isObject(result)) upsertWorkerSnapshot(state, result as WorkerSnapshot)
    return
  }
  if (command === "cancel_worker" || command === "resume_worker") {
    if (isObject(result)) {
      updateWorkerFromRun(state, result as WorkerRunSnapshot)
      if (command === "resume_worker") {
        const workerId = stringValue(result.worker_id)
        const worker = workerId ? state.workers[workerId] : undefined
        const action = stringValue(result.action)
        const labels: Record<string, string> = {
          already_running: "Already running",
          continued: "Continuation scheduled",
          restarted: "Clean retry scheduled",
          scheduled: "Run scheduled",
        }
        if (action) {
          addMasterItem(state, {
            body: [worker?.label, labels[action] ?? oneLine(action, 80)]
              .filter(Boolean)
              .join(" · "),
            kind: "lifecycle",
            signal: "high",
            status: action === "already_running" ? "running" : "partial",
            title: "WORKER RECOVERY",
          })
        }
      }
    }
    return
  }
  if (command === "list_sessions") {
    const sessions = arrayOfObjects(result)
    const body = sessions.length
      ? sessions
          .slice(0, 12)
          .map((item) => {
            const id = stringValue(item.session_id)
            const turns = numberValue(item.turns) ?? 0
            const title = oneLine(stringValue(item.title), 56)
            return `${id} · ${turns} turn${turns === 1 ? "" : "s"}${title ? ` · ${title}` : ""}`
          })
          .join("\n")
      : "No saved sessions."
    addMasterItem(state, { body, kind: "system", signal: "high", title: "SESSIONS" })
    return
  }
  if (command === "discover_profiles") {
    const profiles = arrayOfObjects(result)
    const names = profiles
      .map((item) => objectValue(item.profile)?.profile_id ?? item.profile_id)
      .map(stringValue)
      .filter(Boolean)
    addMasterItem(state, {
      body: names.length ? names.join(" · ") : "No active Worker profiles.",
      kind: "system",
      signal: "high",
      title: "PROFILES",
    })
  }
}

export function hydrateWorkerSnapshot(
  state: AppState,
  snapshot: unknown,
): WorkerInfo | undefined {
  return isObject(snapshot) ? upsertWorkerSnapshot(state, snapshot as WorkerSnapshot) : undefined
}

export function applyCommandError(state: AppState, command: string, message: string): void {
  if (command === "prompt") state.running = false
  addMasterItem(state, {
    body: oneLine(message, 280),
    kind: "error",
    signal: "high",
    status: "failed",
    title: command.replaceAll("_", " ").toUpperCase(),
  })
}

function hydrateSessionRecord(state: AppState, record: SessionRecord): void {
  const prompt = stringValue(record.user_prompt)
  const answer = stringValue(record.final_content)
  if (prompt) {
    addMasterItem(state, { body: prompt, kind: "user", signal: "high", title: "YOU" })
  }
  if (answer) {
    addMasterItem(state, {
      body: answer,
      kind: "assistant",
      signal: "high",
      status: "done",
      title: "AELOON",
    })
  }
  const usage = objectValue(record.usage)
  const tools = Array.isArray(record.tools_used) ? record.tools_used.filter(isString) : []
  const summary = formatTurnSummary(undefined, usage ?? {}, tools)
  if (summary !== "Completed") {
    addMasterItem(state, { body: summary, kind: "summary", signal: "high", title: "TURN" })
  }
}

function onBlockAdd(state: AppState, payload: JsonObject): void {
  const block = objectValue(payload.block)
  if (!block) return
  const id = stringValue(block.id)
  const type = stringValue(block.type)
  if (!id || !type) return
  if (type === "reasoning") {
    const item = addMasterItem(state, {
      body: stringValue(block.content),
      collapsed: true,
      kind: "thinking",
      signal: "high",
      status: "running",
      title: "THINKING",
      ts: stringValue(block.created_at),
    })
    state.pendingBlocks[id] = {
      itemId: item.id,
      startedAt: stringValue(block.created_at),
      type,
    }
    return
  }

  if (type === "text") {
    const item = addMasterItem(state, {
      body: stringValue(block.content),
      kind: "narration",
      signal: "high",
      status: "running",
      title: "NARRATION",
      ts: stringValue(block.created_at),
    })
    state.pendingBlocks[id] = { itemId: item.id, type }
    return
  }
  if (type !== "tool_call") return

  const name = stringValue(block.name) || "tool"
  const workerLabel = stringValue(block.subagent_label)
  const signal: SignalLevel = LOW_SIGNAL_TOOLS.has(name) ? "low" : "high"
  let aggregateId: string | undefined
  if (signal === "low") aggregateId = bumpMasterAggregate(state, name)
  const item = addMasterItem(state, {
    aggregateId,
    body: summarizeToolArguments(name, block.arguments),
    collapsed: true,
    kind: "tool",
    primary: summarizeToolArguments(name, block.arguments),
    rawDetail: safeJson(block.arguments),
    signal,
    status: "running",
    title: name === "todowrite" ? "TODO" : "TOOL",
    toolName: name,
    ts: stringValue(block.created_at),
    verb: toolVerb(name),
    workerLabel: workerLabel || undefined,
  })
  state.pendingBlocks[id] = {
    aggregateId,
    arguments: block.arguments,
    itemId: item.id,
    name,
    startedAt: stringValue(block.created_at),
    type,
    workerLabel: workerLabel || undefined,
  }
  if (!state.turnToolNames.includes(name)) state.turnToolNames.push(name)
}

function onBlockDelta(state: AppState, payload: JsonObject): void {
  const id = stringValue(payload.block_id)
  const block = state.pendingBlocks[id]
  if (!block || (block.type !== "text" && block.type !== "reasoning") || !block.itemId) return
  const item = findMasterItem(state, block.itemId)
  if (item) item.body = `${item.body ?? ""}${stringValue(payload.delta)}`
}

function onBlockUpdate(state: AppState, payload: JsonObject): void {
  const id = stringValue(payload.block_id)
  const block = state.pendingBlocks[id]
  if (!block) return
  const patch = objectValue(payload.patch)
  if (!patch) return

  const item = block.itemId ? findMasterItem(state, block.itemId) : undefined
  if (block.type === "text" || block.type === "reasoning") {
    if (item && "content" in patch) item.body = stringValue(patch.content)
    if (item && stringValue(patch.status) === "done") {
      item.status = "done"
      if (block.type === "text") {
        item.kind = "assistant"
        item.title = "AELOON"
      }
      if (block.type === "reasoning") item.metrics = formatDuration(numberValue(patch.duration_ms))
    }
    return
  }
  if (block.type !== "tool_call") return

  const name = block.name ?? "tool"
  const failed = isFailedStatus(patch.status) || toolResultFailed(patch.result)
  if (item) {
    item.metrics = summarizeToolResult(
      name,
      patch.result,
      block.arguments,
      failed,
      numberValue(patch.duration_ms),
    )
    item.body = item.metrics
    item.resultPreview = truncateText(stringValue(patch.result), RAW_DETAIL_LIMIT)
    item.collapsed = !failed
    item.status = failed ? "failed" : "done"
    if (failed && item.signal === "low") {
      item.signal = "high"
      removeFromAggregate(state.masterTimeline, item.aggregateId, name)
      state.masterAggregateId = undefined
    }
  }
}

function addGuard(state: AppState, payload: JsonObject): void {
  const action = stringValue(payload.action) || "decision"
  const source = stringValue(payload.source) || "guard"
  const reason = stringValue(payload.event)
  addMasterItem(state, {
    body: [guardActionLabel(action), reason].filter(Boolean).join(" · "),
    kind: "guard",
    signal: "high",
    status: action === "finalize" ? "failed" : "partial",
    title: `GUARD · ${source}`,
    ts: stringValue(payload.ts),
    workerLabel: stringValue(payload.subagent_label) || undefined,
  })
}

function addWorkerGuard(state: AppState, payload: JsonObject): void {
  const worker = ensureWorker(state, payload)
  if (!workerEventMatchesCurrentRun(worker, payload)) return
  const action = stringValue(payload.action) || "decision"
  const reason = stringValue(payload.event)
  const item: Omit<TimelineItem, "id"> = {
    body: [guardActionLabel(action), reason].filter(Boolean).join(" · "),
    kind: "guard",
    signal: "high",
    status: action === "finalize" ? "failed" : "partial",
    title: "GUARD",
    ts: stringValue(payload.ts),
  }
  appendWorkerItem(state, worker, item)
  addMasterItem(state, { ...item, workerLabel: worker.label })
  markWorkerUnread(state, worker)
}

function onWorkerLifecycle(state: AppState, payload: JsonObject): void {
  const worker = ensureWorker(state, payload)
  if (
    !beginWorkerRun(
      worker,
      stringValue(payload.run_id),
      numberValue(payload.run_sequence),
    )
  ) return
  const phase = stringValue(payload.phase) || stringValue(payload.status) || "running"
  const status = phase === "created" ? stringValue(payload.status) || "queued" : phase
  if (!advanceWorkerStatus(worker, normalizeWorkerStatus(status))) return
  worker.durationMs = numberValue(payload.duration_ms) ?? worker.durationMs
  if (worker.status === "running" && !worker.startedAt) worker.startedAt = Date.now()
  if (TERMINAL_WORKER_STATES.has(worker.status)) {
    worker.aggregateId = undefined
    worker.currentStep = undefined
    worker.phase = friendlyPhase(worker.status, [])
    if (worker.phases.at(-1) !== worker.phase) worker.phases.push(worker.phase)
  }

  const timelineState = timelineStatus(worker.status)
  const body = lifecycleBody(worker.status, worker.durationMs)
  appendWorkerItem(state, worker, {
    body,
    kind: "lifecycle",
    signal: "high",
    status: timelineState,
    title: "RUN",
    ts: stringValue(payload.ts),
  })
  addMasterItem(state, {
    body,
    kind: "lifecycle",
    signal: "high",
    status: timelineState,
    title: phase === "created" ? "WORKER DISPATCHED" : "WORKER",
    ts: stringValue(payload.ts),
    workerLabel: worker.label,
  })
  markWorkerUnread(state, worker)
}

function onWorkerActivity(state: AppState, payload: JsonObject): void {
  const worker = ensureWorker(state, payload)
  if (!workerEventMatchesCurrentRun(worker, payload)) return
  if (TERMINAL_WORKER_STATES.has(worker.status)) return
  const label = stringValue(payload.label) || worker.label
  const revision = numberValue(payload.revision) ?? 0
  if (revision && revision <= (worker.lastRevision[label] ?? 0)) return
  if (revision) worker.lastRevision[label] = revision

  const phase = stringValue(payload.phase)
  const toolNames = arrayOfStrings(payload.tool_names)
  worker.phase = friendlyPhase(phase, toolNames)
  if (worker.phase && worker.phases.at(-1) !== worker.phase) worker.phases.push(worker.phase)
  const currentStep = oneLine(stringValue(payload.current_step), 120)
  worker.todoCompleted = numberValue(payload.todo_completed) ?? worker.todoCompleted
  worker.todoTotal = numberValue(payload.todo_total) ?? worker.todoTotal
  if (currentStep && currentStep !== worker.currentStep) {
    worker.currentStep = currentStep
    appendWorkerItem(state, worker, {
      body: formatProgress(currentStep, worker.todoCompleted, worker.todoTotal),
      kind: "step",
      signal: "high",
      status: "running",
      title: "CURRENT",
      ts: stringValue(payload.ts),
    })
    markWorkerUnread(state, worker)
  }
}

function onWorkerToolResult(state: AppState, payload: JsonObject): void {
  const worker = ensureWorker(state, payload)
  if (!workerEventMatchesCurrentRun(worker, payload)) return
  if (TERMINAL_WORKER_STATES.has(worker.status)) return
  const name = stringValue(payload.tool_name) || "tool"
  const failed = stringValue(payload.status) !== "done"
  const signal: SignalLevel = failed || !LOW_SIGNAL_TOOLS.has(name) ? "high" : "low"
  const display = workerToolDisplay(name, objectValue(payload.metrics) ?? {}, numberValue(payload.duration_ms))
  const body = [display.primary, display.metrics].filter(Boolean).join(" · ")
  let aggregateId: string | undefined
  if (signal === "low") aggregateId = bumpWorkerAggregate(state, worker, name)
  appendWorkerItem(state, worker, {
    aggregateId,
    body,
    collapsed: !failed,
    kind: "tool",
    metrics: display.metrics,
    primary: display.primary,
    signal,
    status: failed ? "failed" : "done",
    title: "TOOL",
    toolName: name,
    ts: stringValue(payload.ts),
    verb: toolVerb(name),
  })
  if (signal === "high") {
    addMasterItem(state, {
      body,
      collapsed: !failed,
      kind: "tool",
      metrics: display.metrics,
      primary: display.primary,
      signal: "high",
      status: failed ? "failed" : "done",
      title: "TOOL",
      toolName: name,
      ts: stringValue(payload.ts),
      verb: toolVerb(name),
      workerLabel: worker.label,
    })
  }
  markWorkerUnread(state, worker)
}

function updateWorkerHeartbeat(state: AppState, payload: JsonObject): void {
  const worker = ensureWorker(state, payload)
  if (!workerEventMatchesCurrentRun(worker, payload)) return
  if (!TERMINAL_WORKER_STATES.has(worker.status)) worker.status = stringValue(payload.status) || "running"
  const elapsed = numberValue(payload.elapsed_ms)
  if (elapsed !== undefined) worker.durationMs = elapsed
}

function onLogEntry(state: AppState, payload: JsonObject): void {
  const log: GatewayLog = {
    detail: objectValue(payload.detail),
    id: nextId(state, "log"),
    level: stringValue(payload.level) || "INFO",
    message: oneLine(stringValue(payload.message), 320),
    source: stringValue(payload.source) || "gateway",
    ts: stringValue(payload.ts) || undefined,
  }
  state.gatewayLogs.push(log)
  if (state.gatewayLogs.length > LOG_LIMIT) state.gatewayLogs.splice(0, state.gatewayLogs.length - LOG_LIMIT)
  addMasterItem(state, {
    body: log.message,
    detail: state.logDetail && log.detail ? safeJson(log.detail) : undefined,
    kind: "log",
    signal: "low",
    title: `${log.level} · ${log.source}`,
    ts: log.ts,
  })
}

function onTurnEnd(state: AppState, payload: JsonObject): void {
  state.running = false
  state.turnStartedAt = undefined
  state.lastTurnDurationMs = numberValue(payload.duration_ms)
  const pending = Object.values(state.pendingBlocks)
  for (const block of pending) {
    if (block.type === "text" && block.itemId) {
      const item = findMasterItem(state, block.itemId)
      if (item) item.status = "done"
    }
    if (block.type === "reasoning" && block.itemId) {
      const item = findMasterItem(state, block.itemId)
      if (item) item.status = "done"
    }
  }
  const finalTextBlock = pending.filter((block) => block.type === "text" && block.itemId).at(-1)
  const finalText = finalTextBlock?.itemId ? findMasterItem(state, finalTextBlock.itemId) : undefined
  if (finalText) {
    finalText.kind = "assistant"
    finalText.title = "AELOON"
  }
  addMasterItem(state, {
    body: formatTurnSummary(state.lastTurnDurationMs, state.usage, state.turnToolNames),
    kind: "summary",
    signal: "high",
    status: "done",
    title: "TURN COMPLETE",
    ts: stringValue(payload.ts),
  })
  state.pendingBlocks = {}
  state.masterAggregateId = undefined
}

function onProfileEvent(state: AppState, event: string, payload: JsonObject): void {
  if (event === "chat.profile.pinned") return
  if (event === "chat.profile.route") {
    addMasterItem(state, {
      body: `${stringValue(payload.agent_id) || "agent"}${payload.fallback_used ? " · fallback route" : ""}`,
      kind: "lifecycle",
      signal: "high",
      status: "running",
      title: "AGENT DISPATCHED",
      ts: stringValue(payload.ts),
    })
    return
  }
  if (event === "chat.profile.handoff") {
    addMasterItem(state, {
      body: `${stringValue(payload.from_agent_id) || "agent"} → ${stringValue(payload.recommended_agent_id) || "coordinator"} · ${oneLine(stringValue(payload.summary), 160)}`,
      kind: "lifecycle",
      signal: "high",
      status: "partial",
      title: "HANDOFF",
      ts: stringValue(payload.ts),
    })
    return
  }
  if (event === "chat.profile.delegate.start") {
    addMasterItem(state, {
      body: oneLine(stringValue(payload.task), 180),
      kind: "lifecycle",
      signal: "high",
      status: "running",
      title: "DELEGATED",
      ts: stringValue(payload.ts),
      workerLabel: stringValue(payload.label) || stringValue(payload.agent_id) || undefined,
    })
    return
  }
  if (event === "chat.profile.delegate.complete") {
    const completed = stringValue(payload.status) === "completed"
    addMasterItem(state, {
      body: oneLine(stringValue(payload.summary), 180),
      kind: "lifecycle",
      signal: "high",
      status: completed ? "done" : "failed",
      title: completed ? "DELEGATE COMPLETE" : "DELEGATE FAILED",
      ts: stringValue(payload.ts),
      workerLabel: stringValue(payload.label) || stringValue(payload.agent_id) || undefined,
    })
    return
  }
  if (event === "chat.profile.delegate.join") {
    addMasterItem(state, {
      body: `${numberValue(payload.succeeded) ?? 0}/${numberValue(payload.branch_count) ?? 0} joined${formatDuration(numberValue(payload.duration_ms)) ? ` · ${formatDuration(numberValue(payload.duration_ms))}` : ""}`,
      kind: "lifecycle",
      signal: "high",
      status: "done",
      title: "DELEGATES JOINED",
      ts: stringValue(payload.ts),
    })
    return
  }
  if (event === "chat.profile.completion") {
    addMasterItem(state, {
      body: stringValue(payload.agent_id) || "agent",
      kind: "lifecycle",
      signal: "high",
      status: "done",
      title: "AGENT COMPLETE",
      ts: stringValue(payload.ts),
    })
  }
}

function ensureWorker(state: AppState, payload: JsonObject): WorkerInfo {
  const id = stringValue(payload.worker_id) || `unknown-${stringValue(payload.profile_id) || "worker"}`
  let worker = state.workers[id]
  if (worker) return worker
  const profileId = stringValue(payload.profile_id) || "worker"
  worker = {
    id,
    label: workerLabel(profileId, id),
    lastRevision: {},
    phase: "queued",
    phases: [],
    profileId,
    runId: stringValue(payload.run_id) || undefined,
    runSequence: numberValue(payload.run_sequence),
    status: stringValue(payload.status) || "queued",
    timeline: [],
    unread: 0,
    usage: {},
  }
  state.workers[id] = worker
  state.workerOrder.push(id)
  return state.workers[id] ?? worker
}

function upsertWorkerSnapshot(state: AppState, snapshot: WorkerSnapshot): WorkerInfo {
  const id = stringValue(snapshot.worker_id) || "unknown-worker"
  const profile = objectValue(snapshot.profile)
  const profileId = stringValue(snapshot.profile_id) || stringValue(profile?.profile_id) || "worker"
  const worker = ensureWorker(state, {
    profile_id: profileId,
    status: snapshot.status,
    worker_id: id,
  })
  worker.profileId = profileId
  worker.label = workerLabel(profileId, id)
  const runs = Array.isArray(snapshot.runs)
    ? (arrayOfObjects(snapshot.runs) as WorkerRunSnapshot[])
    : []
  const latest = (objectValue(snapshot.latest_run) as WorkerRunSnapshot | undefined) ?? runs.at(-1)
  if (latest) {
    if (!mergeWorkerRun(worker, latest)) return worker
  } else {
    const snapshotStatus = stringValue(snapshot.status)
    if (snapshotStatus) advanceWorkerStatus(worker, normalizeWorkerStatus(snapshotStatus))
  }

  const phase = stringValue(snapshot.phase)
  if (phase) worker.phase = friendlyPhase(phase, [])
  const phases = arrayOfStrings(snapshot.phases).map((item) => friendlyPhase(item, []))
  if (phases.length) worker.phases = [...new Set(phases)]
  const currentStep = oneLine(stringValue(snapshot.current_step), 120)
  if (currentStep && !TERMINAL_WORKER_STATES.has(worker.status)) {
    worker.currentStep = currentStep
  } else if (TERMINAL_WORKER_STATES.has(worker.status)) {
    worker.currentStep = undefined
  }
  worker.todoCompleted = numberValue(snapshot.todo_completed) ?? worker.todoCompleted
  worker.todoTotal = numberValue(snapshot.todo_total) ?? worker.todoTotal

  const timeline = arrayOfObjects(snapshot.timeline)
  if (timeline.length || snapshot.timeline_available === true) {
    worker.timeline = timeline
      .map((row) => projectWorkerJournalRow(state, row))
      .filter((item): item is TimelineItem => Boolean(item))
  }
  return worker
}

function projectWorkerJournalRow(state: AppState, row: JsonObject): TimelineItem | undefined {
  const kind = stringValue(row.kind)
  const ts = stringValue(row.ts) || undefined
  if (kind === "tools") {
    const counts: Record<string, number> = {}
    const rawCounts = objectValue(row.tool_counts) ?? {}
    for (const [name, value] of Object.entries(rawCounts)) {
      const count = numberValue(value)
      if (count !== undefined && count > 0) counts[name] = count
    }
    return {
      aggregateCounts: counts,
      body: formatAggregate(counts),
      id: nextId(state, "worker-journal"),
      kind: "aggregate",
      signal: "high",
      status: "done",
      title: "ROUTINE ACTIVITY",
      ts,
    }
  }
  if (kind === "tool") {
    const name = stringValue(row.tool_name) || "tool"
    const failed = stringValue(row.status) !== "done"
    const display = workerToolDisplay(
      name,
      objectValue(row.metrics) ?? {},
      numberValue(row.duration_ms),
    )
    const body = [display.primary, display.metrics].filter(Boolean).join(" · ")
    return {
      body,
      collapsed: !failed,
      id: nextId(state, "worker-journal"),
      kind: "tool",
      metrics: display.metrics,
      primary: display.primary,
      signal: failed || stringValue(row.signal) !== "low" ? "high" : "low",
      status: failed ? "failed" : "done",
      title: "TOOL",
      toolName: name,
      ts,
      verb: toolVerb(name),
    }
  }
  if (kind === "phase") {
    const step = oneLine(stringValue(row.current_step), 120)
    const phase = friendlyPhase(stringValue(row.phase), arrayOfStrings(row.tool_names))
    return {
      body: step
        ? formatProgress(step, numberValue(row.todo_completed), numberValue(row.todo_total))
        : phase,
      id: nextId(state, "worker-journal"),
      kind: step ? "step" : "lifecycle",
      signal: "high",
      status: "running",
      title: step ? "CURRENT" : "PHASE",
      ts,
    }
  }
  if (kind === "guard") {
    const action = stringValue(row.action) || "decision"
    const source = stringValue(row.source)
    const reason = oneLine(stringValue(row.event), 160)
    return {
      body: [guardActionLabel(action), reason].filter(Boolean).join(" · "),
      id: nextId(state, "worker-journal"),
      kind: "guard",
      signal: "high",
      status: action === "finalize" ? "failed" : "partial",
      title: source ? `GUARD · ${source}` : "GUARD",
      ts,
    }
  }
  if (kind === "lifecycle") {
    const phase = stringValue(row.phase)
    const status = normalizeWorkerStatus(stringValue(row.status) || phase)
    const summary = oneLine(
      stringValue(row.error_summary) || stringValue(row.summary),
      220,
    )
    return {
      body: summary || lifecycleBody(status, numberValue(row.duration_ms)),
      id: nextId(state, "worker-journal"),
      kind: "lifecycle",
      signal: "high",
      status: timelineStatus(status),
      title: "RUN",
      ts,
    }
  }
  return undefined
}

function updateWorkerFromRun(state: AppState, run: WorkerRunSnapshot): void {
  const id = stringValue(run.worker_id)
  const worker = id ? state.workers[id] : undefined
  if (worker) mergeWorkerRun(worker, run)
}

function mergeWorkerRun(worker: WorkerInfo, run: WorkerRunSnapshot): boolean {
  if (
    !beginWorkerRun(
      worker,
      stringValue(run.run_id),
      numberValue(run.run_sequence),
    )
  ) return false
  advanceWorkerStatus(worker, normalizeWorkerStatus(stringValue(run.status)))
  worker.goal = stringValue(run.goal) || stringValue(run.goal_preview) || worker.goal
  if ("summary" in run) worker.report = stringValue(run.summary) || undefined
  worker.durationMs = numberValue(run.duration_ms) ?? worker.durationMs
  const usage = objectValue(run.usage)
  if (usage) worker.usage = { ...usage }
  if (TERMINAL_WORKER_STATES.has(worker.status)) worker.currentStep = undefined
  return true
}

function beginWorkerRun(
  worker: WorkerInfo,
  runId: string,
  runSequence?: number,
): boolean {
  if (!runId) return true
  if (runSequence !== undefined && worker.runSequence !== undefined) {
    if (runSequence < worker.runSequence) return false
    if (
      runSequence === worker.runSequence &&
      worker.runId !== undefined &&
      worker.runId !== runId
    ) return false
  }
  if (!worker.runId) {
    worker.runId = runId
    worker.runSequence = runSequence ?? worker.runSequence
    return true
  }
  if (worker.runId === runId) {
    worker.runSequence = runSequence ?? worker.runSequence
    return true
  }
  if (worker.runSequence !== undefined && runSequence === undefined) return false

  worker.runId = runId
  worker.runSequence = runSequence
  worker.aggregateId = undefined
  worker.currentStep = undefined
  worker.durationMs = undefined
  worker.goal = undefined
  worker.lastRevision = {}
  worker.phase = "queued"
  worker.phases = []
  worker.report = undefined
  worker.startedAt = undefined
  worker.status = "queued"
  worker.timeline = []
  worker.todoCompleted = undefined
  worker.todoTotal = undefined
  worker.usage = {}
  return true
}

function workerEventMatchesCurrentRun(worker: WorkerInfo, payload: JsonObject): boolean {
  const runId = stringValue(payload.run_id)
  if (!runId) return true
  if (!worker.runId) {
    return beginWorkerRun(worker, runId, numberValue(payload.run_sequence))
  }
  if (worker.runId !== runId) return false
  const runSequence = numberValue(payload.run_sequence)
  return (
    runSequence === undefined ||
    worker.runSequence === undefined ||
    runSequence === worker.runSequence
  )
}

function advanceWorkerStatus(worker: WorkerInfo, incoming: string): boolean {
  if (!incoming) return true
  const currentTerminal = TERMINAL_WORKER_STATES.has(worker.status)
  const incomingTerminal = TERMINAL_WORKER_STATES.has(incoming)
  if (currentTerminal && !incomingTerminal) return false
  const ranks: Record<string, number> = {
    created: 0,
    queued: 0,
    running: 1,
    waiting_for_context: 2,
  }
  if (!incomingTerminal && (ranks[incoming] ?? 0) < (ranks[worker.status] ?? 0)) {
    return false
  }
  worker.status = incoming
  return true
}

function settlePendingBlocks(state: AppState, status: TimelineStatus): void {
  for (const block of Object.values(state.pendingBlocks)) {
    if (!block.itemId) continue
    const item = findMasterItem(state, block.itemId)
    if (item?.status === "running") item.status = status
  }
  state.pendingBlocks = {}
  state.masterAggregateId = undefined
}

function addMasterItem(state: AppState, item: Omit<TimelineItem, "id">): TimelineItem {
  const complete: TimelineItem = { ...item, id: nextId(state, "master") }
  state.masterTimeline.push(complete)
  if (item.signal === "high" && item.kind !== "aggregate") state.masterAggregateId = undefined
  return state.masterTimeline.at(-1) ?? complete
}

function appendWorkerItem(
  state: AppState,
  worker: WorkerInfo,
  item: Omit<TimelineItem, "id">,
): TimelineItem {
  const complete: TimelineItem = { ...item, id: nextId(state, `worker-${worker.id.slice(0, 4)}`) }
  worker.timeline.push(complete)
  if (item.signal === "high" && item.kind !== "aggregate") worker.aggregateId = undefined
  return worker.timeline.at(-1) ?? complete
}

function bumpMasterAggregate(state: AppState, toolName: string): string {
  let item = state.masterAggregateId ? findMasterItem(state, state.masterAggregateId) : undefined
  if (!item || item.kind !== "aggregate") {
    item = addMasterItem(state, {
      aggregateCounts: {},
      kind: "aggregate",
      signal: "high",
      title: "ROUTINE ACTIVITY",
    })
    state.masterAggregateId = item.id
  }
  item.aggregateCounts ??= {}
  item.aggregateCounts[toolName] = (item.aggregateCounts[toolName] ?? 0) + 1
  item.body = formatAggregate(item.aggregateCounts)
  return item.id
}

function bumpWorkerAggregate(state: AppState, worker: WorkerInfo, toolName: string): string {
  let item = worker.aggregateId
    ? worker.timeline.find((candidate) => candidate.id === worker.aggregateId)
    : undefined
  if (!item || item.kind !== "aggregate") {
    item = appendWorkerItem(state, worker, {
      aggregateCounts: {},
      kind: "aggregate",
      signal: "high",
      title: "ROUTINE ACTIVITY",
    })
    worker.aggregateId = item.id
  }
  item.aggregateCounts ??= {}
  item.aggregateCounts[toolName] = (item.aggregateCounts[toolName] ?? 0) + 1
  item.body = formatAggregate(item.aggregateCounts)
  return item.id
}

function removeFromAggregate(
  timeline: TimelineItem[],
  aggregateId: string | undefined,
  toolName: string,
): void {
  if (!aggregateId) return
  const item = timeline.find((candidate) => candidate.id === aggregateId)
  if (!item?.aggregateCounts) return
  item.aggregateCounts[toolName] = Math.max(0, (item.aggregateCounts[toolName] ?? 0) - 1)
  if (!item.aggregateCounts[toolName]) delete item.aggregateCounts[toolName]
  item.body = formatAggregate(item.aggregateCounts)
}

function findMasterItem(state: AppState, id: string): TimelineItem | undefined {
  return state.masterTimeline.find((item) => item.id === id)
}

function markWorkerUnread(state: AppState, worker: WorkerInfo): void {
  if (state.view.kind !== "worker" || state.view.workerId !== worker.id) {
    worker.unread = Math.min(99, worker.unread + 1)
  }
}

function sameView(left: View, right: View): boolean {
  if (left.kind !== right.kind) return false
  return left.kind !== "worker" || (right.kind === "worker" && left.workerId === right.workerId)
}

function nextId(state: AppState, prefix: string): string {
  state.sequence += 1
  return `${prefix}-${state.sequence}`
}

function emptyTurn(id: string, user?: TimelineItem): TurnGroup {
  return {
    collapsed: false,
    id,
    process: [],
    processSummary: "",
    user,
  }
}

function formatProcessSummary(items: TimelineItem[], summary?: TimelineItem): string {
  let tools = 0
  const workers = new Set<string>()
  let failures = 0
  for (const item of items) {
    if (item.kind === "tool") tools += 1
    if (item.kind === "aggregate") {
      tools += Object.values(item.aggregateCounts ?? {}).reduce((total, count) => total + count, 0)
    }
    if (item.workerLabel) workers.add(item.workerLabel)
    if (item.status === "failed") failures += 1
  }
  const parts = [`${tools} tool${tools === 1 ? "" : "s"}`]
  if (workers.size) parts.push(`${workers.size} worker${workers.size === 1 ? "" : "s"}`)
  const duration = summary?.body?.match(/(?:^| · )(\d+(?:\.\d+)?(?:ms|s)|\d+m\d+s)(?: · |$)/)?.[1]
  if (duration) parts.push(duration)
  if (failures) parts.push(`${failures} failed`)
  return parts.join(" · ")
}

function toolVerb(name: string): string {
  return ({
    await: "AWAIT",
    edit: "EDIT",
    exec: "RAN",
    glob: "INSPECT",
    grep: "INSPECT",
    inspect_worker: "INSPECT",
    read: "READ",
    spawn: "SPAWN",
    spawn_worker: "SPAWN",
    webfetch: "FETCHED",
    websearch: "SEARCHED",
    write: "WROTE",
  } as Record<string, string>)[name] ?? name.replaceAll("_", " ").toUpperCase()
}

function formatAggregate(counts: Record<string, number>): string {
  const entries = Object.entries(counts).filter(([, count]) => count > 0)
  if (!entries.length) return "Routine checks completed"
  return entries.map(([name, count]) => `${name} ×${count}`).join(" · ")
}

function summarizeToolArguments(name: string, value: unknown): string {
  const args = objectValue(value) ?? {}
  if (name === "read") return stringValue(args.path) || "Reading file"
  if (name === "write") {
    const content = stringValue(args.content)
    return [stringValue(args.path), content ? `${content.length} chars` : ""].filter(Boolean).join(" · ")
  }
  if (name === "edit") return stringValue(args.path) || "Editing file"
  if (name === "exec") return oneLine(stringValue(args.command), 140) || "Running command"
  if (name === "todowrite") {
    const todos = Array.isArray(args.todos) ? args.todos : []
    return `${todos.length} item${todos.length === 1 ? "" : "s"}`
  }
  if (name === "glob" || name === "grep") {
    return [stringValue(args.pattern), stringValue(args.path) || stringValue(args.root)]
      .filter(Boolean)
      .join(" · ")
  }
  return oneLine(safeJson(value), 160)
}

function summarizeToolResult(
  name: string,
  result: unknown,
  argumentsValue: unknown,
  failed: boolean,
  durationMs?: number,
): string {
  const text = stringValue(result)
  const args = objectValue(argumentsValue) ?? {}
  const duration = formatDuration(durationMs)
  if (failed) return ["Failed", oneLine(text, 160), duration].filter(Boolean).join(" · ")
  if (name === "write") {
    return [stringValue(args.path), `${stringValue(args.content).length} chars written`, duration]
      .filter(Boolean)
      .join(" · ")
  }
  if (name === "edit") return [stringValue(args.path), "Updated", duration].filter(Boolean).join(" · ")
  if (name === "exec") {
    return [exitCode(text), `${text.length} chars / ${lineCount(text)} lines`, duration]
      .filter(Boolean)
      .join(" · ")
  }
  if (name === "read") {
    return [stringValue(args.path), `${text.length} chars / ${lineCount(text)} lines`, duration]
      .filter(Boolean)
      .join(" · ")
  }
  return [`${text.length} chars / ${lineCount(text)} lines`, duration].filter(Boolean).join(" · ")
}

function workerToolDisplay(
  name: string,
  metrics: JsonObject,
  durationMs?: number,
): { metrics: string; primary: string } {
  const primary = name === "exec"
    ? oneLine(stringValue(metrics.command), 160) || "exec"
    : stringValue(metrics.resource) || name
  const parts: string[] = []
  if (name === "write" && numberValue(metrics.input_chars) !== undefined) {
    parts.push(`${numberValue(metrics.input_chars)} chars written`)
  } else if (name === "edit") {
    const oldChars = numberValue(metrics.old_chars)
    const newChars = numberValue(metrics.new_chars)
    if (oldChars !== undefined && newChars !== undefined) parts.push(`${oldChars} → ${newChars} chars`)
  } else if (name === "exec" && numberValue(metrics.exit_code) !== undefined) {
    parts.push(`exit ${numberValue(metrics.exit_code)}`)
  } else if (numberValue(metrics.item_count) !== undefined) {
    parts.push(`${numberValue(metrics.item_count)} items`)
  } else if (numberValue(metrics.result_chars) !== undefined) {
    parts.push(`${numberValue(metrics.result_chars)} chars / ${numberValue(metrics.result_lines) ?? 0} lines`)
  }
  const duration = formatDuration(durationMs)
  if (duration) parts.push(duration)
  return { metrics: parts.join(" · ") || "Completed", primary }
}

function formatTurnSummary(durationMs: number | undefined, usage: JsonObject, tools: string[]): string {
  const parts = ["Completed"]
  const duration = formatDuration(durationMs)
  if (duration) parts.push(duration)
  const { input: prompt, output: completion, total } = usageCounters(usage)
  if (prompt !== undefined || completion !== undefined) {
    parts.push(`tokens ${prompt ?? "?"} in / ${completion ?? "?"} out`)
  } else if (total !== undefined) {
    parts.push(`tokens ${total}`)
  }
  if (tools.length) parts.push(`${tools.length} tool${tools.length === 1 ? "" : "s"}`)
  return parts.join(" · ")
}

export function usageCounters(usage: JsonObject): {
  input?: number
  output?: number
  total?: number
} {
  const nested = objectValue(usage.totals) ?? objectValue(usage.total) ?? {}
  const input =
    numberValue(usage.prompt_tokens) ??
    numberValue(usage.input_tokens) ??
    numberValue(nested.prompt_tokens) ??
    numberValue(nested.input_tokens)
  const output =
    numberValue(usage.completion_tokens) ??
    numberValue(usage.output_tokens) ??
    numberValue(nested.completion_tokens) ??
    numberValue(nested.output_tokens)
  const total = numberValue(usage.total_tokens) ?? numberValue(nested.total_tokens)
  return { input, output, total }
}

function formatProgress(step: string, completed?: number, total?: number): string {
  if (completed === undefined || total === undefined || total <= 0) return step
  return `${completed}/${total} · ${step}`
}

function friendlyPhase(phase: string, tools: string[]): string {
  if (phase === "using_tool") {
    if (tools.some((name) => name === "write" || name === "edit")) return "editing"
    if (tools.includes("exec")) return "testing"
    if (tools.some((name) => LOW_SIGNAL_TOOLS.has(name))) return "analyzing"
    return "executing"
  }
  return (
    {
      analyzing: "analyzing",
      branch_done: "synthesizing",
      branch_running: "executing",
      delegating: "delegating",
      drafting: "drafting",
      finalizing: "finalizing",
      handoff: "handoff",
      planning: "planning",
      processing: "processing",
      synthesizing: "synthesizing",
      working_step: "working",
    }[phase] ?? phase
  )
}

function lifecycleBody(status: string, durationMs?: number): string {
  const label =
    {
      cancelled: "Cancelled",
      completed: "Completed",
      failed: "Failed",
      partial: "Partially completed",
      queued: "Queued",
      running: "Started",
      timed_out: "Timed out",
      waiting_for_context: "Waiting for context",
    }[status] ?? status
  const duration = formatDuration(durationMs)
  return duration ? `${label} · ${duration}` : label
}

function normalizeWorkerStatus(value: string): string {
  if (value === "timed_out") return "timed_out"
  return value || "running"
}

function timelineStatus(status: string): TimelineStatus {
  if (status === "completed") return "done"
  if (status === "partial" || status === "waiting_for_context") return "partial"
  if (status === "cancelled") return "cancelled"
  if (status === "failed" || status === "timed_out") return "failed"
  return "running"
}

function guardActionLabel(action: string): string {
  return { continue: "Continue", finalize: "Stopped", retry: "Retrying" }[action] ?? action
}

function workerLabel(profileId: string, workerId: string): string {
  const suffix = workerId.replace(/[^a-zA-Z0-9]/g, "").slice(0, 4)
  return suffix ? `${profileId}#${suffix}` : profileId
}

function isFailedStatus(value: unknown): boolean {
  return ["error", "failed", "failure"].includes(stringValue(value).toLowerCase())
}

function toolResultFailed(value: unknown): boolean {
  const text = stringValue(value).trimStart().toLowerCase()
  return text.startsWith("error") || text.startsWith("fatal") || text.includes("exit code: 1")
}

function exitCode(text: string): string {
  for (const line of text.split("\n").reverse()) {
    const match = line.trim().match(/^Exit code:\s*(.+)$/i)
    if (match?.[1]) return `exit ${match[1]}`
  }
  return "command completed"
}

function formatDuration(value?: number): string {
  if (value === undefined || !Number.isFinite(value)) return ""
  const milliseconds = Math.max(0, Math.round(value))
  if (milliseconds < 1_000) return `${milliseconds}ms`
  if (milliseconds >= 60_000) {
    const seconds = Math.floor(milliseconds / 1_000)
    return `${Math.floor(seconds / 60)}m${String(seconds % 60).padStart(2, "0")}s`
  }
  return `${(milliseconds / 1_000).toFixed(1)}s`
}

function lineCount(text: string): number {
  return text ? text.split("\n").length : 0
}

function safeJson(value: unknown): string {
  let rendered: string
  try {
    rendered = JSON.stringify(value, null, 2) ?? ""
  } catch {
    rendered = String(value ?? "")
  }
  if (rendered.length <= RAW_DETAIL_LIMIT) return rendered
  return `${rendered.slice(0, RAW_DETAIL_LIMIT)}… [${rendered.length - RAW_DETAIL_LIMIT} chars hidden]`
}

function truncateText(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit)}… [${value.length - limit} chars hidden]`
}

function oneLine(value: string, limit: number): string {
  const clean = value.replace(/\r\n?/g, "\n").split(/\s+/).filter(Boolean).join(" ")
  if (clean.length <= limit) return clean
  return `${clean.slice(0, limit)}…`
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : value === null || value === undefined ? "" : String(value)
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function objectValue(value: unknown): JsonObject | undefined {
  return isObject(value) ? value : undefined
}

function arrayOfObjects(value: unknown): JsonObject[] {
  return Array.isArray(value) ? value.filter(isObject) : []
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter(isString) : []
}

function isString(value: unknown): value is string {
  return typeof value === "string"
}
