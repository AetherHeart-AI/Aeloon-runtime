import type { KeyEvent, KeyBinding, ScrollBoxRenderable, TextareaRenderable } from "@opentui/core"
import { For, Show, createEffect, createMemo, createSignal } from "solid-js"
import type { CommandDefinition } from "./commands"
import type {
  AppState,
  GatewayLog,
  TimelineItem,
  TurnGroup,
  View,
  WorkerInfo,
} from "./model"
import {
  usageCounters,
  visibleMasterTurns,
  visibleWorkerItems,
  waitingSummary,
} from "./model"
import type { Palette } from "./theme"

function EmptySlot() {
  return <box height={0} width={0} flexShrink={0} />
}

export interface HeaderProps {
  palette: Palette
  state: AppState
}

export function Header(props: HeaderProps) {
  const location = createMemo(() => compactPath(props.state.workspace))
  const viewLabel = createMemo(() => {
    if (props.state.view.kind === "worker") {
      return props.state.workers[props.state.view.workerId]?.label ?? "WORKER"
    }
    return props.state.view.kind.toUpperCase()
  })
  return (
    <box
      height={1}
      width="100%"
      flexDirection="row"
      alignItems="center"
      paddingLeft={1}
      paddingRight={1}
      backgroundColor={props.palette.panel}
      flexShrink={0}
    >
      <text fg={props.palette.accent} width={12} selectable={false}>
        <strong>AELOON</strong>
      </text>
      <text fg={props.palette.foreground} flexGrow={1} overflow="hidden" wrapMode="none">
        {location()}
      </text>
      <text fg={props.palette.muted} overflow="hidden" wrapMode="none">
        {props.state.model || "runtime"} · {viewLabel()}
      </text>
    </box>
  )
}

interface TranscriptPaneProps {
  expandedReport: boolean
  focus: boolean
  onFocus: () => void
  onPinChange: (pinned: boolean) => void
  onScrollRef: (renderable: ScrollBoxRenderable) => void
  onToggleReport: () => void
  onToggleTimelineItem: (itemId: string) => void
  onToggleTurnProcess: (turnId: string) => void
  palette: Palette
  state: AppState
}

export function TranscriptPane(props: TranscriptPaneProps) {
  const masterTurns = createMemo(() => visibleMasterTurns(props.state))
  const selectedWorker = createMemo(() => {
    const view = props.state.view
    return view.kind === "worker" ? props.state.workers[view.workerId] : undefined
  })
  return (
    <scrollbox
      ref={props.onScrollRef}
      focused={props.focus}
      stickyScroll={props.state.pinToBottom}
      stickyStart="bottom"
      viewportCulling
      onMouseDown={props.onFocus}
      onMouseScroll={() => props.onPinChange(false)}
      style={{
        flexGrow: 1,
        height: "100%",
        width: "100%",
        paddingLeft: 1,
        paddingRight: 1,
        paddingTop: 1,
        scrollbarOptions: {
          showArrows: false,
          trackOptions: {
            foregroundColor: props.palette.border,
            backgroundColor: props.palette.background,
          },
        },
      }}
    >
      <Show when={props.state.view.kind === "master"} fallback={<EmptySlot />}>
        <For each={masterTurns()} fallback={<EmptySlot />}>
          {(turn) => (
            <TurnView
              onToggleItem={props.onToggleTimelineItem}
              onToggleProcess={props.onToggleTurnProcess}
              palette={props.palette}
              showRaw={props.state.verbosity === "verbose"}
              turn={turn}
            />
          )}
        </For>
        <Show when={waitingSummary(props.state)} fallback={<EmptySlot />}>
          {(summary) => (
            <box marginBottom={1} flexShrink={0}>
              <text fg={props.palette.warning} selectable selectionBg={props.palette.selection}>
                ◌ {summary()}
              </text>
            </box>
          )}
        </Show>
      </Show>
      <Show when={selectedWorker()} fallback={<EmptySlot />}>
        {(worker) => (
          <WorkerDetail
            expandedReport={props.expandedReport}
            onToggleReport={props.onToggleReport}
            onToggleTimelineItem={props.onToggleTimelineItem}
            palette={props.palette}
            state={props.state}
            worker={worker()}
          />
        )}
      </Show>
      <Show when={props.state.view.kind === "logs"} fallback={<EmptySlot />}>
        <LogsView logs={props.state.gatewayLogs} logDetail={props.state.logDetail} palette={props.palette} />
      </Show>
    </scrollbox>
  )
}

interface TimelineRowProps {
  item: TimelineItem
  onToggle?: (itemId: string) => void
  palette: Palette
  showRaw?: boolean
}

export function TimelineRow(props: TimelineRowProps) {
  const color = createMemo(() => statusColor(props.item, props.palette))
  const icon = createMemo(() => statusIcon(props.item))
  const heading = createMemo(() => {
    const parts = [props.item.title]
    if (props.item.toolName && props.item.title !== props.item.toolName) parts.push(props.item.toolName)
    if (props.item.workerLabel) parts.push(props.item.workerLabel)
    return parts.join(" · ")
  })
  const expanded = createMemo(() => props.showRaw || props.item.collapsed === false)
  const result = createMemo(() =>
    expanded()
      ? props.item.resultDetail ?? props.item.resultPreview
      : props.item.resultPreview,
  )
  const canExpand = createMemo(() => Boolean(
    props.item.rawDetail
      || props.item.kind === "thinking"
      || (props.item.resultDetail && props.item.resultDetail !== props.item.resultPreview),
  ))
  if (props.item.kind === "aggregate") {
    return (
      <text
        fg={props.palette.muted}
        selectable
        selectionBg={props.palette.selection}
        wrapMode="word"
        flexShrink={0}
      >
        ⋯ {props.item.body}
      </text>
    )
  }
  if (props.item.kind === "tool") {
    return (
      <box
        width="100%"
        flexDirection="column"
        marginBottom={expanded() || result() ? 1 : 0}
        flexShrink={0}
      >
        <text
          fg={color()}
          selectable
          selectionBg={props.palette.selection}
          selectionFg={props.palette.foreground}
          wrapMode="word"
          onMouseDown={() => canExpand() && props.onToggle?.(props.item.id)}
        >
          {icon()} <strong>{props.item.verb ?? props.item.toolName?.toUpperCase() ?? "TOOL"}</strong>
          {props.item.primary || props.item.body ? ` ${props.item.primary ?? props.item.body}` : ""}
          {props.item.metrics ? ` · ${props.item.metrics}` : ""}
          {props.item.workerLabel ? ` · ${props.item.workerLabel}` : ""}
          {canExpand() ? ` ${expanded() ? "▾" : "▸"}` : ""}
        </text>
        <Show when={result()} fallback={<EmptySlot />}>
          {(detail) => (
            <text fg={props.palette.foreground} selectable selectionBg={props.palette.selection} wrapMode="word">
              {detail()}
            </text>
          )}
        </Show>
        <Show when={expanded() && props.item.rawDetail} fallback={<EmptySlot />}>
          {(detail) => (
            <text fg={props.palette.muted} selectable selectionBg={props.palette.selection} wrapMode="word">
              {detail()}
            </text>
          )}
        </Show>
      </box>
    )
  }
  if (props.item.kind === "thinking") {
    return (
      <box width="100%" flexDirection="column" marginBottom={expanded() ? 1 : 0} flexShrink={0}>
        <text
          fg={props.palette.muted}
          selectable
          selectionBg={props.palette.selection}
          onMouseDown={() => props.onToggle?.(props.item.id)}
        >
          {expanded() ? "▾" : "▸"} thinking{props.item.metrics ? ` · ${props.item.metrics}` : ""}
        </text>
        <Show when={expanded() && props.item.body} fallback={<EmptySlot />}>
          {(body) => (
            <text fg={props.palette.muted} selectable selectionBg={props.palette.selection} wrapMode="word">
              {body()}
            </text>
          )}
        </Show>
      </box>
    )
  }
  return (
    <box width="100%" flexDirection="column" marginBottom={1} flexShrink={0}>
      <text
        fg={color()}
        selectable
        selectionBg={props.palette.selection}
        selectionFg={props.palette.foreground}
        wrapMode="word"
      >
        {icon()} <strong>{heading()}</strong>
      </text>
      <Show when={props.item.body} fallback={<EmptySlot />}>
        {(body) => (
          <text
            fg={props.item.kind === "summary" || props.item.kind === "aggregate" ? props.palette.muted : props.palette.foreground}
            selectable
            selectionBg={props.palette.selection}
            selectionFg={props.palette.foreground}
            wrapMode="word"
          >
            {body()}
          </text>
        )}
      </Show>
      <Show when={props.item.detail} fallback={<EmptySlot />}>
        {(detail) => (
          <text fg={props.palette.muted} selectable selectionBg={props.palette.selection} wrapMode="word">
            {detail()}
          </text>
        )}
      </Show>
      <Show when={props.showRaw && props.item.rawDetail} fallback={<EmptySlot />}>
        {(detail) => (
          <text fg={props.palette.muted} selectable selectionBg={props.palette.selection} wrapMode="word">
            {detail()}
          </text>
        )}
      </Show>
    </box>
  )
}

interface TurnViewProps {
  onToggleItem: (itemId: string) => void
  onToggleProcess: (turnId: string) => void
  palette: Palette
  showRaw: boolean
  turn: TurnGroup
}

function TurnView(props: TurnViewProps) {
  return (
    <box width="100%" flexDirection="column" marginBottom={1} flexShrink={0}>
      <Show when={props.turn.user} fallback={<EmptySlot />}>
        {(item) => <TimelineRow item={item()} palette={props.palette} showRaw={props.showRaw} />}
      </Show>
      <Show when={props.turn.process.length} fallback={<EmptySlot />}>
        <box width="100%" flexDirection="column" flexShrink={0}>
          <text
            fg={props.turn.process.some((item) => item.status === "failed") ? props.palette.error : props.palette.muted}
            selectable={false}
            onMouseDown={() => props.onToggleProcess(props.turn.id)}
          >
            {props.turn.collapsed ? "▸" : "▾"} PROCESS · {props.turn.processSummary}
          </text>
          <Show when={!props.turn.collapsed} fallback={<EmptySlot />}>
            <For each={props.turn.process} fallback={<EmptySlot />}>
              {(item) => (
                <TimelineRow
                  item={item}
                  onToggle={props.onToggleItem}
                  palette={props.palette}
                  showRaw={props.showRaw}
                />
              )}
            </For>
          </Show>
        </box>
      </Show>
      <Show when={props.turn.answer} fallback={<EmptySlot />}>
        {(item) => <TimelineRow item={item()} palette={props.palette} showRaw={props.showRaw} />}
      </Show>
      <Show when={props.turn.summary} fallback={<EmptySlot />}>
        {(item) => <TimelineRow item={item()} palette={props.palette} showRaw={props.showRaw} />}
      </Show>
    </box>
  )
}

interface WorkerDetailProps {
  expandedReport: boolean
  onToggleReport: () => void
  onToggleTimelineItem: (itemId: string) => void
  palette: Palette
  state: AppState
  worker: WorkerInfo
}

export function WorkerDetail(props: WorkerDetailProps) {
  const items = createMemo(() => visibleWorkerItems(props.state, props.worker.id))
  const report = createMemo(() => {
    const value = props.worker.report ?? ""
    if (props.expandedReport || value.length <= 700) return value
    return `${value.slice(0, 700)}…`
  })
  const remaining = createMemo(() => {
    if (props.worker.todoTotal === undefined) return undefined
    return Math.max(0, props.worker.todoTotal - (props.worker.todoCompleted ?? 0) - (props.worker.currentStep ? 1 : 0))
  })
  return (
    <box width="100%" flexDirection="column" flexShrink={0}>
      <box width="100%" flexDirection="column" marginBottom={1} flexShrink={0}>
        <text fg={props.palette.accent} selectable selectionBg={props.palette.selection}>
          <strong>{props.worker.label}</strong> · {props.worker.profileId} · {workerStatusLabel(props.worker.status)}
          {props.worker.durationMs !== undefined ? ` · ${formatElapsed(props.worker.durationMs)}` : ""}
        </text>
        <text fg={props.palette.foreground} selectable selectionBg={props.palette.selection} wrapMode="word">
          {props.worker.goal || "Goal will appear when this Worker publishes its run context."}
        </text>
      </box>

      <box width="100%" flexDirection="column" marginBottom={1} flexShrink={0}>
        <text fg={props.palette.muted} selectable={false}>
          PHASE
        </text>
        <text fg={props.palette.foreground} selectable selectionBg={props.palette.selection} wrapMode="word">
          {props.worker.phases.length ? props.worker.phases.join(" → ") : props.worker.phase || "queued"}
        </text>
      </box>

      <Show when={props.worker.todoTotal !== undefined || props.worker.currentStep} fallback={<EmptySlot />}>
        <box width="100%" flexDirection="column" marginBottom={1} flexShrink={0}>
          <text fg={props.palette.muted} selectable={false}>
            TODO
          </text>
          <Show when={(props.worker.todoCompleted ?? 0) > 0} fallback={<EmptySlot />}>
            <text fg={props.palette.success} selectable selectionBg={props.palette.selection}>
              [x] {props.worker.todoCompleted} completed
            </text>
          </Show>
          <Show when={props.worker.currentStep} fallback={<EmptySlot />}>
            {(step) => (
              <text fg={props.palette.warning} selectable selectionBg={props.palette.selection} wrapMode="word">
                [&gt;] {step()}
              </text>
            )}
          </Show>
          <Show when={(remaining() ?? 0) > 0} fallback={<EmptySlot />}>
            <text fg={props.palette.muted} selectable selectionBg={props.palette.selection}>
              [ ] {remaining()} remaining
            </text>
          </Show>
        </box>
      </Show>

      <text fg={props.palette.muted} marginBottom={1} selectable={false} flexShrink={0}>
        TIMELINE
      </text>
      <For each={items()} fallback={<EmptySlot />}>
        {(item) => (
          <TimelineRow
            item={item}
            onToggle={props.onToggleTimelineItem}
            palette={props.palette}
            showRaw={props.state.verbosity === "verbose"}
          />
        )}
      </For>

      <Show when={report()} fallback={<EmptySlot />}>
        {(value) => (
          <box
            width="100%"
            flexDirection="column"
            marginBottom={1}
            padding={1}
            border
            borderColor={props.palette.border}
            onMouseDown={props.onToggleReport}
            flexShrink={0}
          >
            <text fg={statusColor({ ...({} as TimelineItem), status: workerTimelineStatus(props.worker.status) }, props.palette)}>
              <strong>RESULT · {workerStatusLabel(props.worker.status)}</strong>
            </text>
            <text fg={props.palette.foreground} selectable selectionBg={props.palette.selection} wrapMode="word">
              {value()}
            </text>
            <Show when={(props.worker.report?.length ?? 0) > 700} fallback={<EmptySlot />}>
              <text fg={props.palette.muted} selectable={false}>
                click or press e to {props.expandedReport ? "collapse" : "expand"}
              </text>
            </Show>
          </box>
        )}
      </Show>
    </box>
  )
}

interface LogsViewProps {
  logDetail: boolean
  logs: GatewayLog[]
  palette: Palette
}

export function LogsView(props: LogsViewProps) {
  return (
    <box width="100%" flexDirection="column" flexShrink={0}>
      <text fg={props.palette.warning} marginBottom={1} selectable={false}>
        GATEWAY LOGS · operator diagnostics · {props.logDetail ? "detail" : "compact"}
      </text>
      <Show when={props.logs.length} fallback={<text fg={props.palette.muted}>No gateway logs yet.</text>}>
        <For each={props.logs} fallback={<EmptySlot />}>
          {(log) => (
            <box width="100%" flexDirection="column" marginBottom={1} flexShrink={0}>
              <text fg={logLevelColor(log.level, props.palette)} selectable selectionBg={props.palette.selection}>
                {shortTime(log.ts)} {log.level.padEnd(7)} {log.source}
              </text>
              <text fg={props.palette.foreground} selectable selectionBg={props.palette.selection} wrapMode="word">
                {log.message}
              </text>
              <Show when={props.logDetail && log.detail} fallback={<EmptySlot />}>
                <text fg={props.palette.muted} selectable selectionBg={props.palette.selection} wrapMode="word">
                  {safeStringify(log.detail)}
                </text>
              </Show>
            </box>
          )}
        </For>
      </Show>
    </box>
  )
}

interface WorkerStripProps {
  onSelect: (view: View) => void
  palette: Palette
  state: AppState
}

export function WorkerStrip(props: WorkerStripProps) {
  return (
    <scrollbox
      height={2}
      width="100%"
      scrollX
      scrollY={false}
      flexShrink={0}
      style={{
        contentOptions: { flexDirection: "row", alignItems: "center", gap: 1 },
        scrollbarOptions: { showArrows: false },
      }}
    >
      <WorkerTab
        active={props.state.view.kind === "master"}
        label="MASTER"
        meta={props.state.running ? "running" : "transcript"}
        onSelect={() => props.onSelect({ kind: "master" })}
        palette={props.palette}
        status={props.state.running ? "running" : "idle"}
      />
      <For each={props.state.workerOrder} fallback={<EmptySlot />}>
        {(workerId) => {
          const worker = () => props.state.workers[workerId]
          return (
            <Show when={worker()} fallback={<EmptySlot />}>
              {(item) => (
                <WorkerTab
                  active={props.state.view.kind === "worker" && props.state.view.workerId === item().id}
                  label={item().label}
                  meta={workerProgress(item())}
                  onSelect={() => props.onSelect({ kind: "worker", workerId: item().id })}
                  palette={props.palette}
                  status={item().status}
                  unread={item().unread}
                />
              )}
            </Show>
          )
        }}
      </For>
    </scrollbox>
  )
}

interface WorkerTabProps {
  active: boolean
  label: string
  meta: string
  onSelect: () => void
  palette: Palette
  status: string
  unread?: number
}

function WorkerTab(props: WorkerTabProps) {
  return (
    <box
      height={2}
      minWidth={18}
      maxWidth={32}
      paddingLeft={1}
      paddingRight={1}
      flexDirection="column"
      flexShrink={0}
      backgroundColor={props.active ? props.palette.panel : props.palette.background}
      onMouseDown={props.onSelect}
    >
      <text fg={workerStatusColor(props.status, props.palette)} overflow="hidden" wrapMode="none" selectable={false}>
        {props.active ? "▸" : statusDot(props.status)} <strong>{props.label}</strong>
        {props.unread ? ` •${props.unread > 9 ? "9+" : props.unread}` : ""}
      </text>
      <text fg={props.palette.muted} overflow="hidden" wrapMode="none" selectable={false}>
        {props.meta}
      </text>
    </box>
  )
}

interface StatusBarProps {
  clipboardStatus: string
  now: number
  palette: Palette
  state: AppState
}

export function StatusBar(props: StatusBarProps) {
  const status = createMemo(() => {
    const parts: string[] = []
    if (props.state.bridgeError) {
      parts.push("failed")
    } else if (props.state.sessionSwitching) {
      parts.push(`${spinner(props.now)} switching session`)
    } else if (!props.state.bridgeReady) {
      parts.push(`${spinner(props.now)} connecting`)
    } else if (props.state.running) {
      const elapsed = props.state.turnStartedAt ? props.now - props.state.turnStartedAt : 0
      parts.push(`${spinner(props.now)} running ${formatElapsed(elapsed)}`)
    } else {
      parts.push("idle")
    }
    const usage = usageLabel(props.state.usage)
    if (usage) parts.push(usage)
    if (props.state.queuedPrompts.length) parts.push(`queued (${props.state.queuedPrompts.length})`)
    parts.push(props.state.verbosity)
    parts.push(props.state.pinToBottom ? "pin:on" : "pin:off")
    parts.push(`focus:${props.state.focus}`)
    if (props.clipboardStatus) parts.push(props.clipboardStatus)
    return parts.join(" · ")
  })
  return (
    <box
      height={1}
      width="100%"
      flexDirection="row"
      paddingLeft={1}
      paddingRight={1}
      gap={1}
      backgroundColor={props.palette.panel}
      flexShrink={0}
    >
      <text
        fg={props.state.bridgeError ? props.palette.error : props.state.running ? props.palette.warning : props.palette.muted}
        flexGrow={1}
        flexShrink={1}
        minWidth={0}
        overflow="hidden"
        wrapMode="none"
      >
        {status()}
      </text>
      <text
        fg={props.palette.muted}
        width={32}
        flexShrink={0}
        overflow="hidden"
        wrapMode="none"
        selectable={false}
      >
          Tab views · Esc focus · /help
      </text>
    </box>
  )
}

interface ComposerProps {
  draft: string
  focused: boolean
  history: readonly string[]
  onFocus: () => void
  onFocusTranscript: () => void
  onSubmit: (value: string) => void
  onValueChange: (value: string) => void
  palette: Palette
  queue: readonly string[]
  suggestions: CommandDefinition[]
}

const COMPOSER_KEY_BINDINGS: KeyBinding[] = [
  { name: "return", action: "submit" },
  { name: "return", shift: true, action: "newline" },
  { name: "linefeed", action: "submit" },
]

export function Composer(props: ComposerProps) {
  let textarea: TextareaRenderable | undefined
  const [value, setValue] = createSignal("")
  const [selectedSuggestion, setSelectedSuggestion] = createSignal(0)
  const [historyIndex, setHistoryIndex] = createSignal(-1)
  const [historyDraft, setHistoryDraft] = createSignal("")
  const rows = createMemo(() => Math.min(6, Math.max(2, value().split("\n").length + 1)))

  const registerTextarea = (renderable: TextareaRenderable) => {
    textarea = renderable
    queueMicrotask(() => {
      if (props.focused) renderable.focus()
    })
  }

  createEffect(() => {
    if (props.focused) textarea?.focus()
    else textarea?.blur()
  })
  createEffect(() => {
    props.suggestions
    setSelectedSuggestion(0)
  })
  createEffect(() => {
    const draft = props.draft
    if (draft === value()) return
    setValue(draft)
    textarea?.setText(draft)
    textarea?.gotoBufferEnd()
  })

  const replaceValue = (next: string) => {
    setValue(next)
    props.onValueChange(next)
    textarea?.setText(next)
    textarea?.gotoBufferEnd()
  }

  const selectSuggestion = () => {
    const suggestion = props.suggestions[selectedSuggestion()]
    if (!suggestion) return
    const hasArguments = suggestion.usage.includes("<") || suggestion.usage.includes("[")
    replaceValue(`/${suggestion.name}${hasArguments ? " " : ""}`)
  }

  const recallHistory = (direction: -1 | 1) => {
    if (!props.history.length) return
    if (historyIndex() < 0) setHistoryDraft(value())
    let next = historyIndex() + direction
    next = Math.max(-1, Math.min(props.history.length - 1, next))
    replaceValue(next < 0 ? historyDraft() : (props.history[props.history.length - 1 - next] ?? ""))
    setHistoryIndex(next)
  }

  const handleKeyDown = (key: KeyEvent) => {
    if (props.suggestions.length) {
      if (key.name === "up" || key.name === "down") {
        key.preventDefault()
        const direction = key.name === "up" ? -1 : 1
        setSelectedSuggestion(
          (current) => (current + direction + props.suggestions.length) % props.suggestions.length,
        )
        return
      }
      if (key.name === "tab") {
        key.preventDefault()
        selectSuggestion()
        return
      }
    }
    if (key.name === "tab") {
      key.preventDefault()
      props.onFocusTranscript()
      return
    }
    if (
      (key.name === "up" || key.name === "down") &&
      !value().includes("\n") &&
      textarea?.logicalCursor.row === 0
    ) {
      key.preventDefault()
      recallHistory(key.name === "up" ? 1 : -1)
    }
  }

  const submit = () => {
    const submitted = value()
    if (!submitted.trim()) return
    props.onSubmit(submitted)
    replaceValue("")
    setHistoryIndex(-1)
  }

  return (
    <box width="100%" flexDirection="column" flexShrink={0}>
      <Show when={props.suggestions.length} fallback={<EmptySlot />}>
        <box
          width="100%"
          height={Math.min(6, props.suggestions.length)}
          flexDirection="column"
          paddingLeft={2}
          paddingRight={2}
          backgroundColor={props.palette.panel}
          flexShrink={0}
        >
          <For each={props.suggestions} fallback={<EmptySlot />}>
            {(suggestion, index) => (
              <box
                width="100%"
                flexDirection="row"
                backgroundColor={index() === selectedSuggestion() ? props.palette.selection : props.palette.panel}
                onMouseDown={() => {
                  setSelectedSuggestion(index())
                  selectSuggestion()
                }}
              >
                <text fg={index() === selectedSuggestion() ? props.palette.accent : props.palette.foreground} width={30}>
                  {suggestion.usage}
                </text>
                <text fg={props.palette.muted} overflow="hidden" wrapMode="none">
                  {suggestion.description}
                </text>
              </box>
            )}
          </For>
        </box>
      </Show>
      <box
        width="100%"
        height={rows() + 2}
        paddingLeft={1}
        paddingRight={1}
        border
        borderColor={props.focused ? props.palette.accent : props.palette.border}
        backgroundColor={props.palette.background}
        flexShrink={0}
        onMouseDown={props.onFocus}
      >
        <textarea
          ref={registerTextarea}
          id="composer"
          focused={props.focused}
          height={rows()}
          width="100%"
          placeholder="Ask Aeloon to work in this workspace…"
          placeholderColor={props.palette.muted}
          textColor={props.palette.foreground}
          focusedTextColor={props.palette.foreground}
          backgroundColor={props.palette.background}
          focusedBackgroundColor={props.palette.background}
          cursorColor={props.palette.accent}
          selectionBg={props.palette.selection}
          selectionFg={props.palette.foreground}
          wrapMode="word"
          keyBindings={COMPOSER_KEY_BINDINGS}
          onContentChange={() => {
            const next = textarea?.plainText ?? ""
            setValue(next)
            props.onValueChange(next)
            if (historyIndex() >= 0) setHistoryIndex(-1)
          }}
          onKeyDown={handleKeyDown}
          onSubmit={submit}
        />
      </box>
      <Show when={props.queue.length} fallback={<EmptySlot />}>
        <text height={1} fg={props.palette.muted} paddingLeft={1} overflow="hidden" wrapMode="none" flexShrink={0}>
          queued ({props.queue.length}) · {props.queue.map((prompt) => singleLine(prompt, 36)).join(" → ")}
        </text>
      </Show>
    </box>
  )
}

function statusIcon(item: TimelineItem): string {
  if (item.kind === "user") return "›"
  if (item.kind === "assistant") return "A"
  if (item.kind === "aggregate") return "⋯"
  if (item.kind === "guard") return "↻"
  if (item.status === "done") return "✓"
  if (item.status === "failed") return "✕"
  if (item.status === "cancelled") return "−"
  if (item.status === "partial") return "!"
  return "◆"
}

function statusColor(item: TimelineItem, palette: Palette): string {
  if (item.status === "failed" || item.kind === "error") return palette.error
  if (item.status === "done") return palette.success
  if (item.status === "partial" || item.kind === "guard") return palette.warning
  if (item.status === "cancelled") return palette.cancelled
  if (item.kind === "user") return palette.warning
  if (item.kind === "summary" || item.kind === "aggregate" || item.kind === "log") return palette.muted
  return palette.accent
}

function workerStatusColor(status: string, palette: Palette): string {
  if (status === "completed") return palette.success
  if (status === "failed" || status === "timed_out") return palette.error
  if (status === "cancelled" || status === "archived") return palette.cancelled
  if (status === "partial" || status === "waiting_for_context") return palette.warning
  return palette.accent
}

function statusDot(status: string): string {
  if (status === "completed") return "●"
  if (status === "failed" || status === "timed_out") return "●"
  if (status === "cancelled" || status === "archived") return "○"
  if (status === "partial" || status === "waiting_for_context") return "◐"
  return "◒"
}

function workerProgress(worker: WorkerInfo): string {
  if (
    ["completed", "failed", "partial", "cancelled", "timed_out"].includes(worker.status)
  ) {
    const terminal = workerStatusLabel(worker.status)
    return worker.durationMs === undefined
      ? terminal
      : terminal + " · " + formatElapsed(worker.durationMs)
  }
  if (worker.todoTotal !== undefined && worker.todoTotal > 0) {
    return `${worker.todoCompleted ?? 0}/${worker.todoTotal} · ${worker.currentStep ?? worker.phase}`
  }
  if (worker.currentStep) return worker.currentStep
  return worker.phase || workerStatusLabel(worker.status)
}

function workerStatusLabel(status: string): string {
  return (
    {
      cancelled: "cancelled",
      completed: "completed",
      failed: "failed",
      idle: "idle",
      partial: "partial",
      queued: "queued",
      running: "running",
      timed_out: "timed out",
      waiting_for_context: "waiting",
    }[status] ?? status
  )
}

function workerTimelineStatus(status: string): TimelineItem["status"] {
  if (status === "completed") return "done"
  if (status === "cancelled") return "cancelled"
  if (status === "partial" || status === "waiting_for_context") return "partial"
  if (status === "failed" || status === "timed_out") return "failed"
  return "running"
}

function logLevelColor(level: string, palette: Palette): string {
  if (["ERROR", "CRITICAL"].includes(level)) return palette.error
  if (level === "WARNING") return palette.warning
  if (level === "SUCCESS") return palette.success
  return palette.accent
}

function usageLabel(usage: Record<string, unknown>): string {
  const { input, output, total } = usageCounters(usage)
  if (input !== undefined || output !== undefined) return `tokens ${input ?? "?"}/${output ?? "?"}`
  return total !== undefined ? `tokens ${total}` : ""
}

function spinner(now: number): string {
  return ["◐", "◓", "◑", "◒"][Math.floor(now / 120) % 4] ?? "◐"
}

function formatElapsed(milliseconds: number): string {
  if (milliseconds < 1_000) return `${Math.max(0, Math.floor(milliseconds))}ms`
  if (milliseconds >= 60_000) {
    const seconds = Math.floor(milliseconds / 1_000)
    return `${Math.floor(seconds / 60)}m${String(seconds % 60).padStart(2, "0")}s`
  }
  return `${(milliseconds / 1_000).toFixed(1)}s`
}

function compactPath(value: string): string {
  if (!value) return "workspace"
  const parts = value.split(/[\\/]/).filter(Boolean)
  return parts.length <= 3 ? value : `…/${parts.slice(-3).join("/")}`
}

function singleLine(value: string, limit: number): string {
  const text = value.replace(/\s+/g, " ").trim()
  return text.length <= limit ? text : `${text.slice(0, limit)}…`
}

function shortTime(value?: string): string {
  if (!value) return ""
  const parsed = new Date(value)
  return Number.isNaN(parsed.valueOf()) ? "" : parsed.toLocaleTimeString([], { hour12: false })
}

function safeStringify(value: unknown): string {
  try {
    const text = JSON.stringify(value, null, 2) ?? ""
    return text.length <= 2_000 ? text : `${text.slice(0, 2_000)}…`
  } catch {
    return String(value ?? "")
  }
}
