import {
  CliRenderEvents,
  type ScrollBoxRenderable,
  type Selection,
  type ThemeMode,
} from "@opentui/core"
import { useKeyboard, useRenderer, useSelectionHandler } from "@opentui/solid"
import { Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js"
import { createMutable } from "solid-js/store"
import {
  BridgeClient,
  BridgeRequestError,
  type BridgeClientOptions,
} from "./bridge-client"
import { COMMANDS, commandSuggestions, parseCommand } from "./commands"
import { Composer, Header, StatusBar, TranscriptPane } from "./components"
import {
  type AppState,
  type View,
  type WorkerInfo,
  appendSystemNotice,
  appendUserPrompt,
  applyCommandError,
  applyCommandResult,
  applyEnvelope,
  clearMasterTimeline,
  createAppState,
  cycleView,
  hydrateReady,
  hydrateWorkerSnapshot,
  markBridgeFailed,
  markTurnDispatching,
  markTurnCancelled,
  markTurnFailed,
  setFocus,
  setPinned,
  setQueuedPrompts,
  setVerbosity,
  setView,
  toggleVerbosity,
  toggleTimelineItem,
  toggleTurnProcess,
} from "./model"
import { PromptQueue } from "./prompt-queue"
import type { BridgeEnvelope, JsonObject, ReadySnapshot } from "./protocol"
import { DARK, LIGHT, createMarkdownSyntaxStyle } from "./theme"

export interface RuntimeBridge {
  close: () => Promise<void>
  request: (command: string, payload?: JsonObject) => Promise<unknown>
  start: () => void
}

export interface AppProps {
  bridgeFactory?: (options: BridgeClientOptions) => RuntimeBridge
  connect?: boolean
  initialEnvelopes?: BridgeEnvelope[]
  initialSnapshot?: ReadySnapshot
  onExit?: () => void
}

export function App(props: AppProps) {
  const renderer = useRenderer()
  const initial = createAppState()
  if (props.initialSnapshot) hydrateReady(initial, props.initialSnapshot)
  for (const envelope of props.initialEnvelopes ?? []) applyEnvelope(initial, envelope)
  applyStartupPreferences(initial)

  const state = createMutable<AppState>(initial)
  const initialPalette = renderer.themeMode === "light" ? LIGHT : DARK
  const [palette, setPalette] = createSignal(initialPalette)
  let currentMarkdownStyle = createMarkdownSyntaxStyle(initialPalette)
  const [markdownStyle, setMarkdownStyle] = createSignal(currentMarkdownStyle)
  const [composerValue, setComposerValue] = createSignal("")
  const [inputHistory, setInputHistory] = createSignal<string[]>([])
  const [expandedReport, setExpandedReport] = createSignal(false)
  const [revealedMarkdownId, setRevealedMarkdownId] = createSignal<string>()
  const [now, setNow] = createSignal(Date.now())
  const [clipboardStatus, setClipboardStatus] = createSignal("")
  const suggestions = createMemo(() => commandSuggestions(composerValue()))

  let bridge: RuntimeBridge | undefined
  let scrollbox: ScrollBoxRenderable | undefined
  let clipboardTimer: ReturnType<typeof setTimeout> | undefined
  let revealTimer: ReturnType<typeof setTimeout> | undefined
  let exiting = false
  let queueHeldForSessionSwitch = false
  let sessionTransition: Promise<void> | undefined

  const mutate = (change: (draft: AppState) => void) => {
    change(state)
  }

  const protocolError = (message: string) => {
    mutate((draft) => appendSystemNotice(draft, "RUNTIME", message, "error"))
  }

  const bridgeOptions: BridgeClientOptions = {
    onEnvelope: (envelope) => {
      mutate((draft) => {
        applyEnvelope(draft, envelope)
        if (envelope.type === "ready") applyStartupPreferences(draft)
      })
      if (envelope.type === "ready") promptQueue.resume()
      if (envelope.type === "event" && envelope.event === "chat.worker.lifecycle") {
        const eventSessionId = envelope.payload.session_id
        if (
          typeof eventSessionId === "string" &&
          state.sessionId &&
          eventSessionId !== state.sessionId
        ) {
          return
        }
        const phase = String(envelope.payload.phase ?? envelope.payload.status ?? "")
        const workerId = envelope.payload.worker_id
        if (
          typeof workerId === "string" &&
          [
            "cancelled",
            "completed",
            "failed",
            "partial",
            "waiting_for_context",
          ].includes(phase)
        ) {
          void refreshFinishedWorker(
            workerId,
            state.sessionId,
            typeof envelope.payload.run_id === "string" ? envelope.payload.run_id : undefined,
          )
        }
      }
    },
    onFatal: (message) => {
      mutate((draft) => markBridgeFailed(draft, message))
      promptQueue.fail(new Error(message))
    },
    onProtocolError: protocolError,
  }

  async function refreshFinishedWorker(
    workerId: string,
    sessionId: string,
    runId?: string,
  ): Promise<void> {
    if (!bridge) return
    try {
      const result = await bridge.request("inspect_worker", { worker_id: workerId })
      if (state.sessionId !== sessionId) return
      if (runId && state.workers[workerId]?.runId !== runId) return
      mutate((draft) => void hydrateWorkerSnapshot(draft, result))
    } catch {
      // The lifecycle row remains authoritative; an explicit /worker retry can
      // fetch detail later without turning a projection failure into UI noise.
    }
  }

  const requestControl = async (
    command: string,
    payload: JsonObject = {},
    applyResult = true,
  ): Promise<unknown | undefined> => {
    const requestedSessionId = state.sessionId
    if (!bridge) {
      protocolError("Runtime bridge is still starting. Your input remains available.")
      return undefined
    }
    try {
      const result = await bridge.request(command, payload)
      const scopedResult = [
        "cancel_worker",
        "inspect_worker",
        "list_workers",
        "resume_worker",
        "spawn_worker",
      ].includes(command)
      if (
        applyResult &&
        (!scopedResult || state.sessionId === requestedSessionId)
      ) {
        mutate((draft) => applyCommandResult(draft, command, result))
      }
      return result
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      mutate((draft) => applyCommandError(draft, command, message))
      return undefined
    }
  }

  const promptQueue = new PromptQueue(
    (prompt) => {
      if (!bridge) return Promise.reject(new Error("Runtime bridge is not available."))
      return bridge.request("prompt", { prompt })
    },
    {
      onChanged: (prompts) => mutate((draft) => setQueuedPrompts(draft, prompts)),
      onDispatch: (prompt) =>
        mutate((draft) => {
          appendUserPrompt(draft, prompt)
          markTurnDispatching(draft)
        }),
      onError: (_prompt, error) => {
        if (error instanceof BridgeRequestError && error.code === "turn_cancelled") {
          mutate(markTurnCancelled)
          return
        }
        const message = error instanceof Error ? error.message : String(error)
        mutate((draft) => markTurnFailed(draft, message))
      },
    },
    !state.bridgeReady,
  )

  const selectedWorker = (): WorkerInfo | undefined => {
    const view = state.view
    return view.kind === "worker" ? state.workers[view.workerId] : undefined
  }

  const findWorker = (value: string): WorkerInfo | undefined => {
    const query = value.trim().toLowerCase()
    if (!query) return selectedWorker()
    return state.workerOrder
      .map((id) => state.workers[id])
      .find((worker) => {
        if (!worker) return false
        return [worker.id, worker.label, worker.workerTypeId, worker.runId]
          .filter(Boolean)
          .some(
            (candidate) =>
              candidate?.toLowerCase() === query || candidate?.toLowerCase().startsWith(query),
          )
      })
  }

  const refuseSessionSwitchWhileBusy = (): boolean => {
    if (
      !state.running &&
      !promptQueue.active &&
      (state.queuedPrompts.length === 0 || queueHeldForSessionSwitch)
    ) return false
    mutate((draft) =>
      appendSystemNotice(
        draft,
        "SESSION BUSY",
        "Cancel or finish the active turn and queued prompts before changing sessions.",
        "error",
      ),
    )
    return true
  }

  const openWorker = (worker: WorkerInfo) => {
    const requestedSessionId = state.sessionId
    const requestedRunId = worker.runId
    mutate((draft) => {
      setView(draft, { kind: "worker", workerId: worker.id })
      setFocus(draft, "transcript")
    })
    void requestControl("inspect_worker", { worker_id: worker.id }, false).then((result) => {
      if (result === undefined || state.sessionId !== requestedSessionId) return
      const currentRunId = state.workers[worker.id]?.runId
      if (requestedRunId && currentRunId && currentRunId !== requestedRunId) return
      mutate((draft) => void hydrateWorkerSnapshot(draft, result))
    })
  }

  const executeCommand = async (source: string) => {
    const command = parseCommand(source)
    if (!command) return
    if (
      sessionTransition &&
      !["clear", "help", "logs", "master", "quit", "verbosity"].includes(command.name)
    ) await sessionTransition
    const argument = command.args.join(" ")

    switch (command.name) {
      case "help":
        mutate((draft) =>
          appendSystemNotice(
            draft,
            "COMMANDS",
            COMMANDS.map((item) => `${item.usage.padEnd(31)} ${item.description}`).join("\n"),
          ),
        )
        return
      case "verbosity": {
        const requested = command.args[0]?.toLowerCase()
        if (requested && requested !== "compact" && requested !== "verbose") {
          mutate((draft) =>
            appendSystemNotice(draft, "VERBOSITY", "Use compact or verbose.", "error"),
          )
          return
        }
        mutate((draft) => {
          const next =
            requested === "verbose"
              ? "verbose"
              : requested === "compact"
                ? "compact"
                : draft.verbosity === "compact"
                  ? "verbose"
                  : "compact"
          setVerbosity(draft, next)
          appendSystemNotice(draft, "VERBOSITY", `Transcript is now ${next}.`)
        })
        return
      }
      case "logs": {
        const mode = command.args[0]?.toLowerCase() || "on"
        mutate((draft) => {
          if (mode === "off") {
            draft.logDetail = false
            setView(draft, { kind: "master" })
          } else {
            draft.logDetail = mode === "detail"
            setView(draft, { kind: "logs" })
          }
        })
        return
      }
      case "master":
        mutate((draft) => setView(draft, { kind: "master" }))
        return
      case "clear":
        mutate(clearMasterTimeline)
        return
      case "workers":
        await requestControl("list_workers", { session_id: state.sessionId })
        return
      case "worker": {
        const worker = findWorker(argument)
        if (!worker) {
          mutate((draft) =>
            appendSystemNotice(draft, "WORKER", "No matching Worker. Use /workers first.", "error"),
          )
          return
        }
        openWorker(worker)
        return
      }
      case "worker-types":
        await requestControl("discover_worker_types")
        return
      case "sessions":
        await requestControl("list_sessions")
        return
      case "new":
        if (!refuseSessionSwitchWhileBusy()) await switchSession("new_session")
        return
      case "resume":
        if (!command.args[0]) {
          mutate((draft) =>
            appendSystemNotice(draft, "RESUME", "Provide a session from /sessions.", "error"),
          )
          return
        }
        if (!refuseSessionSwitchWhileBusy()) {
          await switchSession("resume_session", { session_id: command.args[0] })
        }
        return
      case "spawn": {
        const [workerTypeId, ...objectiveParts] = command.args
        const objective = objectiveParts.join(" ")
        if (!workerTypeId || !objective) {
          mutate((draft) =>
            appendSystemNotice(
              draft,
              "SPAWN",
              "Use /spawn <worker-type> <objective>.",
              "error",
            ),
          )
          return
        }
        await requestControl("spawn_worker", {
          objective,
          session_id: state.sessionId,
          worker_type_id: workerTypeId,
        })
        return
      }
      case "cancel": {
        const worker = findWorker(command.args[0] ?? "")
        const runId = worker?.runId ?? command.args[0]
        if (!runId) {
          mutate((draft) =>
            appendSystemNotice(draft, "CANCEL", "Select a running Worker first.", "error"),
          )
          return
        }
        await requestControl("cancel_worker", { run_id: runId })
        return
      }
      case "resume-worker": {
        const worker = selectedWorker()
        const response = command.args.join(" ")
        if (!worker?.runId) {
          mutate((draft) =>
            appendSystemNotice(
              draft,
              "RESUME WORKER",
              "Open a waiting Worker first, then provide its requested context.",
              "error",
            ),
          )
          return
        }
        if (worker.status !== "waiting_for_context") {
          mutate((draft) =>
            appendSystemNotice(
              draft,
              "RESUME WORKER",
              "Only a Worker waiting for context can be resumed.",
              "error",
            ),
          )
          return
        }
        if (!response) {
          mutate((draft) =>
            appendSystemNotice(
              draft,
              "RESUME WORKER",
              "Provide a response to the Worker's waiting question.",
              "error",
            ),
          )
          return
        }
        await requestControl("resume_worker", {
          response,
          run_id: worker.runId,
        })
        return
      }
      case "cancel-turn": {
        const result = await requestControl("cancel_turn", {}, false)
        if (result === undefined) return
        const cancelled = Boolean(
          result && typeof result === "object" && "cancelled" in result && result.cancelled,
        )
        mutate((draft) =>
          appendSystemNotice(
            draft,
            "TURN",
            cancelled ? "Cancellation requested." : "No active Master turn.",
          ),
        )
        return
      }
      case "quit":
        await exitApp()
        return
      default:
        mutate((draft) =>
          appendSystemNotice(
            draft,
            "COMMAND",
            `Unknown command /${command.name}. Try /help.`,
            "error",
          ),
        )
    }
  }

  const switchSession = async (command: "new_session" | "resume_session", payload: JsonObject = {}) => {
    promptQueue.pause()
    mutate((draft) => {
      draft.sessionSwitching = true
    })
    let switched = false
    const operation = requestControl(command, payload).then((result) => {
      switched = result !== undefined
    })
    sessionTransition = operation
    try {
      await operation
    } finally {
      sessionTransition = undefined
      mutate((draft) => {
        draft.sessionSwitching = false
      })
      if (switched) {
        queueHeldForSessionSwitch = false
        promptQueue.resume()
      } else {
        queueHeldForSessionSwitch = true
        mutate((draft) =>
          appendSystemNotice(
            draft,
            "SESSION HELD",
            "Session did not change. Queued prompts remain held; retry /new or /resume.",
            "error",
          ),
        )
      }
    }
  }

  const submit = (value: string) => {
    setInputHistory((history) => [...history, value])
    if (value.trimStart().startsWith("/")) {
      void executeCommand(value)
      return
    }
    promptQueue.submit(value)
  }

  const chooseView = (view: View) => {
    if (view.kind === "worker") {
      const worker = state.workers[view.workerId]
      if (worker) openWorker(worker)
      return
    }
    mutate((draft) => {
      setView(draft, view)
      setFocus(draft, "transcript")
    })
  }

  const togglePin = () => {
    const pin = !state.pinToBottom
    mutate((draft) => setPinned(draft, pin))
    if (pin) queueMicrotask(() => scrollbox?.scrollTo({ x: 0, y: scrollbox.scrollHeight }))
  }

  const revealMarkdown = (itemId: string) => {
    setRevealedMarkdownId(itemId)
    if (revealTimer) clearTimeout(revealTimer)
    revealTimer = setTimeout(() => setRevealedMarkdownId(undefined), 4_000)
  }

  const exitApp = async () => {
    if (exiting) return
    exiting = true
    promptQueue.stop()
    await bridge?.close().catch(() => undefined)
    props.onExit?.()
    renderer.destroy()
  }

  useKeyboard((key) => {
    if (key.ctrl && (key.name === "d" || key.name === "c")) {
      key.preventDefault()
      if (key.name === "c" && state.running) void executeCommand("/cancel-turn")
      else void exitApp()
      return
    }
    if (key.name === "escape") {
      key.preventDefault()
      mutate((draft) => {
        if (draft.focus === "composer") setFocus(draft, "transcript")
        else if (draft.view.kind !== "master") setView(draft, { kind: "master" })
        else setFocus(draft, "composer")
      })
      return
    }
    if (state.focus !== "transcript") return
    if (key.name === "tab") {
      key.preventDefault()
      mutate((draft) => cycleView(draft, key.shift ? -1 : 1))
      return
    }
    if (key.name === "return" || key.name === "i") {
      key.preventDefault()
      mutate((draft) => setFocus(draft, "composer"))
      return
    }
    if (key.name === "v") {
      key.preventDefault()
      mutate(toggleVerbosity)
      return
    }
    if (key.name === "p") {
      key.preventDefault()
      togglePin()
      return
    }
    if (key.name === "t" && state.view.kind === "master") {
      key.preventDefault()
      mutate((draft) => toggleTurnProcess(draft))
      return
    }
    if (key.name === "m") {
      key.preventDefault()
      chooseView({ kind: "master" })
      return
    }
    if (key.name === "l") {
      key.preventDefault()
      chooseView({ kind: "logs" })
      return
    }
    if (state.view.kind === "worker" && key.name === "c") {
      key.preventDefault()
      void executeCommand("/cancel")
      return
    }
    if (state.view.kind === "worker" && key.name === "r") {
      key.preventDefault()
      if (selectedWorker()?.status !== "waiting_for_context") {
        mutate((draft) =>
          appendSystemNotice(
            draft,
            "RESUME WORKER",
            "This Worker is not waiting for context.",
            "error",
          ),
        )
        return
      }
      setComposerValue("/resume-worker ")
      mutate((draft) => setFocus(draft, "composer"))
      return
    }
    if (state.view.kind === "worker" && key.name === "e") {
      key.preventDefault()
      setExpandedReport((value) => !value)
    }
  })

  useSelectionHandler((selection: Selection) => {
    const text = selection.getSelectedText()
    if (text) {
      const copied = renderer.copyToClipboardOSC52(text)
      setClipboardStatus(copied ? `copied ${text.length}` : "selected")
      if (clipboardTimer) clearTimeout(clipboardTimer)
      clipboardTimer = setTimeout(() => setClipboardStatus(""), 1_500)
    }
    queueMicrotask(() => setRevealedMarkdownId(undefined))
  })

  const applyTheme = (mode: ThemeMode | null) => {
    const next = mode === "light" ? LIGHT : DARK
    if (next !== palette()) {
      const previousStyle = currentMarkdownStyle
      currentMarkdownStyle = createMarkdownSyntaxStyle(next)
      setMarkdownStyle(currentMarkdownStyle)
      setPalette(next)
      queueMicrotask(() => previousStyle.destroy())
    }
    renderer.setBackgroundColor(next.background)
  }
  const themeHandler = (mode: ThemeMode) => applyTheme(mode)
  renderer.on(CliRenderEvents.THEME_MODE, themeHandler)

  createEffect(() => {
    state.view
    setExpandedReport(false)
  })

  onMount(() => {
    renderer.setTerminalTitle("Aeloon · agent workspace")
    applyTheme(renderer.themeMode)
    void renderer.waitForThemeMode(250).then(applyTheme)
    const clock = setInterval(() => setNow(Date.now()), 100)
    onCleanup(() => clearInterval(clock))

    if (props.connect !== false) {
      const factory =
        props.bridgeFactory ?? ((options: BridgeClientOptions) => new BridgeClient(options))
      bridge = factory(bridgeOptions)
      bridge.start()
    }
  })

  onCleanup(() => {
    renderer.off(CliRenderEvents.THEME_MODE, themeHandler)
    if (clipboardTimer) clearTimeout(clipboardTimer)
    if (revealTimer) clearTimeout(revealTimer)
    currentMarkdownStyle.destroy()
    promptQueue.stop()
    if (!exiting) void bridge?.close()
  })

  return (
    <box
      width="100%"
      height="100%"
      flexDirection="column"
      backgroundColor={palette().background}
    >
      <Header palette={palette()} state={state} />
      <box width="100%" height="100%" minHeight={0} flexGrow={1}>
        <TranscriptPane
          expandedReport={expandedReport()}
          focus={state.focus === "transcript"}
          markdownStyle={markdownStyle()}
          onFocus={() => mutate((draft) => setFocus(draft, "transcript"))}
          onPinChange={(pinned) => mutate((draft) => setPinned(draft, pinned))}
          onRevealMarkdown={revealMarkdown}
          onSelectWorker={openWorker}
          onScrollRef={(value) => (scrollbox = value)}
          onToggleReport={() => setExpandedReport((value) => !value)}
          onToggleTimelineItem={(itemId) => mutate((draft) => toggleTimelineItem(draft, itemId))}
          onToggleTurnProcess={(turnId) => mutate((draft) => toggleTurnProcess(draft, turnId))}
          palette={palette()}
          revealedMarkdownId={revealedMarkdownId()}
          state={state}
        />
      </box>
      <StatusBar
        clipboardStatus={clipboardStatus()}
        now={now()}
        palette={palette()}
        state={state}
      />
      <Composer
        draft={composerValue()}
        focused={state.focus === "composer"}
        history={inputHistory()}
        onFocus={() => mutate((draft) => setFocus(draft, "composer"))}
        onFocusTranscript={() => mutate((draft) => setFocus(draft, "transcript"))}
        onSubmit={submit}
        onValueChange={setComposerValue}
        palette={palette()}
        queue={state.queuedPrompts}
        suggestions={suggestions()}
      />
    </box>
  )
}

function applyStartupPreferences(state: AppState): void {
  if (process.env.AELOON_CORE_TUI_INITIAL_VERBOSITY === "verbose") {
    state.verbosity = "verbose"
  }
  if (process.env.AELOON_CORE_TUI_INITIAL_VIEW === "logs") {
    state.view = { kind: "logs" }
  }
  if (process.env.AELOON_CORE_TUI_LOG_DETAIL === "1") state.logDetail = true
}
