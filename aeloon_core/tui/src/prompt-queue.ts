export interface PromptQueueCallbacks {
  onAccepted?: (prompt: string, queued: boolean) => void
  onChanged?: (queuedPrompts: string[]) => void
  onDispatch?: (prompt: string) => void
  onError?: (prompt: string, error: unknown) => void
}

/** A single-flight FIFO. Session turns must never overlap in the Python runtime. */
export class PromptQueue {
  private activePrompt: string | undefined
  private readonly callbacks: PromptQueueCallbacks
  private readonly pending: string[] = []
  private readonly runPrompt: (prompt: string) => Promise<unknown>
  private paused: boolean
  private stopped = false
  private terminalError: unknown

  constructor(
    runPrompt: (prompt: string) => Promise<unknown>,
    callbacks: PromptQueueCallbacks = {},
    startPaused = false,
  ) {
    this.runPrompt = runPrompt
    this.callbacks = callbacks
    this.paused = startPaused
  }

  get active(): string | undefined {
    return this.activePrompt
  }

  get queued(): readonly string[] {
    return this.pending
  }

  submit(prompt: string): boolean {
    if (!prompt.trim() || this.stopped) return false
    if (this.terminalError !== undefined) {
      this.callbacks.onAccepted?.(prompt, true)
      this.pending.push(prompt)
      this.emitChanged()
      return true
    }
    const queued = Boolean(this.activePrompt) || this.paused
    this.callbacks.onAccepted?.(prompt, queued)
    if (queued) {
      this.pending.push(prompt)
      this.emitChanged()
    } else {
      void this.dispatch(prompt)
    }
    return true
  }

  clearPending(): void {
    this.pending.splice(0)
    this.emitChanged()
  }

  pause(): void {
    if (!this.stopped) this.paused = true
  }

  resume(): void {
    if (!this.paused || this.stopped || this.terminalError !== undefined) return
    this.paused = false
    if (this.activePrompt) return
    const next = this.pending.shift()
    this.emitChanged()
    if (next) void this.dispatch(next)
  }

  stop(): void {
    this.stopped = true
    this.clearPending()
  }

  /** Pause after a fatal transport error while retaining prompts for recovery or copying. */
  fail(error: unknown): void {
    if (this.terminalError !== undefined || this.stopped) return
    this.terminalError = error
    this.paused = true
    this.emitChanged()
  }

  private async dispatch(prompt: string): Promise<void> {
    if (this.stopped) return
    this.activePrompt = prompt
    this.callbacks.onDispatch?.(prompt)
    try {
      await this.runPrompt(prompt)
    } catch (error) {
      this.callbacks.onError?.(prompt, error)
    } finally {
      this.activePrompt = undefined
      if (this.terminalError !== undefined) {
        this.emitChanged()
        return
      }
      const next = this.pending.shift()
      this.emitChanged()
      if (next && !this.stopped) void this.dispatch(next)
    }
  }

  private emitChanged(): void {
    this.callbacks.onChanged?.([...this.pending])
  }
}
