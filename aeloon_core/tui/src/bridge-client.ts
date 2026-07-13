import {
  closeSync,
  createReadStream,
  createWriteStream,
  type ReadStream,
  type WriteStream,
} from "node:fs"
import type { BridgeCommand, BridgeEnvelope, JsonObject } from "./protocol"
import { isBridgeEnvelope } from "./protocol"

interface PendingRequest {
  command: string
  reject: (error: Error) => void
  resolve: (result: unknown) => void
}

export interface BridgeClientOptions {
  onEnvelope: (envelope: BridgeEnvelope) => void
  onFatal?: (message: string) => void
  onProtocolError?: (message: string) => void
  pythonExecutable?: string
}

export class BridgeClient {
  private child?: Bun.PipedSubprocess
  private closed = false
  private lineBuffer = ""
  private readonly onEnvelope: (envelope: BridgeEnvelope) => void
  private readonly onFatal?: (message: string) => void
  private readonly onProtocolError?: (message: string) => void
  private readonly pending = new Map<string, PendingRequest>()
  private readonly pythonExecutable: string
  private requestSequence = 0
  private inheritedFd?: number
  private socketReader?: ReadStream
  private socketWriter?: WriteStream

  constructor(options: BridgeClientOptions) {
    this.onEnvelope = options.onEnvelope
    this.onFatal = options.onFatal
    this.onProtocolError = options.onProtocolError
    this.pythonExecutable =
      options.pythonExecutable ?? process.env.AELOON_CORE_PYTHON ?? "python3"
  }

  start(): void {
    if (this.child || this.socketReader) return
    const inheritedFdValue = process.env.AELOON_CORE_TUI_BRIDGE_FD
    delete process.env.AELOON_CORE_TUI_BRIDGE_FD
    const inheritedFd = inheritedFdValue && /^\d+$/.test(inheritedFdValue)
      ? Number(inheritedFdValue)
      : Number.NaN
    if (Number.isInteger(inheritedFd) && inheritedFd >= 0) {
      try {
        this.inheritedFd = inheritedFd
        const reader = createReadStream("", { autoClose: false, fd: inheritedFd })
        const writer = createWriteStream("", { autoClose: false, fd: inheritedFd })
        this.socketReader = reader
        this.socketWriter = writer
        reader.setEncoding("utf8")
        reader.on("data", (chunk: string) => this.consumeText(chunk))
        reader.once("error", (error) =>
          this.failTransport(`Runtime bridge connection failed: ${asError(error).message}`),
        )
        writer.once("error", (error) =>
          this.failTransport(`Runtime bridge connection failed: ${asError(error).message}`),
        )
        reader.once("end", () => this.failTransport("Runtime bridge connection closed."))
        reader.resume()
        return
      } catch (error) {
        this.closeInheritedFd()
        this.failTransport(`Could not attach to the runtime bridge: ${asError(error).message}`)
        return
      }
    }

    const childEnvironment = { ...process.env }
    this.child = Bun.spawn([this.pythonExecutable, "-m", "aeloon_core.tui_bridge"], {
      cwd: process.env.AELOON_CORE_WORKSPACE ?? process.cwd(),
      env: childEnvironment,
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    })
    void this.readStdout(this.child.stdout)
    void this.readStderr(this.child.stderr)
    void this.watchExit()
  }

  request(command: string, payload: JsonObject = {}): Promise<unknown> {
    if ((!this.child && !this.socketWriter) || this.closed) {
      return Promise.reject(new Error("Runtime bridge is not available."))
    }
    const requestId = `ui-${++this.requestSequence}`
    const message: BridgeCommand = {
      command,
      payload,
      request_id: requestId,
      type: "command",
    }
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { command, reject, resolve })
      try {
        this.writeLine(message)
      } catch (error) {
        this.pending.delete(requestId)
        reject(asError(error))
      }
    })
  }

  async close(): Promise<void> {
    if (this.closed) return
    const shutdown = {
      command: "shutdown",
      payload: {},
      request_id: "ui-shutdown",
      type: "command",
    }
    try {
      this.writeLine(shutdown)
    } catch {
      // The bridge may already have exited.
    }
    this.closed = true
    if (this.socketReader && this.socketWriter) {
      const reader = this.socketReader
      const writer = this.socketWriter
      const closed = reader.readableEnded
        ? Promise.resolve()
        : new Promise<void>((resolve) => reader.once("end", resolve))
      writer.end()
      let timer: ReturnType<typeof setTimeout> | undefined
      const timeout = new Promise<void>((resolve) => {
        timer = setTimeout(resolve, 600)
      })
      await Promise.race([closed, timeout])
      clearTimeout(timer)
      reader.destroy()
      writer.destroy()
      this.closeInheritedFd()
    }
    if (this.child) {
      try {
        this.child.stdin.end()
      } catch {
        // The bridge may already have exited.
      }
      const timer = setTimeout(() => this.child?.kill(), 600)
      await this.child.exited.catch(() => undefined)
      clearTimeout(timer)
    }
    this.rejectPending(new Error("Runtime bridge closed."))
  }

  private async readStdout(stream: ReadableStream<Uint8Array>): Promise<void> {
    const reader = stream.getReader()
    const decoder = new TextDecoder()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        this.consumeText(decoder.decode(value, { stream: true }))
      }
      this.consumeText(decoder.decode(), true)
    } catch (error) {
      if (!this.closed) this.onProtocolError?.(`Bridge output failed: ${asError(error).message}`)
    }
  }

  private consumeText(text: string, final = false): void {
    this.lineBuffer += text
    let newline = this.lineBuffer.indexOf("\n")
    while (newline >= 0) {
      const line = this.lineBuffer.slice(0, newline).trim()
      this.lineBuffer = this.lineBuffer.slice(newline + 1)
      if (line) this.handleLine(line)
      newline = this.lineBuffer.indexOf("\n")
    }
    if (final && this.lineBuffer.trim()) this.handleLine(this.lineBuffer.trim())
    if (final) this.lineBuffer = ""
  }

  private async readStderr(stream: ReadableStream<Uint8Array>): Promise<void> {
    const text = await new Response(stream).text().catch(() => "")
    const clean = text.trim()
    if (clean && !this.closed) this.onProtocolError?.(`Runtime bridge: ${clean.slice(0, 500)}`)
  }

  private handleLine(line: string): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch {
      this.onProtocolError?.(`Ignored malformed bridge output: ${line.slice(0, 160)}`)
      return
    }
    if (!isBridgeEnvelope(parsed)) {
      this.onProtocolError?.("Ignored an unknown bridge message.")
      return
    }
    this.onEnvelope(parsed)
    if (parsed.type !== "response") return
    const pending = this.pending.get(parsed.request_id)
    if (!pending) return
    this.pending.delete(parsed.request_id)
    if (parsed.ok) {
      pending.resolve(parsed.result)
    } else {
      pending.reject(
        new BridgeRequestError(
          pending.command,
          parsed.error?.message ?? "Command failed.",
          parsed.error?.code,
        ),
      )
    }
  }

  private async watchExit(): Promise<void> {
    if (!this.child) return
    const exitCode = await this.child.exited
    this.failTransport(`Runtime bridge exited with code ${exitCode}.`)
  }

  private writeLine(value: object): void {
    const line = `${JSON.stringify(value)}\n`
    if (this.socketWriter) {
      this.socketWriter.write(line)
      return
    }
    if (!this.child) throw new Error("Runtime bridge is not available.")
    this.child.stdin.write(line)
    this.child.stdin.flush()
  }

  private failTransport(message: string): void {
    if (this.closed) return
    this.closed = true
    this.socketReader?.destroy()
    this.socketWriter?.destroy()
    this.closeInheritedFd()
    const error = new Error(message)
    this.rejectPending(error)
    if (this.onFatal) this.onFatal(message)
    else this.onProtocolError?.(message)
  }

  private closeInheritedFd(): void {
    if (this.inheritedFd === undefined) return
    try {
      closeSync(this.inheritedFd)
    } catch {
      // The transport may already have been closed by the runtime.
    }
    this.inheritedFd = undefined
  }

  private rejectPending(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
  }
}

export class BridgeRequestError extends Error {
  readonly command: string
  readonly code?: string

  constructor(command: string, message: string, code?: string) {
    super(message)
    this.name = "BridgeRequestError"
    this.command = command
    this.code = code
  }
}

function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value))
}
