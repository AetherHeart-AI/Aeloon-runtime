import {
  AlertTriangle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Copy,
  FileText,
  History,
  Loader2,
  MessageSquareText,
  PanelRight,
  Plus,
  RefreshCw,
  Send,
  TerminalSquare,
  Trash2,
  UserRound,
  XCircle,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Tab = "chat" | "logs";
type ConnectionState = "connecting" | "connected" | "disconnected";
type LogLevelFilter = "DEBUG" | "INFO" | "WARN" | "ERROR";

type Block = {
  id: string;
  type: "text" | "tool_call" | "reasoning";
  role?: string;
  content?: string;
  name?: string;
  arguments?: unknown;
  result?: string | null;
  status?: "running" | "done" | "error";
  created_at?: string;
};

type Turn = {
  id: string;
  sessionId: string;
  user: string;
  blocks: Block[];
  final?: string;
  status: "running" | "done" | "error";
};

type LogEntry = {
  level: string;
  message: string;
  source: string;
  sessionId?: string;
  ts: string;
  detail?: unknown;
};

type ReasoningEntry = {
  timestamp: string;
  kind: string;
  text: string;
  callId?: string;
  toolName?: string;
  arguments?: unknown;
  status?: string;
  result?: string;
  resultLength?: number;
};

type SessionSummary = {
  session_id: string;
  title: string;
  updated_at: string;
  turns: number;
};

type RpcResponse = {
  type: "response";
  id: string;
  result?: unknown;
  error?: { code: string; message: string };
};

type EventEnvelope = {
  type: "event";
  event: string;
  payload: Record<string, unknown>;
};

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
};

const MAX_LOGS = 2000;
const TOOL_RESULT_PREVIEW_CHARS = 1400;
const GLOBAL_LOG_KEY = "__global__";
const LOG_LEVEL_FILTERS: LogLevelFilter[] = ["DEBUG", "INFO", "WARN", "ERROR"];
const DEFAULT_LOG_LEVELS: LogLevelFilter[] = ["INFO", "WARN", "ERROR"];

export function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [logBuckets, setLogBuckets] = useState<Record<string, LogEntry[]>>({});
  const [visibleLogLevels, setVisibleLogLevels] = useState<LogLevelFilter[]>(DEFAULT_LOG_LEVELS);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const pendingRef = useRef<Map<string, PendingRequest>>(new Map());
  const pendingPromptRef = useRef("");
  const requestCounter = useRef(0);
  const reconnectTimer = useRef<number | null>(null);

  const pushLog = useCallback((entry: LogEntry) => {
    const key = logBucketKey(entry.sessionId);
    setLogBuckets((current) => ({
      ...current,
      [key]: [entry, ...(current[key] ?? [])].slice(0, MAX_LOGS),
    }));
  }, []);

  const logs = activeSessionId ? (logBuckets[activeSessionId] ?? []) : [];
  const visibleLogs = useMemo(
    () => logs.filter((log) => visibleLogLevels.includes(normalizeLogLevel(log.level))),
    [logs, visibleLogLevels],
  );

  const handleEvent = useCallback(
    (envelope: EventEnvelope) => {
      const payload = envelope.payload;
      if (envelope.event === "log.entry") {
        const sessionId = extractLogSessionId(payload);
        if (!sessionId) {
          return;
        }
        pushLog({
          level: String(payload.level ?? "INFO"),
          message: String(payload.message ?? ""),
          source: String(payload.source ?? "runtime"),
          sessionId,
          ts: String(payload.ts ?? new Date().toISOString()),
          detail: payload.detail ?? payload,
        });
        return;
      }

      if (envelope.event === "chat.turn.start") {
        const turnId = String(payload.turn_id ?? crypto.randomUUID());
        const sessionId = String(payload.session_id ?? "");
        setActiveSessionId(sessionId);
        setTurns((current) => [
          ...current,
          {
            id: turnId,
            sessionId,
            user: pendingPromptRef.current,
            blocks: [],
            status: "running",
          },
        ]);
        return;
      }

      if (envelope.event === "chat.block.add") {
        const turnId = String(payload.turn_id ?? "");
        const block = payload.block as Block;
        setTurns((current) =>
          current.map((turn) =>
            turn.id === turnId ? { ...turn, blocks: upsertBlock(turn.blocks, block) } : turn,
          ),
        );
        return;
      }

      if (envelope.event === "chat.block.delta") {
        const turnId = String(payload.turn_id ?? "");
        const blockId = String(payload.block_id ?? "");
        const delta = String(payload.delta ?? "");
        setTurns((current) =>
          current.map((turn) =>
            turn.id === turnId
              ? {
                  ...turn,
                  blocks: turn.blocks.map((block) =>
                    block.id === blockId
                      ? { ...block, content: `${block.content ?? ""}${delta}` }
                      : block,
                  ),
                }
              : turn,
          ),
        );
        return;
      }

      if (envelope.event === "chat.reasoning.delta") {
        const turnId = String(payload.turn_id ?? "");
        const delta = String(payload.delta ?? "");
        setTurns((current) =>
          current.map((turn) =>
            turn.id === turnId
              ? {
                  ...turn,
                  blocks: upsertBlockDelta(turn.blocks, {
                    id: `reasoning-${turnId}`,
                    type: "reasoning",
                    status: "running",
                    content: delta,
                  }),
                }
              : turn,
          ),
        );
        return;
      }

      if (envelope.event === "chat.block.update") {
        const turnId = String(payload.turn_id ?? "");
        const blockId = String(payload.block_id ?? "");
        const patch = payload.patch as Partial<Block>;
        setTurns((current) =>
          current.map((turn) =>
            turn.id === turnId
              ? {
                  ...turn,
                  blocks: turn.blocks.map((block) =>
                    block.id === blockId ? { ...block, ...patch } : block,
                  ),
                }
              : turn,
          ),
        );
        return;
      }

      if (envelope.event === "chat.turn.end") {
        const turnId = String(payload.turn_id ?? "");
        setTurns((current) =>
          current.map((turn) =>
            turn.id === turnId
              ? { ...turn, final: String(payload.final ?? ""), status: "done" }
              : turn,
          ),
        );
        setIsSending(false);
      }
    },
    [pushLog],
  );

  const sendRpc = useCallback((method: string, params: Record<string, unknown> = {}) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("WebSocket is not connected"));
    }
    const id = `rpc-${++requestCounter.current}`;
    const promise = new Promise((resolve, reject) => {
      pendingRef.current.set(id, { resolve, reject });
    });
    ws.send(JSON.stringify({ id, method, params }));
    return promise;
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const result = (await sendRpc("session.list")) as { sessions?: SessionSummary[] };
      setSessions(result.sessions ?? []);
    } catch {
      setSessions([]);
    }
  }, [sendRpc]);

  const connect = useCallback(() => {
    setConnection("connecting");
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnection("connected");
      loadSessions();
    };

    ws.onmessage = (message) => {
      const payload = JSON.parse(message.data) as RpcResponse | EventEnvelope;
      if (payload.type === "event") {
        handleEvent(payload);
        return;
      }
      const pending = pendingRef.current.get(payload.id);
      if (!pending) {
        return;
      }
      pendingRef.current.delete(payload.id);
      if (payload.error) {
        pending.reject(new Error(payload.error.message));
      } else {
        pending.resolve(payload.result);
      }
    };

    ws.onclose = () => {
      setConnection("disconnected");
      if (reconnectTimer.current === null) {
        reconnectTimer.current = window.setTimeout(() => {
          reconnectTimer.current = null;
          connect();
        }, 1500);
      }
    };

    ws.onerror = () => {
      setConnection("disconnected");
    };
  }, [handleEvent, loadSessions]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current !== null) {
        window.clearTimeout(reconnectTimer.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  const createSession = async () => {
    const result = (await sendRpc("session.new")) as { session_id: string };
    setActiveSessionId(result.session_id);
    setTurns([]);
    setLogBuckets((current) => ({ ...current, [result.session_id]: [] }));
    await loadSessions();
  };

  const resumeSession = async (sessionId: string) => {
    const result = (await sendRpc("session.resume", { session_id: sessionId })) as {
      history?: Array<{
        user_prompt?: string;
        final_content?: string;
        blocks?: Block[];
        created_at?: string;
      }>;
    };
    setActiveSessionId(sessionId);
    setTurns(
      (result.history ?? []).map((record, index) => ({
        id: `${sessionId}-${index}`,
        sessionId,
        user: String(record.user_prompt ?? ""),
        final: String(record.final_content ?? ""),
        blocks: record.blocks ?? [],
        status: "done",
      })),
    );
  };

  const deleteSession = async (sessionId: string) => {
    await sendRpc("session.delete", { session_id: sessionId });
    setLogBuckets((current) => {
      const next = { ...current };
      delete next[sessionId];
      return next;
    });
    if (activeSessionId === sessionId) {
      setActiveSessionId("");
      setTurns([]);
    }
    await loadSessions();
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const message = input.trim();
    if (!message || isSending) {
      return;
    }
    pendingPromptRef.current = message;
    setInput("");
    setIsSending(true);
    try {
      const result = (await sendRpc("chat.send", {
        message,
        session_id: activeSessionId || undefined,
      })) as { session_id?: string };
      if (result.session_id) {
        setActiveSessionId(result.session_id);
      }
      await loadSessions();
    } catch (error) {
      void error;
      setIsSending(false);
    }
  };

  const toggleLogLevel = (level: LogLevelFilter) => {
    setVisibleLogLevels((current) =>
      current.includes(level)
        ? current.filter((item) => item !== level)
        : [...current, level],
    );
  };

  const toolCount = useMemo(
    () => turns.reduce((count, turn) => count + turn.blocks.filter((b) => b.type === "tool_call").length, 0),
    [turns],
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Bot size={18} />
          </div>
          <div>
            <h1>Aeloon Core</h1>
            <span>local loop workbench</span>
          </div>
        </div>
        <nav className="tabs" aria-label="Primary">
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>
            <MessageSquareText size={16} />
            Chat
          </button>
          <button className={tab === "logs" ? "active" : ""} onClick={() => setTab("logs")}>
            <FileText size={16} />
            Logs
          </button>
        </nav>
        <div className="top-actions">
          <span className={`connection ${connection}`}>
            <span />
            {connection === "connected" ? "Connected" : connection}
          </span>
          <button className="icon-button" onClick={loadSessions} title="Refresh sessions">
            <RefreshCw size={16} />
          </button>
          <button className="primary-action" onClick={createSession}>
            <Plus size={16} />
            New session
          </button>
        </div>
      </header>

      <aside className="session-rail">
        <SectionTitle icon={<History size={15} />} title="Session history" />
        <div className="session-list">
          {sessions.length === 0 ? (
            <p className="empty-note">No saved turns yet.</p>
          ) : (
            sessions.map((session) => (
              <button
                key={session.session_id}
                className={`session-row ${activeSessionId === session.session_id ? "selected" : ""}`}
                onClick={() => resumeSession(session.session_id)}
              >
                <span>{session.title}</span>
                <small>{session.turns} turn{session.turns === 1 ? "" : "s"}</small>
              </button>
            ))
          )}
        </div>
        <div className="rail-metrics">
          <Metric label="Turns" value={turns.length} />
          <Metric label="Tool calls" value={toolCount} />
          <Metric label="Logs" value={visibleLogs.length} />
        </div>
        {activeSessionId && (
          <button className="danger-action" onClick={() => deleteSession(activeSessionId)}>
            <Trash2 size={15} />
            Delete session
          </button>
        )}
      </aside>

      <main className="workspace">
        {tab === "chat" ? (
          <ChatView
            turns={turns}
            input={input}
            isSending={isSending}
            onInput={setInput}
            onSubmit={submit}
          />
        ) : (
          <LogsView
            logs={visibleLogs}
            totalLogCount={logs.length}
            activeSessionId={activeSessionId}
            visibleLogLevels={visibleLogLevels}
            onToggleLevel={toggleLogLevel}
          />
        )}
      </main>

      <aside className="event-rail">
        <SectionTitle icon={<PanelRight size={15} />} title="Loop events" />
        <EventTimeline turns={turns} logs={visibleLogs} />
      </aside>
    </div>
  );
}

function ChatView({
  turns,
  input,
  isSending,
  onInput,
  onSubmit,
}: {
  turns: Turn[];
  input: string;
  isSending: boolean;
  onInput: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <section className="chat-panel">
      <div className="transcript">
        {turns.length === 0 ? (
          <div className="empty-state">
            <TerminalSquare size={34} />
            <h2>Start a loop</h2>
            <p>Ask Aeloon Core to inspect files, run commands, or fetch a page.</p>
          </div>
        ) : (
          turns.map((turn) => <TurnGroup key={turn.id} turn={turn} />)
        )}
      </div>
      <form className="composer" onSubmit={onSubmit}>
        <textarea
          value={input}
          onChange={(event) => onInput(event.target.value)}
          placeholder="Message Aeloon Core"
          rows={2}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              event.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <button type="submit" disabled={!input.trim() || isSending}>
          {isSending ? <Loader2 className="spin" size={17} /> : <Send size={17} />}
          Send
        </button>
      </form>
    </section>
  );
}

function TurnGroup({ turn }: { turn: Turn }) {
  const toolBlocks = turn.blocks.filter((block) => block.type === "tool_call");
  const visibleBlocks = turn.blocks.filter((block) => block.type !== "tool_call");

  return (
    <article className="turn">
      <div className="message user-message">
        <div className="avatar user">
          <UserRound size={16} />
        </div>
        <p>{turn.user}</p>
      </div>
      <div className="message assistant-message">
        <div className="avatar assistant">
          <Bot size={16} />
        </div>
        <div className="assistant-stack">
          {visibleBlocks.map((block) =>
            block.type === "reasoning" ? (
              <ReasoningBlock key={block.id} block={block} toolBlocks={toolBlocks} />
            ) : (
              <TextBlock key={block.id} block={block} />
            ),
          )}
        </div>
      </div>
    </article>
  );
}

function TextBlock({ block }: { block: Block }) {
  const content = block.content || "";
  if (!content.trim()) {
    return null;
  }
  return (
    <div className="text-block">
      {content.split("\n").map((line, index) => (
        <p key={index}>{line || "\u00a0"}</p>
      ))}
    </div>
  );
}

function ReasoningBlock({ block, toolBlocks }: { block: Block; toolBlocks: Block[] }) {
  const entries = parseReasoningEntries(block.content || "");
  if (entries.length === 0) {
    return null;
  }
  const status = block.status ?? "running";
  const rows = attachToolBlocks(entries, toolBlocks);
  return (
    <details className={`reasoning-block ${status}`} open>
      <summary>
        <span className="reasoning-title">
          <BrainCircuit size={15} />
          Reasoning process
        </span>
        <span className="tool-status">
          {status === "running" ? (
            <Loader2 className="spin" size={15} />
          ) : (
            <CheckCircle2 size={15} />
          )}
          {status}
        </span>
      </summary>
      <div className="reasoning-entries">
        {rows.map(({ entry, toolBlock }, index) => (
          <ReasoningEntryRow
            key={`${entry.timestamp}-${entry.kind}-${index}`}
            entry={entry}
            toolBlock={toolBlock}
          />
        ))}
      </div>
    </details>
  );
}

function ReasoningEntryRow({
  entry,
  toolBlock,
}: {
  entry: ReasoningEntry;
  toolBlock?: Block;
}) {
  const argumentsValue = entry.arguments ?? toolBlock?.arguments;
  return (
    <div className={`reasoning-entry ${entry.kind}`}>
      <span>{entry.timestamp ? formatTime(entry.timestamp) : "model"}</span>
      <strong>{formatReasoningKind(entry.kind)}</strong>
      <div className="reasoning-entry-body">
        <p>{entry.text}</p>
        {entry.kind === "tool_call" && argumentsValue !== undefined && (
          <details className="reasoning-tool-detail">
            <summary>Arguments</summary>
            <pre>{formatValue(argumentsValue)}</pre>
          </details>
        )}
        {entry.kind === "tool_result" && (
          <ReasoningToolResult entry={entry} toolBlock={toolBlock} />
        )}
      </div>
    </div>
  );
}

function ReasoningToolResult({
  entry,
  toolBlock,
}: {
  entry: ReasoningEntry;
  toolBlock?: Block;
}) {
  const status = toolBlock?.status ?? entry.status ?? "running";
  const result = String(toolBlock?.result ?? entry.result ?? "");
  const resultLength = entry.resultLength ?? result.length;
  const isTruncated = result.length > TOOL_RESULT_PREVIEW_CHARS;
  const preview = isTruncated
    ? `${result.slice(0, TOOL_RESULT_PREVIEW_CHARS)}\n... truncated ${result.length - TOOL_RESULT_PREVIEW_CHARS} characters`
    : result;

  if (!result) {
    return <span className={`reasoning-result-empty ${status}`}>Waiting for full result.</span>;
  }

  return (
    <div className={`reasoning-tool-result ${status}`}>
      <div className="reasoning-result-header">
        <span>Result preview</span>
        <span>{formatCharCount(resultLength)} chars</span>
        <button
          className="reasoning-result-copy"
          type="button"
          onClick={() => void copyText(result)}
          title="Copy full tool result"
        >
          <Copy size={13} />
          Copy
        </button>
      </div>
      <pre className="reasoning-result-preview">{preview}</pre>
      {isTruncated && (
        <details className="reasoning-full-result">
          <summary>Full result</summary>
          <pre>{result}</pre>
        </details>
      )}
    </div>
  );
}

function LogsView({
  logs,
  totalLogCount,
  activeSessionId,
  visibleLogLevels,
  onToggleLevel,
}: {
  logs: LogEntry[];
  totalLogCount: number;
  activeSessionId: string;
  visibleLogLevels: LogLevelFilter[];
  onToggleLevel: (level: LogLevelFilter) => void;
}) {
  return (
    <section className="logs-panel">
      <div className="logs-header">
        <div>
          <h2>Logs</h2>
          <p>
            {activeSessionId
              ? `${logs.length} of ${totalLogCount} entries for current session`
              : "Start or select a session to view logs"}
          </p>
        </div>
        <div className="logs-header-actions">
          <div className="log-level-filter" aria-label="Log level filter">
            {LOG_LEVEL_FILTERS.map((level) => (
              <button
                key={level}
                className={visibleLogLevels.includes(level) ? "active" : ""}
                type="button"
                onClick={() => onToggleLevel(level)}
              >
                {level}
              </button>
            ))}
          </div>
          <button
            className="secondary-action"
            type="button"
            disabled={logs.length === 0}
            onClick={() => void copyText(formatLogsForCopy(logs))}
          >
            <Copy size={15} />
            Copy all
          </button>
        </div>
      </div>
      <div className="log-table">
        {logs.length === 0 ? (
          <p className="empty-note">No logs yet.</p>
        ) : (
          logs.map((log, index) => <LogRow key={`${log.ts}-${index}`} log={log} />)
        )}
      </div>
    </section>
  );
}

function EventTimeline({ turns, logs }: { turns: Turn[]; logs: LogEntry[] }) {
  const events = [
    ...turns.flatMap((turn) => [
      { id: `${turn.id}-start`, label: "turn started", kind: "turn", status: turn.status },
      ...turn.blocks
        .filter((block) => block.type === "tool_call" || block.type === "reasoning")
        .map((block) => ({
          id: block.id,
          label: block.type === "reasoning" ? "reasoning process" : block.name || "tool",
          kind: block.type === "reasoning" ? "reasoning" : "tool",
          status: block.status || "running",
        })),
    ]),
    ...logs.slice(0, 6).map((log, index) => ({
      id: `${log.ts}-${index}`,
      label: log.message,
      kind: log.level.toLowerCase(),
      status: log.level.toLowerCase(),
    })),
  ].slice(-18);

  if (events.length === 0) {
    return <p className="empty-note">Waiting for activity.</p>;
  }
  return (
    <div className="timeline">
      {events.map((event) => (
        <div key={event.id} className="timeline-row">
          <span className={`timeline-dot ${event.status}`} />
          <div>
            <strong>{event.kind}</strong>
            <p>{event.label}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function LogRow({ log }: { log: LogEntry }) {
  const Icon =
    log.level === "ERROR" ? XCircle : log.level === "WARNING" || log.level === "WARN" ? AlertTriangle : Clock3;
  const json = formatLogForCopy(log);
  return (
    <article className={`log-row ${log.level.toLowerCase()}`}>
      <div className="log-row-header">
        <Icon size={15} />
        <span>{formatTime(log.ts)}</span>
        <strong>{log.level}</strong>
        <code>{log.source}</code>
        <p>{log.message}</p>
        <button
          className="log-copy"
          type="button"
          onClick={() => void copyText(json)}
          title="Copy full log JSON"
        >
          <Copy size={14} />
          Copy JSON
        </button>
      </div>
      <pre className="log-json">{json}</pre>
    </article>
  );
}

function SectionTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="section-title">
      {icon}
      <span>{title}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function upsertBlock(blocks: Block[], block: Block) {
  if (blocks.some((item) => item.id === block.id)) {
    return blocks.map((item) => (item.id === block.id ? { ...item, ...block } : item));
  }
  return [...blocks, block];
}

function upsertBlockDelta(blocks: Block[], block: Block) {
  const existing = blocks.find((item) => item.id === block.id);
  if (!existing) {
    return [...blocks, block];
  }
  return blocks.map((item) =>
    item.id === block.id ? { ...item, content: `${item.content ?? ""}${block.content ?? ""}` } : item,
  );
}

function parseReasoningEntries(content: string): ReasoningEntry[] {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.map((line) => {
    const match = line.match(/^(?<timestamp>\S+) \[(?<kind>[^\]]+)] (?<text>[\s\S]+)$/);
    if (!match?.groups) {
      return { timestamp: "", kind: "reasoning", text: line };
    }
    const parsed = parseReasoningPayload(match.groups.text);
    if (parsed) {
      return {
        timestamp: match.groups.timestamp,
        kind: match.groups.kind,
        text: stringField(parsed.text) ?? stringField(parsed.summary) ?? match.groups.text,
        callId: stringField(parsed.call_id),
        toolName: stringField(parsed.tool_name),
        arguments: parsed.arguments,
        status: stringField(parsed.status),
        result: parsed.result === undefined ? undefined : stringifyUnknown(parsed.result),
        resultLength: numberField(parsed.result_length),
      };
    }
    return {
      timestamp: match.groups.timestamp,
      kind: match.groups.kind,
      text: match.groups.text,
      toolName: inferToolName(match.groups.kind, match.groups.text),
    };
  });
}

function parseReasoningPayload(raw: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return null;
  }
  return null;
}

function attachToolBlocks(entries: ReasoningEntry[], toolBlocks: Block[]) {
  const byId = new Map(toolBlocks.map((block) => [block.id, block]));
  const resultUses = new Map<string, number>();
  return entries.map((entry) => {
    let toolBlock = entry.callId ? byId.get(entry.callId) : undefined;
    if (!toolBlock && (entry.kind === "tool_call" || entry.kind === "tool_result")) {
      const key = entry.toolName ?? "*";
      const matches = toolBlocks.filter((block) => key === "*" || block.name === key);
      const index = entry.kind === "tool_result" ? resultUses.get(key) ?? 0 : 0;
      toolBlock = matches[index] ?? matches[matches.length - 1];
      if (entry.kind === "tool_result") {
        resultUses.set(key, index + 1);
      }
    }
    return { entry, toolBlock };
  });
}

function inferToolName(kind: string, text: string) {
  if (kind === "tool_call") {
    return text.match(/^Call\s+(?<name>\S+)/)?.groups?.name;
  }
  if (kind === "tool_result") {
    return text.match(/^(?<name>\S+)\s+returned\b/)?.groups?.name;
  }
  return undefined;
}

function formatReasoningKind(kind: string) {
  return kind.replace(/_/g, " ");
}

function formatValue(value: unknown) {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return stringifyUnknown(value);
  }
}

function stringifyUnknown(value: unknown) {
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function stringField(value: unknown) {
  return typeof value === "string" ? value : undefined;
}

function numberField(value: unknown) {
  return typeof value === "number" ? value : undefined;
}

function formatCharCount(value: number) {
  return value.toLocaleString();
}

function normalizeLogLevel(level: string): LogLevelFilter {
  const upper = level.toUpperCase();
  if (upper === "DEBUG" || upper === "INFO" || upper === "ERROR") {
    return upper;
  }
  return "WARN";
}

function logBucketKey(sessionId: string | undefined) {
  return sessionId || GLOBAL_LOG_KEY;
}

function extractLogSessionId(payload: Record<string, unknown>) {
  return stringField(payload.session_id) ?? extractSessionIdFromUnknown(payload.detail);
}

function extractSessionIdFromUnknown(value: unknown): string | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const direct = stringField(value.session_id);
  if (direct) {
    return direct;
  }
  return (
    extractSessionIdFromUnknown(value.params) ??
    extractSessionIdFromUnknown(value.result) ??
    extractSessionIdFromUnknown(value.request)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function logPayload(log: LogEntry) {
  return {
    ts: log.ts,
    level: log.level,
    source: log.source,
    session_id: log.sessionId ?? null,
    message: log.message,
    detail: log.detail ?? null,
  };
}

function formatLogForCopy(log: LogEntry) {
  return JSON.stringify(logPayload(log), null, 2);
}

function formatLogsForCopy(logs: LogEntry[]) {
  return JSON.stringify(logs.map(logPayload), null, 2);
}

async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function formatTime(raw: string) {
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) {
    return "--:--:--";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
