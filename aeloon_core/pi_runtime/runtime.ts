import {
  NodeExecutionEnv,
  createBashTool,
  createEditTool,
  createReadTool,
  createWriteTool,
  estimateContextTokens,
  runAgentLoop,
  type AgentContext,
  type AgentEvent,
  type AgentLoopConfig,
  type AgentMessage,
  type AgentTool,
} from "@earendil-works/pi-agent-core/node";
import {
  Type,
  createAssistantMessageEventStream,
  createModels,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type ToolCall,
  type Usage,
} from "@earendil-works/pi-ai";
import { deepseekProvider } from "@earendil-works/pi-ai/providers/deepseek";
import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import { createInterface } from "node:readline";

type JsonObject = Record<string, any>;

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
const pending = new Map<number, { resolve: (value: JsonObject) => void; reject: (error: Error) => void }>();
let nextRpcId = 1;
let startResolve: ((request: JsonObject) => void) | undefined;
const startPromise = new Promise<JsonObject>((resolve) => {
  startResolve = resolve;
});

input.on("line", (line) => {
  let message: JsonObject;
  try {
    message = JSON.parse(line);
  } catch {
    return;
  }
  if (message.type === "start") {
    startResolve?.(message.request);
    startResolve = undefined;
    return;
  }
  if (message.type !== "rpc_result" || typeof message.id !== "number") return;
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (message.error) waiter.reject(new Error(String(message.error)));
  else waiter.resolve(message.result ?? {});
});

function send(message: JsonObject): void {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function rpc(method: string, payload: JsonObject): Promise<JsonObject> {
  const id = nextRpcId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    send({ type: "rpc", id, method, payload });
  });
}

function emptyUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function errorStream(model: Model<any>, message: string): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  queueMicrotask(() => {
    const error: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: emptyUsage(),
      stopReason: "error",
      errorMessage: message,
      timestamp: Date.now(),
    };
    stream.push({ type: "start", partial: { ...error, stopReason: "pending" } });
    stream.push({ type: "error", reason: "error", error });
  });
  return stream;
}

function scriptedStream(
  model: Model<any>,
  response: JsonObject | undefined,
): AssistantMessageEventStream {
  if (!response) return errorStream(model, "Scripted Pi model has no response left");
  const stream = createAssistantMessageEventStream();
  queueMicrotask(() => {
    const content = normalizeScriptedContent(response);
    const stopReason = response.stopReason ?? (content.some((part) => part.type === "toolCall") ? "toolUse" : "stop");
    const message: AssistantMessage = {
      role: "assistant",
      content,
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: { ...emptyUsage(), ...(response.usage ?? {}) },
      stopReason,
      errorMessage: response.errorMessage,
      timestamp: Date.now(),
    };
    const partial: AssistantMessage = { ...message, content: [], stopReason: "pending" };
    stream.push({ type: "start", partial });
    for (const [contentIndex, part] of content.entries()) {
      if (part.type === "text") {
        stream.push({ type: "text_start", contentIndex, partial });
        stream.push({ type: "text_delta", contentIndex, delta: part.text, partial: message });
        stream.push({ type: "text_end", contentIndex, content: part.text, partial: message });
      } else if (part.type === "thinking") {
        stream.push({ type: "thinking_start", contentIndex, partial });
        stream.push({ type: "thinking_delta", contentIndex, delta: part.thinking, partial: message });
        stream.push({ type: "thinking_end", contentIndex, content: part.thinking, partial: message });
      } else {
        stream.push({ type: "toolcall_start", contentIndex, partial });
        stream.push({ type: "toolcall_end", contentIndex, toolCall: part, partial: message });
      }
    }
    if (stopReason === "error" || stopReason === "aborted") {
      stream.push({ type: "error", reason: stopReason, error: message });
    } else {
      stream.push({ type: "done", reason: stopReason, message });
    }
  });
  return stream;
}

function normalizeScriptedContent(response: JsonObject): AssistantMessage["content"] {
  if (Array.isArray(response.content)) return response.content;
  const content: AssistantMessage["content"] = [];
  if (typeof response.text === "string") content.push({ type: "text", text: response.text });
  for (const call of response.tool_calls ?? []) {
    content.push({
      type: "toolCall",
      id: String(call.id),
      name: String(call.name),
      arguments: call.arguments ?? {},
    });
  }
  return content;
}

function bindHarnessTool(tool: any, env: NodeExecutionEnv): AgentTool {
  return {
    ...tool,
    execute: (id: string, params: any, signal?: AbortSignal, onUpdate?: any) =>
      tool.execute(id, params, signal, onUpdate, { env }),
  };
}

function hostTool(definition: JsonObject): AgentTool {
  return {
    name: definition.name,
    label: definition.name,
    description: definition.description ?? "",
    parameters: Type.Unsafe(definition.input_schema ?? { type: "object" }),
    executionMode: definition.mode === "read_only" ? "parallel" : "sequential",
    async execute(toolCallId, params) {
      const response = await rpc("tool_call", {
        call_id: toolCallId,
        name: definition.name,
        arguments: params,
      });
      if (response.is_error) throw new Error(String(response.result ?? "tool failed"));
      return {
        content: [{ type: "text", text: String(response.result ?? "") }],
        details: response.details ?? {},
      };
    },
  };
}

function terminalTool(definition: JsonObject, state: RunState): AgentTool {
  return {
    name: definition.name,
    label: definition.name,
    description: definition.description,
    parameters: Type.Unsafe(definition.input_schema),
    executionMode: "sequential",
    async execute(_toolCallId, params) {
      state.finalOutput = { name: definition.name, arguments: params };
      return {
        content: [{ type: "text", text: "Structured output accepted." }],
        details: {},
        terminate: true,
      };
    },
  };
}

interface RunState {
  requestCount: number;
  totalUsage: Usage;
  finalOutput?: { name: string; arguments: JsonObject };
  limitReason?: string;
  failureReason?: string;
  missingOutputRetries: number;
  finalRequestActive: boolean;
}

function addUsage(target: Usage, usage: Usage): void {
  target.input += usage.input ?? 0;
  target.output += usage.output ?? 0;
  target.cacheRead += usage.cacheRead ?? 0;
  target.cacheWrite += usage.cacheWrite ?? 0;
  target.totalTokens += usage.totalTokens ?? 0;
  if (usage.reasoning !== undefined) target.reasoning = (target.reasoning ?? 0) + usage.reasoning;
  for (const key of ["input", "output", "cacheRead", "cacheWrite", "total"] as const) {
    target.cost[key] += usage.cost?.[key] ?? 0;
  }
}

function compactMessages(messages: Message[], capability: JsonObject | undefined): Message[] {
  if (!capability) return messages;
  if (estimateContextTokens(messages).tokens <= capability.max_tokens) return messages;
  const userIndexes = messages.flatMap((message, index) => message.role === "user" ? [index] : []);
  let cut = userIndexes.at(-1) ?? Math.max(0, messages.length - 1);
  for (const index of [...userIndexes].reverse()) {
    if (estimateContextTokens(messages.slice(index)).tokens > capability.keep_tokens) break;
    cut = index;
  }
  const tail = messages.slice(cut);
  if (!capability.preserve_first_user_message) return tail;
  const firstUser = messages.find((message) => message.role === "user");
  return firstUser && tail[0] !== firstUser ? [firstUser, ...tail] : tail;
}

function globPattern(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${escaped.replaceAll("**", "§§").replaceAll("*", "[^/]*").replaceAll("§§", ".*")}$`);
}

async function isProtectedPath(
  path: unknown,
  filesystem: JsonObject | undefined,
  write: boolean,
): Promise<boolean> {
  if (!filesystem || typeof path !== "string") return false;
  const root = resolve(String(filesystem.cwd));
  const candidate = isAbsolute(path) ? resolve(path) : resolve(root, path);
  const lexicalRelative = relative(root, candidate);
  if (lexicalRelative === ".." || lexicalRelative.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`) || isAbsolute(lexicalRelative)) return true;
  const canonicalRoot = await realpath(root).catch(() => root);
  let existing = candidate;
  while (true) {
    try {
      const canonicalBase = await realpath(existing);
      const canonicalCandidate = resolve(canonicalBase, relative(existing, candidate));
      const canonicalRelative = relative(canonicalRoot, canonicalCandidate);
      if (canonicalRelative === ".." || canonicalRelative.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`) || isAbsolute(canonicalRelative)) return true;
      break;
    } catch {
      const parent = dirname(existing);
      if (parent === existing) break;
      existing = parent;
    }
  }
  if (!write) return false;
  const normalizedRelative = lexicalRelative.replaceAll("\\", "/");
  return (filesystem.protected_patterns ?? []).some((pattern: string) => {
    if (pattern.endsWith("/*") && normalizedRelative.startsWith(pattern.slice(0, -1))) return true;
    const matched = pattern.includes("/")
      ? normalizedRelative
      : normalizedRelative.split("/").at(-1) ?? normalizedRelative;
    return globPattern(pattern).test(matched);
  });
}

function safeEnvironment(patterns: string[]): Record<string, string> {
  const denied = patterns.map(globPattern);
  return Object.fromEntries(
    Object.entries(process.env).flatMap(([name, value]) =>
      value !== undefined && !denied.some((pattern) => pattern.test(name)) ? [[name, value]] : [],
    ),
  );
}

async function assertExpectedHash(
  root: string,
  path: string,
  expectedHash: unknown,
  allowMissing: boolean,
): Promise<void> {
  if (typeof expectedHash !== "string") return;
  const absolutePath = isAbsolute(path) ? resolve(path) : resolve(root, path);
  let content: string;
  try {
    content = await readFile(absolutePath, "utf8");
  } catch (error: any) {
    if (allowMissing && error?.code === "ENOENT") return;
    throw error;
  }
  const actual = createHash("sha256").update(content).digest("hex").slice(0, 12);
  if (actual !== expectedHash) {
    throw new Error(
      `Conflict: file ${path} has changed (expected hash:${expectedHash}, got hash:${actual}). Re-read the file and retry.`,
    );
  }
}

function blockedShellReason(argumentsValue: JsonObject, shell: JsonObject | undefined): string | undefined {
  if (!shell || typeof argumentsValue.command !== "string") return undefined;
  const command = argumentsValue.command.trim();
  const executable = command.match(/^([^\s]+)/)?.[1];
  if (executable && (shell.denied_commands ?? []).includes(executable)) {
    return `Command ${executable} is denied.`;
  }
  if (shell.allow_interactive !== true) {
    const interactive = /^(vi|vim|nano|emacs|less|more|top|htop|man|sudo|passwd|ssh|telnet|ftp)\b/;
    if (interactive.test(command)) return `Interactive command ${executable ?? command} is not allowed.`;
  }
  return undefined;
}

async function main(request: JsonObject): Promise<JsonObject> {
  const state: RunState = {
    requestCount: 0,
    totalUsage: emptyUsage(),
    missingOutputRetries: 0,
    finalRequestActive: false,
  };
  const modelRequest = request.model;
  const models = createModels();
  models.setProvider(deepseekProvider());
  const providerModels = models.getModels(modelRequest.provider);
  let model = models.getModel(modelRequest.provider, modelRequest.model_id);
  if (!model && modelRequest.kind === "scripted") {
    model = {
      id: modelRequest.model_id,
      name: modelRequest.model_id,
      api: "openai-completions",
      provider: modelRequest.provider,
      baseUrl: "http://127.0.0.1/unused",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 1_000_000,
      maxTokens: 384_000,
    };
  } else if (!model && providerModels[0]) {
    model = { ...providerModels[0], id: modelRequest.model_id, name: modelRequest.model_id };
  }
  if (!model) throw new Error(`pi-ai provider/model is unavailable: ${modelRequest.provider}/${modelRequest.model_id}`);

  const capabilities: JsonObject[] = request.capabilities ?? [];
  const filesystem = capabilities.find((item) => item.kind === "filesystem");
  const shell = capabilities.find((item) => item.kind === "shell");
  const slidingWindow = capabilities.find((item) => item.kind === "sliding_window");
  const env = new NodeExecutionEnv({ cwd: filesystem?.cwd ?? shell?.cwd ?? request.workspace });
  const tools: AgentTool[] = (request.tools ?? []).map(hostTool);
  const filesystemTools = new Map<string, boolean>();
  if (filesystem) {
    const names = filesystem.tool_names ?? {};
    const read = bindHarnessTool(createReadTool(), env);
    const write = bindHarnessTool(createWriteTool(), env);
    const edit = bindHarnessTool(createEditTool(), env);
    read.name = String(names.read ?? "read");
    read.label = read.name;
    write.name = String(names.write ?? "write");
    write.label = write.name;
    edit.name = String(names.edit ?? "edit");
    edit.label = edit.name;
    read.parameters = Type.Object({
      path: Type.String({ description: "File path relative to the workspace." }),
      offset: Type.Optional(Type.Number({ minimum: 0, description: "Zero-based line offset." })),
      limit: Type.Optional(Type.Number({ minimum: 1, description: "Maximum lines to return." })),
    });
    const executeRead = read.execute;
    read.execute = (id, params, signal, onUpdate) => {
      const normalized = params as JsonObject;
      return executeRead(
        id,
        {
          path: normalized.path,
          offset: normalized.offset === undefined ? undefined : normalized.offset + 1,
          limit: normalized.limit,
        },
        signal,
        onUpdate,
      );
    };
    write.parameters = Type.Object({
      path: Type.String({ description: "File path relative to the workspace." }),
      content: Type.String({ description: "Complete UTF-8 file content." }),
      expected_hash: Type.Optional(Type.String({ description: "Previously observed content hash." })),
    });
    const executeWrite = write.execute;
    write.execute = async (id, params, signal, onUpdate) => {
      const normalized = params as JsonObject;
      await assertExpectedHash(filesystem.cwd, normalized.path, normalized.expected_hash, true);
      await realpath(dirname(resolve(filesystem.cwd, normalized.path))).catch(() => {
        throw new Error(`Parent directory for ${normalized.path} does not exist. Use create_directory first.`);
      });
      return executeWrite(
        id,
        { path: normalized.path, content: normalized.content },
        signal,
        onUpdate,
      );
    };
    edit.parameters = Type.Object({
      path: Type.String({ description: "File path relative to the workspace." }),
      old_text: Type.String({ description: "Exact text that must occur once." }),
      new_text: Type.String({ description: "Replacement text." }),
      expected_hash: Type.Optional(Type.String({ description: "Previously observed content hash." })),
    });
    const executeEdit = edit.execute;
    edit.execute = async (id, params, signal, onUpdate) => {
      const normalized = params as JsonObject;
      await assertExpectedHash(filesystem.cwd, normalized.path, normalized.expected_hash, false);
      return executeEdit(
        id,
        {
          path: normalized.path,
          edits: [{ oldText: normalized.old_text, newText: normalized.new_text }],
        },
        signal,
        onUpdate,
      );
    };
    filesystemTools.set(read.name, false);
    filesystemTools.set(write.name, true);
    filesystemTools.set(edit.name, true);
    tools.push(
      read,
      write,
      edit,
    );
  }
  if (shell) {
    const bash = createBashTool({
      prepare(execution) {
        execution.env = safeEnvironment(shell.denied_env_patterns ?? []);
        execution.inheritEnv = false;
      },
    });
    const boundBash = bindHarnessTool(bash, env);
    boundBash.name = String(shell.tool_name ?? "bash");
    boundBash.label = boundBash.name;
    boundBash.parameters = Type.Object({
      command: Type.String({ description: "The shell command to run." }),
      timeout_seconds: Type.Optional(
        Type.Number({ description: "Maximum seconds to wait." }),
      ),
    });
    const executeBash = boundBash.execute;
    boundBash.execute = (id, params, signal, onUpdate) => {
      const normalized = params as JsonObject;
      return executeBash(
        id,
        {
          command: normalized.command,
          timeout: normalized.timeout_seconds ?? shell.default_timeout,
        },
        signal,
        onUpdate,
      );
    };
    tools.push(boundBash);
  }
  const terminalNames = new Set<string>();
  for (const definition of request.terminals ?? []) {
    terminalNames.add(definition.name);
    tools.push(terminalTool(definition, state));
  }
  const toolSchemas = Object.fromEntries(tools.map((tool) => [tool.name, tool.parameters]));
  const toolCallsById = new Map<string, { name: string; args: JsonObject }>();
  const preflights = new Map<string, JsonObject>();
  const scriptedResponses: JsonObject[] = [...(modelRequest.responses ?? [])];

  const streamFn = (
    activeModel: Model<any>,
    context: Context,
    options?: SimpleStreamOptions,
  ): AssistantMessageEventStream => {
    if (request.request_limit !== null && state.requestCount >= request.request_limit) {
      state.limitReason ??= `The configured ${request.request_limit}-request limit was reached.`;
      return errorStream(activeModel, state.limitReason);
    }
    state.requestCount += 1;
    const isFinalRequest = request.request_limit !== null && state.requestCount === request.request_limit;
    state.finalRequestActive = isFinalRequest;
    let messages = compactMessages(context.messages, slidingWindow);
    let systemPrompt = context.systemPrompt;
    let requestTools = context.tools;
    if (isFinalRequest) {
      requestTools = context.tools?.filter((tool) => terminalNames.has(tool.name)) ?? [];
      const guidance = terminalNames.size
        ? "Return the configured structured output now. Do not call an ordinary tool."
        : "Return the best available final response now. Do not call a tool.";
      systemPrompt = `${systemPrompt ?? ""}\n\nHOST BUDGET NOTICE: This is the final model request. ${guidance}`;
    }
    let maxTokens = request.max_output_tokens ?? activeModel.maxTokens;
    if (request.max_tokens !== null) {
      const estimatedInput = estimateContextTokens(messages).tokens;
      const remaining = request.max_tokens - state.totalUsage.totalTokens - estimatedInput;
      if (remaining < 1) {
        state.limitReason ??= `The next model request would exceed the ${request.max_tokens}-token limit.`;
        return errorStream(activeModel, state.limitReason);
      }
      maxTokens = Math.min(maxTokens, remaining);
    }
    send({
      type: "event",
      event: "model_request",
      request_number: state.requestCount,
      messages,
      tool_names: requestTools?.map((tool) => tool.name) ?? [],
      max_tokens: maxTokens,
    });
    if (modelRequest.kind === "scripted") return scriptedStream(activeModel, scriptedResponses.shift());
    const streamOptions: SimpleStreamOptions = {
      ...options,
      apiKey: modelRequest.api_key,
      temperature: request.settings.temperature,
      reasoning: request.settings.reasoning,
      timeoutMs: request.settings.timeout_ms,
      maxRetries: request.max_retries,
      headers: request.settings.headers,
      maxTokens,
    };
    if (modelRequest.proxy) {
      streamOptions.fetch = ((input: any, init?: any) =>
        fetch(input, { ...(init ?? {}), proxy: modelRequest.proxy } as any)) as typeof fetch;
    }
    return models.streamSimple(activeModel, { ...context, systemPrompt, messages, tools: requestTools }, streamOptions);
  };

  const config: AgentLoopConfig = {
    model,
    convertToLlm: (messages: AgentMessage[]) => messages as Message[],
    toolExecution: "parallel",
    beforeToolCall: async ({ assistantMessage, toolCall, context }) => {
      const calls = assistantMessage.content.filter((part): part is ToolCall => part.type === "toolCall");
      const key = calls.map((call) => call.id).join("|");
      let preflight = preflights.get(key);
      if (!preflight) {
        const disabledOnFinalRequest = state.finalRequestActive
          ? calls.find((call) => !terminalNames.has(call.name))
          : undefined;
        let pathBlock: ToolCall | undefined;
        for (const call of calls) {
          const write = filesystemTools.get(call.name);
          if (write !== undefined && await isProtectedPath(call.arguments?.path, filesystem, write)) {
            pathBlock = call;
            break;
          }
        }
        const shellCall = calls.find((call) => call.name === String(shell?.tool_name ?? "bash"));
        const shellReason = shellCall ? blockedShellReason(shellCall.arguments ?? {}, shell) : undefined;
        preflight = disabledOnFinalRequest
          ? {
              allowed: false,
              reason: `Ordinary tool ${disabledOnFinalRequest.name} is disabled on the final request.`,
              limit_reason: `The configured ${request.request_limit}-request limit was reached before completion.`,
              terminate: true,
            }
          : pathBlock
          ? { allowed: false, reason: `Protected workspace path: ${pathBlock.arguments?.path}` }
          : shellReason
          ? { allowed: false, reason: shellReason }
          : await rpc("preflight", {
              calls,
              messages: context.messages as Message[],
              schemas: toolSchemas,
              usage: state.totalUsage,
            });
        preflights.set(key, preflight);
      }
      if (preflight.limit_reason) state.limitReason = preflight.limit_reason;
      if (preflight.failure_reason) state.failureReason = preflight.failure_reason;
      return preflight.allowed ? undefined : { block: true, reason: String(preflight.reason ?? "Tool batch rejected") };
    },
    afterToolCall: async ({ toolCall }) => {
      const callsKey = toolCall.id;
      const matching = [...preflights.entries()].find(([key]) => key.split("|").includes(callsKey))?.[1];
      return { terminate: Boolean(state.finalOutput || matching?.terminate) };
    },
    shouldStopAfterTurn: ({ message }) => {
      if (message.stopReason === "error" || message.stopReason === "aborted") {
        if (!state.limitReason) state.failureReason ??= message.errorMessage ?? "Pi provider request failed";
        return true;
      }
      return Boolean(state.finalOutput || state.limitReason || state.failureReason);
    },
    getFollowUpMessages: async () => {
      if (!terminalNames.size || state.finalOutput || state.limitReason || state.failureReason) return [];
      if (request.request_limit !== null && state.requestCount >= request.request_limit) {
        state.limitReason = `The configured ${request.request_limit}-request limit was reached before structured output.`;
        return [];
      }
      state.missingOutputRetries += 1;
      if (state.missingOutputRetries > request.max_retries) {
        state.failureReason = "The model returned plain text instead of the required structured output.";
        return [];
      }
      return [{
        role: "user",
        content: "Return the required structured output by calling its terminal tool. Do not answer in plain text.",
        timestamp: Date.now(),
      }];
    },
  };

  const emitted = async (event: AgentEvent): Promise<void> => {
    if (event.type === "message_update") {
      const update = event.assistantMessageEvent;
      if (update.type === "text_delta" || update.type === "thinking_delta") {
        send({ type: "event", event: update.type, delta: update.delta });
      }
      return;
    }
    if (event.type === "message_end" && event.message.role === "assistant") {
      addUsage(state.totalUsage, event.message.usage);
      send({ type: "event", event: "model_response", message: event.message });
      return;
    }
    if (event.type === "tool_execution_start") {
      toolCallsById.set(event.toolCallId, { name: event.toolName, args: event.args ?? {} });
      send({
        type: "event",
        event: "tool_start",
        call_id: event.toolCallId,
        name: event.toolName,
        arguments: event.args ?? {},
      });
      return;
    }
    if (event.type === "tool_execution_end") {
      const call = toolCallsById.get(event.toolCallId);
      const text = (event.result?.content ?? [])
        .filter((part: any) => part.type === "text")
        .map((part: any) => part.text)
        .join("\n");
      send({
        type: "event",
        event: "tool_end",
        call_id: event.toolCallId,
        name: event.toolName,
        arguments: call?.args ?? {},
        result: text,
        is_error: event.isError,
      });
    }
  };

  try {
    const prompt = { role: "user", content: request.prompt, timestamp: Date.now() } as const;
    const history = request.history as AgentMessage[];
    const context: AgentContext = {
      systemPrompt: request.instructions,
      messages: history,
      tools,
    };
    const newMessages = await runAgentLoop([prompt], context, config, emitted, undefined, streamFn);
    const messages = [...history, ...newMessages];
    const lastAssistant = [...newMessages].reverse().find((message) => message.role === "assistant") as AssistantMessage | undefined;
    const text = lastAssistant?.content
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("") ?? "";
    const status = state.finalOutput || (!terminalNames.size && lastAssistant?.stopReason === "stop")
      ? "completed"
      : state.limitReason
        ? "limit_exceeded"
        : "failed";
    return {
      status,
      output: state.finalOutput?.arguments ?? (status === "completed" ? text : null),
      output_name: state.finalOutput?.name,
      messages,
      usage: { ...state.totalUsage, requests: state.requestCount },
      failure: state.limitReason ?? state.failureReason ?? lastAssistant?.errorMessage,
    };
  } finally {
    await env.cleanup();
  }
}

try {
  const request = await startPromise;
  const result = await main(request);
  send({ type: "result", result });
  input.close();
} catch (error) {
  send({ type: "fatal", error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
  input.close();
  process.exitCode = 1;
}
