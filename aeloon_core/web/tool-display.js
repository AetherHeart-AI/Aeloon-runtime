const TOOL_LABELS = {
  edit_file: "编辑文件",
  execute: "执行命令",
  glob: "查找文件",
  grep: "搜索内容",
  list: "列出目录",
  list_files: "列出目录",
  read: "读取文件",
  read_file: "读取文件",
  run_workflow: "运行 Worker 工作流",
  shell: "执行命令",
  write_file: "写入文件",
  write_plan: "更新计划",
  workflow_describe: "查看工作流模板",
  workflow_execute: "运行固定工作流",
  workflow_search: "搜索工作流模板",
};

const EMPTY_RESULT_TEXT = new Set(["", "{}", "[]", "null", "none", "undefined"]);
const SENSITIVE_KEY =
  /(^|_)(api_?key|authorization|credential|password|secret|access_?token|refresh_?token)($|_)/i;

export function describeToolBlock(block = {}) {
  const name = String(block.name || "tool");
  const status = normalizeStatus(block.status);
  const argumentsValue = sanitizeValue(asRecord(block.arguments));
  const argumentsText = hasKeys(argumentsValue)
    ? JSON.stringify(argumentsValue, null, 2)
    : "";
  const argumentSummary = summarizeArguments(name, argumentsValue);
  const result = normalizeResult(block.result);
  const resultSummary = summarizeResult(result);
  const fallback = {
    running: "正在执行…",
    error: "执行失败，未返回错误详情",
    cancelled: "执行已取消",
    done: "执行完成，工具未返回文本结果",
  }[status];

  return {
    name,
    label: TOOL_LABELS[name] || humanizeName(name),
    status,
    icon: { running: "·", done: "✓", error: "×", cancelled: "–" }[status],
    headline:
      status === "running"
        ? argumentSummary || fallback
        : resultSummary || argumentSummary || fallback,
    argumentSummary,
    argumentsText,
    resultSummary,
    resultText: result?.text || "",
    hasDetails: Boolean(argumentsText || result?.text),
  };
}

export function summarizeArguments(name, value = {}) {
  if (!hasKeys(value)) return "";

  const path = firstText(value, ["path", "file_path", "directory", "cwd"]);
  const query = firstText(value, ["query", "pattern", "search"]);
  const task = firstText(value, ["task", "objective", "prompt"]);
  const command = firstText(value, ["command", "cmd"]);

  if (name === "run_workflow") {
    const workflowTask = extractWorkflowTask(String(value.code || ""));
    if (workflowTask) return workflowTask;
    return task ? truncate(task) : "编排并运行临时 Worker";
  }
  if (name === "workflow_execute") {
    const template = firstText(value, ["template_id"]);
    const inputs = asRecord(value.inputs);
    const objective = firstText(inputs, ["task", "objective", "prompt"]);
    return [template, objective ? truncate(objective) : ""].filter(Boolean).join(" · ");
  }
  if (name === "workflow_describe") {
    return firstText(value, ["template_id"]);
  }
  if (task) return truncate(task);
  if (command) return truncate(command.split("\n", 1)[0]);
  if (path && query) return `${truncate(path, 80)} · “${truncate(query, 80)}”`;
  if (path) return truncate(path);
  if (query) return `“${truncate(query)}”`;

  const firstEntry = Object.entries(value).find(([, item]) => isScalar(item));
  if (!firstEntry) return "";
  return `${humanizeName(firstEntry[0])}: ${truncate(String(firstEntry[1]))}`;
}

export function summarizeResult(result) {
  if (!result) return "";
  const value = result.parsed;
  if (typeof value === "string") return truncate(firstMeaningfulLine(value));
  if (Array.isArray(value)) {
    return value.length ? `返回 ${value.length} 项` : "";
  }
  if (!value || typeof value !== "object") return truncate(String(value));

  const nestedResult = value.result;
  const candidate =
    firstText(value, ["error", "message", "summary", "output", "detail"]) ||
    (nestedResult && typeof nestedResult === "object"
      ? firstText(nestedResult, ["error", "message", "summary", "output", "detail"])
      : "");
  const status = firstText(value, ["status"]);
  if (candidate && status) return `${status} · ${truncate(firstMeaningfulLine(candidate))}`;
  if (candidate) return truncate(firstMeaningfulLine(candidate));
  if (status) return status;
  return `返回 ${Object.keys(value).length} 个字段`;
}

function normalizeResult(value) {
  if (value === null || value === undefined) return null;
  const text =
    typeof value === "string" ? value.trim() : JSON.stringify(value, null, 2).trim();
  if (EMPTY_RESULT_TEXT.has(text.toLowerCase())) return null;
  let parsed = value;
  if (typeof value === "string") {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  if (
    parsed === null ||
    (Array.isArray(parsed) && parsed.length === 0) ||
    (parsed && typeof parsed === "object" && Object.keys(parsed).length === 0)
  ) {
    return null;
  }
  return { parsed, text };
}

function extractWorkflowTask(code) {
  const match = code.match(
    /\b([A-Za-z_]\w*)\s*\(\s*task\s*=\s*(["'])([\s\S]*?)\2\s*\)/,
  );
  if (!match) return "";
  return `${humanizeName(match[1])} · ${truncate(match[3])}`;
}

function normalizeStatus(status) {
  const value = String(status || "running").toLowerCase();
  if (["done", "completed", "success"].includes(value)) return "done";
  if (["error", "failed", "failure"].includes(value)) return "error";
  if (value === "cancelled") return "cancelled";
  return "running";
}

function sanitizeValue(value, key = "") {
  if (SENSITIVE_KEY.test(key)) return "••••";
  if (Array.isArray(value)) return value.map((item) => sanitizeValue(item));
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value).map(([childKey, childValue]) => [
      childKey,
      sanitizeValue(childValue, childKey),
    ]),
  );
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function firstText(value, keys) {
  for (const key of keys) {
    const candidate = value?.[key];
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
  }
  return "";
}

function firstMeaningfulLine(value) {
  return String(value)
    .split("\n")
    .map((line) => line.trim())
    .find(Boolean) || "";
}

function hasKeys(value) {
  return Boolean(value && typeof value === "object" && Object.keys(value).length);
}

function isScalar(value) {
  return ["string", "number", "boolean"].includes(typeof value);
}

function humanizeName(value) {
  return String(value || "tool")
    .replace(/^worker_/, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function truncate(value, limit = 160) {
  const normalized = String(value || "").replace(/\s+/g, " ").trim();
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}…` : normalized;
}
