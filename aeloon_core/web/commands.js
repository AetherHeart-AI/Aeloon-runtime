export const COMMANDS = [
  ["/help", "显示命令和快捷键"],
  ["/agents", "聚焦当前 turn 的 agents"],
  ["/logs", "打开运行日志"],
  ["/new", "新建会话"],
  ["/sessions", "打开会话列表"],
  ["/resume <session>", "恢复会话"],
  ["/cancel-turn", "取消当前 turn"],
  ["/master", "返回主对话"],
  ["/clear", "清空当前可见对话"],
  ["/quit", "停止本地 Web UI"],
];

export function parseCommand(value) {
  const clean = String(value || "").trim();
  if (!clean.startsWith("/")) return null;
  const [name, ...args] = clean.slice(1).split(/\s+/);
  return { name: String(name || "").toLowerCase(), args };
}

export function commandSuggestions(value) {
  const clean = String(value || "").trimStart();
  if (!clean.startsWith("/") || clean.includes(" ") || clean.includes("\n")) return [];
  return COMMANDS.filter(([usage, description]) =>
    usage.toLowerCase().startsWith(clean.toLowerCase()) ||
    description.includes(clean.slice(1)),
  ).slice(0, 6);
}
