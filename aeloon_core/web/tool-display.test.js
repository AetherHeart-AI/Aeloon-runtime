import { describe, expect, test } from "bun:test";

import { describeToolBlock } from "./tool-display.js";

describe("tool display model", () => {
  test("shows a useful target while a read call is running", () => {
    const display = describeToolBlock({
      type: "tool_call",
      name: "read",
      status: "running",
      arguments: { path: "aeloon_core/web/app.js" },
    });

    expect(display.label).toBe("读取文件");
    expect(display.headline).toBe("aeloon_core/web/app.js");
    expect(display.argumentsText).toContain('"path": "aeloon_core/web/app.js"');
  });

  test("does not let an empty result hide meaningful arguments", () => {
    const display = describeToolBlock({
      name: "grep",
      status: "done",
      arguments: { pattern: "renderProcessBlock", path: "aeloon_core/web" },
      result: "{}",
    });

    expect(display.resultText).toBe("");
    expect(display.headline).toContain("aeloon_core/web");
    expect(display.headline).toContain("renderProcessBlock");
    expect(display.argumentsText).not.toBe("");
  });

  test("summarizes structured worker reports instead of showing raw JSON", () => {
    const display = describeToolBlock({
      name: "run_workflow",
      status: "done",
      arguments: {
        code: 'result = builder(task="修复工具调用显示")',
      },
      result: JSON.stringify({
        status: "completed",
        summary: "工具调用显示已修复并通过测试",
        artifacts: [],
      }),
    });

    expect(display.argumentSummary).toBe("Builder · 修复工具调用显示");
    expect(display.resultSummary).toBe("completed · 工具调用显示已修复并通过测试");
    expect(display.resultText).toContain('"artifacts"');
  });

  test("always gives completed empty tools a readable fallback", () => {
    const display = describeToolBlock({
      name: "custom_tool",
      status: "done",
      arguments: {},
      result: null,
    });

    expect(display.label).toBe("Custom Tool");
    expect(display.headline).toBe("执行完成，工具未返回文本结果");
  });

  test("shows fixed workflow template and objective", () => {
    const display = describeToolBlock({
      name: "workflow_execute",
      status: "running",
      arguments: {
        template_id: "implement-review",
        inputs: { objective: "实现模板快速路径" },
      },
    });

    expect(display.label).toBe("运行固定工作流");
    expect(display.headline).toBe("implement-review · 实现模板快速路径");
  });

  test("redacts credentials without hiding harmless token limits", () => {
    const display = describeToolBlock({
      name: "execute",
      status: "running",
      arguments: {
        api_key: "secret-value",
        max_tokens: 4096,
        nested: { access_token: "secret-token" },
      },
    });

    expect(display.argumentsText).not.toContain("secret-value");
    expect(display.argumentsText).not.toContain("secret-token");
    expect(display.argumentsText).toContain('"max_tokens": 4096');
    expect(display.argumentsText).toContain("••••");
  });
});
