---
name: pptx-generator
description: 生成、编辑、读取和验证真实可编辑的 PowerPoint PPTX 文件。当用户要求制作或修改幻灯片、演示文稿、路演材料、培训课件、演讲备注或需要把既有 PPTX 模板变成最终文件时使用；内容叙事先配合 ppt，文件读取可配合 markitdown。
---

# PPTX 生成与验证

先使用 `ppt` 确定受众、叙事和逐页结论，再用本技能生成真实可编辑的 `.pptx`。

## 准备

检查内置生成运行时和可选渲染工具：

```bash
aeloon system skill office preflight --require pptx
aeloon system skill pptx-generator render --check
```

Node.js、PptxGenJS 和 PDF 渲染器随 Aeloon 安装。PPTX 转图片检查还需要外部 LibreOffice；缺少时明确报告，但不得阻止生成可编辑 PPTX。

## 生成

1. 阅读 `references/design-system.md` 和 `references/slide-types.md`，确定配色、字体、页面类型和视觉层级。
2. 创建一个可重复运行的 JS 源文件，使用 PptxGenJS 生成 PPTX。通过 `aeloon system skill pptx-generator node deck.cjs` 执行，不直接调用系统 Node。中文优先使用目标机器已安装的微软雅黑、等线、思源黑体或用户模板字体。
3. 每页只表达一个主要观点；文本、表格、图表和基础图形保持可编辑。
4. 编辑现有模板时先阅读 `references/editing.md`；PptxGenJS 细节查阅 `references/pptxgenjs.md`。
5. 阅读 `references/pitfalls.md` 并运行渲染：

```bash
aeloon system skill pptx-generator render output.pptx --output-dir tmp/pptx-render
```

逐页检查溢出、遮挡、乱码、字体替换、低对比度和过密文字，修改源文件后重新生成与渲染。不得用整页截图冒充可编辑幻灯片。

上游来源：MiniMax-AI `skills/pptx-generator`（MIT）。

## 交付文件

完成并验证真实 PPTX 后，只调用一次 `present_files`，一次性声明最终 `.pptx` 和用户明确要求的其他交付物。不得声明 JS 源码、页面 PNG、缓存或临时 PDF。
