---
name: pptx-generator
description: 生成、编辑、读取和验证真实可编辑的 PowerPoint PPTX 文件。当用户要求制作或修改幻灯片、演示文稿、路演材料、培训课件、演讲备注或需要把既有 PPTX 模板变成最终文件时使用；内容叙事先配合 ppt，文件读取可配合 markitdown。
---

# PPTX 生成与验证

先使用 `ppt` 确定受众、叙事和逐页结论，再用本技能生成真实可编辑的 `.pptx`。

## 准备

检查本地依赖，不要在用户任务中静默全局安装：

```bash
node --version
npm ls pptxgenjs --depth=0
python scripts/render_slides.py --check
```

缺少依赖时明确报告。生成需要 Node.js 与 PptxGenJS；渲染检查需要 LibreOffice 和 Poppler。

## 生成

1. 阅读 `references/design-system.md` 和 `references/slide-types.md`，确定配色、字体、页面类型和视觉层级。
2. 创建一个可重复运行的 JS 源文件，使用 PptxGenJS 生成 PPTX。中文优先使用目标机器已安装的微软雅黑、等线、思源黑体或用户模板字体。
3. 每页只表达一个主要观点；文本、表格、图表和基础图形保持可编辑。
4. 编辑现有模板时先阅读 `references/editing.md`；PptxGenJS 细节查阅 `references/pptxgenjs.md`。
5. 阅读 `references/pitfalls.md` 并运行渲染：

```bash
python scripts/render_slides.py output.pptx --output-dir tmp/pptx-render
```

逐页检查溢出、遮挡、乱码、字体替换、低对比度和过密文字，修改源文件后重新生成与渲染。不得用整页截图冒充可编辑幻灯片。

上游来源：MiniMax-AI `skills/pptx-generator`（MIT）。

## 交付文件

完成并验证真实 PPTX 后，只调用一次 `present_files`，一次性声明最终 `.pptx` 和用户明确要求的其他交付物。不得声明 JS 源码、页面 PNG、缓存或临时 PDF。
