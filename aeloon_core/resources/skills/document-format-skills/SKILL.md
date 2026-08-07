---
name: document-format-skills
description: 创建、诊断、清理和排版中文 DOCX 文档，支持把纯文本或 Markdown 转为 Word，修复中英文标点与空格，应用公文、学术或法律排版预设，整理表格与页码并保留可编辑内容。当用户需要真实 DOCX、中文 Word/WPS 通用文档、方案、公文、报告或排版修复时使用；不处理 WPS 原生 .wps/.dps/.et。
---

# 中文 DOCX 生成与排版

先使用 `document-writing` 完成内容结构和文字审阅，再用本技能生成或整理真实 `.docx`。WPS 兼容性指 WPS 打开的 OOXML `.docx`，不处理原生 `.wps`。

## 环境检查

本技能只需要可选的 `python-docx`，不打入 Aeloon 基础二进制：

```bash
uv run --with python-docx python scripts/process.py --help
```

依赖缺失时明确报告，不静默安装全局包。

## 创建 DOCX

从 Markdown 或纯文本创建并应用中文排版：

```bash
uv run --with python-docx python scripts/from_text.py \
  input.md output.docx --title "文档标题"
```

## 诊断与整理

```bash
uv run --with python-docx python scripts/process.py analyze input.docx --json
uv run --with python-docx python scripts/process.py smart input.docx output.docx --preset official
```

可选预设包括 `official`、`academic` 和 `legal`。只修复标点时使用 `punctuation` 子命令，只调整排版时使用 `format` 子命令。对现有文档先复制到新输出路径，不覆盖原件。

## 核验

- 重新打开输出 DOCX，核对标题层级、段落数、表格数、页眉页脚、页码和图片。
- 在可用时用 LibreOffice、Word 或 WPS 渲染预览，检查中文字体替换、分页和表格溢出。
- 保留事实内容，不因为排版清理重写用户文字；标点自动修复后抽查 URL、邮箱、时间和标准编号。

上游来源：KaguraNanaga/document-format-skills（MIT）。所附脚本保留上游实现和许可。

## 交付文件

完成并验证真实 DOCX 后，只调用一次 `present_files`，一次性声明最终 `.docx` 和用户明确要求的其他交付物。不得声明转换脚本、缓存或临时预览。
