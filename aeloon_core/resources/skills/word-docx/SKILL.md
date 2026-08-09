---
name: word-docx
description: 创建、编辑、修订、批注、渲染并验证可编辑的 Word DOCX 文档。用于从 Markdown 制作专业 Word 文件、保留既有版式地填充或替换内容、添加 track changes 或 comments，以及检查 DOCX 结构与视觉质量；不处理原生 .doc/.wps。
---

# Word DOCX

使用 Python 创建新文档，使用定点 OOXML 修改保留现有版式。始终保留输入文件并写入新的 `.docx`。

## 先确定内容

1. 明确受众、目的、事实来源、统计口径、语气和交付格式。
2. 区分已确认事实、作者判断、合理假设和待确认信息；缺失内容写为 `[待确认：…]`，不得虚构。
3. 先完成标题层级和正文逻辑，再排版。长文、高风险材料或内容尚未确定时，先交付 Markdown 草稿供确认。
4. 核对姓名、日期、数字、单位、来源、指标公式、总计与明细，以及跨文件重复出现的术语和结论。

## 创建

Markdown 支持标题、段落、强调、链接、图片、列表、代码、表格和 `<!-- pagebreak -->`。可增加封面、目录字段、页眉、页脚和页码：

```bash
aeloon system skill word-docx build draft.md output.docx \
  --title "报告标题" --author "作者" --toc --page-numbers
```

只引用本地图片。生成后必须运行 `validate`；正式交付前尽可能运行 `render` 并查看全部页面。

## 编辑、修订与批注

把操作写入 JSON，再执行：

```bash
aeloon system skill word-docx edit input.docx output.docx --spec edits.json
```

普通替换使用 `word-edit-spec/v1`；修订和批注使用 `word-edit-spec/v1.1`。读取 [word-edit-spec.md](references/word-edit-spec.md) 获取字段、范围和示例。替换会处理被多个 run 分割的文本，并保留未触及内容和首个命中 run 的样式。

## 验证与渲染

```bash
aeloon system skill word-docx validate output.docx --report validation.json
aeloon system skill word-docx render output.docx --output-dir rendered
```

`validate` 检查 OPC 关系、XML、元素顺序、内容数量、修订/批注引用，并在 MarkItDown 可用时回读内容。`render` 使用 LibreOffice 转 PDF，再逐页输出 PNG。缺少 LibreOffice 时按命令返回的 macOS 或 Ubuntu 安装提示处理；此时只能声明结构验证完成，不能声明已完成视觉 QA。

## 质量与交付

- 检查分页、断行、表格溢出、字体替换、目录、页眉页脚和残留占位符。
- 保留用户模板、样式、编号和术语；不要用重建文档代替局部编辑。
- 完成并验证真实 DOCX 后，只调用一次 `present_files`，一次性声明最终 `.docx` 和用户明确要求的其他成品。不要声明脚本、缓存、模板副本、临时 PDF 或页面 PNG。
- 最终回复仅说明交付内容、验证范围、关键假设和待确认项；不要粘贴生成代码或用本地 Markdown 链接重复交付。

## 设计边界

实现使用 `markdown-it-py`、`python-docx`、`lxml` 和标准库。修订行为参考 Kimi skill 的公开 README 描述；该仓库未提供许可证，本 skill 未读取、复制或改编其代码或代码结构。详见 [design-notes.md](references/design-notes.md)。
