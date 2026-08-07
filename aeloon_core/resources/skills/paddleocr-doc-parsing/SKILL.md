---
name: paddleocr-doc-parsing
description: 使用本地 PaddleOCR PP-StructureV3 从扫描 PDF 和文档图片中提取中文及多语言文本、表格、公式、图表、印章、多栏布局和正确阅读顺序。当普通 PDF 或 Office 文本提取为空、明显缺失，或用户明确要求 OCR、版面分析、表格提取、公式识别、扫描件结构化时使用；不得调用云端 PaddleOCR API。
---

# 本地 PaddleOCR 文档解析

只运行本地 PP-StructureV3。禁止使用 `paddleocr api`、`PADDLEOCR_ACCESS_TOKEN` 或任何上传式 OCR 服务。

## 首次准备

PaddlePaddle、PaddleOCR 和 PP-StructureV3 文档解析依赖随 Aeloon 安装，不应另建 Python 环境或静默安装全局包。

首次运行会从 Paddle 支持的模型源下载权重，并缓存到 `~/.aeloon-core/models/paddleocr`。机密或完全断网环境应预先准备该缓存；没有缓存时不得偷偷切换云服务。

## 解析

```bash
aeloon system skill paddleocr-doc-parsing parse scan.pdf \
  --output-dir output/scan \
  --model-cache ~/.aeloon-core/models/paddleocr
```

缓存已准备且要求严格断网时加 `--offline`。默认启用方向分类、去畸变和文字行方向识别；平整且方向正确的数字扫描可使用对应的 `--no-*` 参数降低耗时。

## 核验

- 检查合并 Markdown、逐页 JSON 和图片资源是否齐全。
- 抽查页码、列顺序、表格单元格、公式 LaTeX、印章和图注。
- 对低质量或无法识别页面保留明确标记，不臆造文本。
- 普通数字版 DOCX/PPTX/XLSX 优先使用 `markitdown`；文本型 PDF 优先使用 `pdf`。

上游来源：PaddlePaddle/PaddleOCR `skills/paddleocr-doc-parsing`（Apache-2.0）；本适配将官方云 API 流程替换为本地 PP-StructureV3。

## 交付文件

完成并验证最终 Markdown、JSON 或用户要求的结构化结果后，只调用一次 `present_files`。不得声明模型缓存、逐页调试图或临时文件，除非它们本身是用户要求的交付物。
