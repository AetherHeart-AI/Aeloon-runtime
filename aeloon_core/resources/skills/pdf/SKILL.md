---
name: pdf
description: 在本地读取、创建和审阅 PDF，特别适用于需要同时核对文本与页面版面的任务。使用内置 pdfplumber 或 pypdf 提取文本，使用内置 pypdfium2 渲染页面进行视觉检查；扫描件、中文 OCR、复杂表格和公式改用 paddleocr-doc-parsing。
---

# PDF 读取与检查

先检查内置 PDF 渲染器是否可用：

```bash
aeloon system skill pdf render --check
```

## 读取

1. 使用 `pdfplumber` 提取正文和表格，或使用 `pypdf` 做快速文本、元数据和页面检查。
2. 页面布局会影响理解时，运行：

```bash
aeloon system skill pdf render input.pdf --output-dir tmp/pdf-pages
```

3. 检查渲染图中的分栏、图表、脚注、页眉页脚和文字顺序。文本为空或与页面可见内容明显不符时，改用 `paddleocr-doc-parsing`。

`pdfplumber`、`pypdf`、`pypdfium2` 和 `reportlab` 随 Aeloon 安装，不应在用户任务中另建环境或静默安装全局包。

## 创建与修改

使用 `reportlab` 创建排版可控的 PDF，使用 `pypdf` 合并、拆分或处理元数据。每次重要修改后重新渲染，检查裁切、重叠、分页、字体缺失和乱码。

不得仅凭文本提取结果声称版面正确，也不得把本地 PDF 上传到外部 OCR 服务。

上游来源：OpenAI `skills/.curated/pdf`（Apache-2.0）。

## 交付文件

完成并验证真实 PDF 后，只调用一次 `present_files`，一次性声明全部最终交付物。不得声明页面 PNG、缓存或生成脚本。
