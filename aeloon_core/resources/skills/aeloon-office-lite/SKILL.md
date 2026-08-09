---
name: aeloon-office-lite
description: 快速、稳定地读取、创建、渲染和验证本地 PDF、PowerPoint PPTX、Word DOCX 与 Excel XLSX 文件。用于小模型处理简单 Office 内容、提取文本和表格、从精简 JSON 生成文件、把 PDF 或 Office 文件转成页图供视觉模型检查；优先使用捆绑的 Python 脚本，不处理复杂模板、宏、动画、修订、批注、公式计算或像素级排版。
---

# Aeloon Office Lite

优先追求稳定和速度。使用统一入口：

```bash
aeloon system skill aeloon-office-lite ACTION ...
```

## 工作流

1. 先运行 `preflight`。依赖齐全时直接继续，不要创建虚拟环境或安装额外工具。
2. 读取文件时运行 `read INPUT --output OUTPUT.md`，再读取 Markdown。普通 PDF、DOCX、PPTX、XLSX 不要先渲染。
3. 若 PDF 页面没有可提取文本，或用户要求检查版面，再运行 `render INPUT --output-dir DIR`，使用视觉能力读取页图。不要为此安装 OCR 模型。
4. 创建文件前读取 [references/spec.md](references/spec.md)，写一个精简 JSON，再运行 `write SPEC OUTPUT`。输出后必须运行 `validate OUTPUT`。
5. 只有用户要求检查外观时才运行 `render`；复杂版式不是此 skill 的目标。

## 常用命令

```bash
aeloon system skill aeloon-office-lite preflight
aeloon system skill aeloon-office-lite read input.pdf --output extracted.md
aeloon system skill aeloon-office-lite write content.json output.docx
aeloon system skill aeloon-office-lite validate output.docx
aeloon system skill aeloon-office-lite render output.pptx --output-dir rendered
```

支持读写 `.pdf`、`.docx`、`.pptx`、`.xlsx`，并支持只读 `.xlsm`。对于 `.doc`、`.ppt`、`.xls`、`.wps`、`.dps`、`.et`，要求用户先用 Office 或 WPS 另存为现代格式。只接受本地路径；不要覆盖输入文件或已有输出，除非用户明确允许并传入 `--overwrite`。

## 依赖规则

- 不要自动下载包。先向用户说明缺少的包和将执行的命令。
- 在中国网络环境下载 Python 包时，默认提示并使用清华源：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PACKAGE
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple PACKAGE
```

- `preflight` 和缺依赖错误会给出带清华源的准确命令。不要改用重量级 OCR、浏览器或 Node.js 方案。
- `render` 对 PDF 只需 Python；DOCX、PPTX、XLSX 转页图需要本机 LibreOffice。缺少时只报告安装提示，不要声称完成了视觉检查。

## 边界与交付

保持输入文件不变。不要承诺保留复杂模板、宏/VBA、修订、批注、动画、母版、嵌入对象、公式计算或精确分页。读取结果适合快速理解，不等价于取证级版面还原；扫描件以页图和视觉读取为准。

完成后仅用一次 `present_files` 交付最终文件。不要交付 JSON spec、提取的 Markdown、临时 PDF 或渲染页图，除非用户明确要求。
