# `aeloon-office-lite/v1` 写入规范

所有写入都使用 UTF-8 JSON。顶层 `schema` 固定为 `aeloon-office-lite/v1`；输出类型由 `.pdf`、`.docx`、`.pptx` 或 `.xlsx` 扩展名决定。未知字段会被忽略，缺少必需字段时命令失败。

## DOCX 与 PDF

```json
{
  "schema": "aeloon-office-lite/v1",
  "title": "月度简报",
  "author": "Aeloon",
  "blocks": [
    {"type": "heading", "text": "结论", "level": 1},
    {"type": "paragraph", "text": "本月进展符合预期。"},
    {"type": "bullets", "items": ["收入增长", "成本稳定"]},
    {
      "type": "table",
      "headers": ["指标", "数值"],
      "rows": [["收入", "120 万元"], ["增速", "12%"]]
    },
    {"type": "pagebreak"}
  ]
}
```

支持的 block：`heading`、`paragraph`、`bullets`、`table`、`pagebreak`。DOCX 还支持 `image`：`{"type":"image","path":"/绝对/本地图片.png","width_inches":5}`。PDF 忽略 `author` 之外的高级元数据，不支持图片。

## PPTX

```json
{
  "schema": "aeloon-office-lite/v1",
  "title": "季度汇报",
  "subtitle": "2026 Q2",
  "slides": [
    {"title": "核心结论", "bullets": ["增长稳定", "风险可控"], "notes": "口头补充"},
    {
      "title": "关键数据",
      "table": {"headers": ["指标", "Q1", "Q2"], "rows": [["收入", 100, 120]]}
    },
    {"title": "产品截图", "image": "/绝对/本地图片.png"}
  ]
}
```

每页只放一种主体：优先级为 `table`、`image`、`bullets`、`text`。保持每页要点简短，避免超过 8 条。输出为可编辑的 16:9 PPTX。

## XLSX

```json
{
  "schema": "aeloon-office-lite/v1",
  "sheets": [
    {
      "name": "数据",
      "rows": [["月份", "收入"], ["1月", 100], ["2月", 120]],
      "header": true,
      "freeze": "A2",
      "auto_filter": true
    }
  ]
}
```

`rows` 是二维数组，值仅使用字符串、数字、布尔值、ISO 日期字符串或 `null`。默认把首行作为表头设置简单样式；`header: false` 可关闭。此 skill 不计算公式，不执行宏。
