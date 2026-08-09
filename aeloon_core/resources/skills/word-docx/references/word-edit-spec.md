# Word edit specs

## `word-edit-spec/v1`

```json
{
  "schema": "word-edit-spec/v1",
  "operations": [
    {
      "op": "replace",
      "find": "旧文本",
      "replace": "新文本",
      "scope": "all",
      "all": true,
      "case_sensitive": true
    },
    {
      "op": "fill",
      "placeholder": "[待确认：审批人]",
      "value": "张三",
      "scope": "document"
    }
  ]
}
```

`scope` 可为 `document`、`headers`、`footers` 或 `all`。`all` 默认为 `true`；设为 `false` 时只修改第一个命中。`fill` 是要求 placeholder 精确存在的替换。

## `word-edit-spec/v1.1`

v1.1 兼容全部 v1 操作，并增加：

```json
{
  "schema": "word-edit-spec/v1.1",
  "operations": [
    {
      "op": "track_replace",
      "find": "旧条款",
      "replace": "新条款",
      "author": "审阅人",
      "date": "2026-08-08T08:00:00Z",
      "all": false,
      "scope": "document"
    },
    {
      "op": "comment",
      "find": "需要确认的内容",
      "text": "请核对来源。",
      "author": "审阅人",
      "initials": "ZR",
      "date": "2026-08-08T08:00:00Z",
      "all": false
    }
  ]
}
```

- `track_replace` 写入 Word 原生 `w:del`/`w:ins`，不是颜色或删除线模拟。
- `comment` 当前只允许 `document` 范围，并同步 comments part、relationship 和 content type。
- `author` 默认 `Aeloon`，`initials` 默认 `AE`，`date` 默认当前 UTC 时间。
- 找不到目标文本属于错误，任何操作失败时不写输出文件。
- 输入和输出路径必须不同；除非明确传入 `--overwrite`，输出也不得预先存在。
