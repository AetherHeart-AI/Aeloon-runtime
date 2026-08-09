# Markdown and JSON specifications

## Markdown input

Separate slides with a line containing only `---`. The first heading on each slide is its title. The first paragraph on the first slide becomes the cover subtitle.

Supported editable content:

- paragraphs and ordered or unordered lists;
- pipe tables supported by `markdown-it-py`;
- local images with `![alt](relative/or/absolute/path.png)`;
- speaker notes in a fenced block whose info string is `notes`;
- a native chart in a fenced block whose info string is `chart`.

Example:

````markdown
# Quarterly review

Decisions and next steps

```notes
Open with the customer outcome, not the implementation history.
```

---

## Adoption accelerated

- Active teams grew 28%
- Retention improved in every segment

```chart
{"type":"column","categories":["Q1","Q2"],"series":[{"name":"Teams","values":[120,154]}]}
```
````

Chart `type` is one of `column`, `bar`, `line`, `pie`, or `doughnut`. A chart must have non-empty `categories` and `series`; every series must contain one numeric value per category.

## `ppt-template-detail/v1`

`inspect-template` emits slide dimensions plus an ordered `slides` array. Every shape has a stable address for that input deck:

```json
{
  "schema": "ppt-template-detail/v1",
  "source": "/absolute/template.pptx",
  "slide_size": {"width_inches": 13.333, "height_inches": 7.5},
  "slides": [{
    "address": "slide/1",
    "shapes": [{
      "address": "slide/1/shape/1",
      "name": "Title 1",
      "text": "Existing title",
      "paragraphs": [{
        "address": "slide/1/shape/1/paragraph/1",
        "runs": [{"address": "slide/1/shape/1/paragraph/1/run/1", "text": "Existing title"}]
      }]
    }]
  }]
}
```

Addresses are one-based and are valid only for the exact file inspected. Re-run inspection after structural edits.

## `ppt-edit-spec/v1`

Unknown schema major versions and unknown operations are rejected. Operations run in order.

```json
{
  "schema": "ppt-edit-spec/v1",
  "operations": [
    {
      "op": "replace_text",
      "address": "slide/1/shape/1",
      "old": "Existing title",
      "new": "New conclusion"
    },
    {
      "op": "set_text",
      "address": "slide/2/shape/3/paragraph/1/run/1",
      "text": "Exact replacement"
    },
    {
      "op": "replace_chart_data",
      "address": "slide/3/shape/2",
      "categories": ["Q1", "Q2"],
      "series": [{"name": "Revenue", "values": [10, 14]}]
    }
  ]
}
```

- `replace_text` requires one or more exact matches inside the addressed text shape. Replacement text inherits the first affected run's formatting; unaffected runs remain unchanged.
- `set_text` targets exactly one run and preserves its formatting.
- `replace_chart_data` targets an existing native chart and replaces only its cached/workbook data through `python-pptx`.
- Input and output paths must differ. Use `--overwrite` only to replace an existing output file.

## Validation output

`validate` emits `ppt-validation/v1`. Package/open errors are `errors`; heuristic overflow, missing titles, inconsistent title sizes, out-of-bounds shapes, and placeholder text are `warnings`. `--strict` returns a failing status for warnings. Rendering is still required for definitive visual QA.
