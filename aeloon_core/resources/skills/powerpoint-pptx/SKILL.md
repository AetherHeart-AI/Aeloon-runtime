---
name: powerpoint-pptx
description: Create, inspect, edit, render, and validate editable PowerPoint (.pptx) presentations with a pure-Python workflow. Use when Codex must build a deck from Markdown, preserve an existing PPTX template while replacing selected text or chart data, add speaker notes, or check a presentation for overflow, inconsistent titles, and placeholder content.
---

# PowerPoint PPTX

Use the bundled Python entry point for deterministic PPTX work:

```bash
aeloon system skill powerpoint-pptx build deck.md deck.pptx
aeloon system skill powerpoint-pptx inspect-template template.pptx --output detail.json
aeloon system skill powerpoint-pptx apply-template template.pptx edits.json edited.pptx
aeloon system skill powerpoint-pptx validate edited.pptx --output validation.json
aeloon system skill powerpoint-pptx render edited.pptx --output-dir rendered
```

Read [specs.md](references/specs.md) before writing Markdown, consuming template addresses, or creating an edit spec.

## Plan the story

1. Confirm the audience, setting, duration, desired action, and one-sentence conclusion.
2. Build the narrative before laying out slides. Use conclusion-led titles and one main idea per slide.
3. Use visuals, editable charts, tables, comparisons, or callouts to show relationships. Put explanation that belongs in the talk into speaker notes.
4. Keep names, dates, units, metrics, time ranges, and sources consistent across the deck.

## Build or edit

- Use `build` for an original, editable 16:9 deck. The bundled theme is code-generated and contains no third-party template assets.
- Use `inspect-template` before editing a supplied template. Address only shapes/runs exposed in `ppt-template-detail/v1`.
- Use `apply-template` with `ppt-edit-spec/v1`. Restrict changes to the selected text or native chart data; never overwrite the input template.
- Prefer concise text. Do not use a full-slide screenshot to imitate an editable slide.

## Verify and deliver

Run `validate`, then `render` when LibreOffice is available. Inspect every rendered slide for clipping, overlap, bad font substitution, low contrast, and excessive density. Fix issues and re-run validation. If LibreOffice is unavailable, report that visual QA was not completed and include the install hint emitted by the CLI.

Deliver only the final `.pptx` and explicitly requested exports through one `present_files` call. Do not deliver scripts, template copies, render PNGs, or intermediate JSON unless requested.

The implementation is an original Python workflow. `LICENSE.txt` preserves the complete GordenPPTSkill source-code license and records the inspiration/modification boundary; no restricted templates are bundled.
