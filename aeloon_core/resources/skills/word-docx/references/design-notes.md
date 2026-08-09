# Design and attribution notes

- New DOCX creation uses `markdown-it-py` and `python-docx`.
- Existing DOCX editing changes only selected OOXML package parts with `lxml` and `zipfile`; it does not round-trip the source through `python-docx`.
- Tracked changes use native WordprocessingML revision elements. Comments use the standard comments part, document relationship, content-type override, range markers, and references.
- The public behavior description of `thvroyal/kimi-skills` informed the requirement to preserve formatting and support native revisions. That repository has no LICENSE file. No source code, internal structure, or text from it is copied or adapted here.
- The writing workflow consolidates the former bundled `office`, `document-writing`, and `reports` guidance: establish audience and facts first, keep calculations auditable, validate cross-file consistency, and deliver final artifacts only once.
