from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).parents[1] / "aeloon_runtime" / "resources" / "skills" / "aeloon-office-lite"
)


def load_cli():
    path = SKILL_DIR / "scripts" / "cli.py"
    spec = importlib.util.spec_from_file_location("test_aeloon_office_lite_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def office():
    return load_cli()


def test_metadata_and_reference_are_small_model_friendly() -> None:
    skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, body = skill_text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    assert metadata["name"] == "aeloon-office-lite"
    assert set(metadata) == {"name", "description"}
    for action in ("preflight", "read", "write", "render", "validate"):
        assert action in body
    assert "清华" in body
    assert "不要为此安装 OCR 模型" in body
    assert (SKILL_DIR / "references" / "spec.md").is_file()


def test_preflight_is_ready_and_contains_tsinghua_install_hints(office) -> None:
    result = office.preflight()
    assert result["ready"] is True
    assert set(result["packages"]) == {
        "pypdf",
        "pypdfium2",
        "python-docx",
        "python-pptx",
        "openpyxl",
        "reportlab",
    }
    hints = office.install_hints(["example-package"])
    assert office.TSINGHUA_INDEX in hints["pip"]
    assert office.TSINGHUA_INDEX in hints["uv"]
    assert "先告知用户" in hints["notice"]


def test_round_trip_read_write_validate_and_render_all_formats(office, tmp_path: Path) -> None:
    common = {
        "schema": office.SCHEMA,
        "title": "月度简报",
        "author": "Aeloon",
        "blocks": [
            {"type": "heading", "text": "核心结论", "level": 1},
            {"type": "paragraph", "text": "本月进展符合预期。"},
            {"type": "bullets", "items": ["收入增长", "成本稳定"]},
            {
                "type": "table",
                "headers": ["指标", "数值"],
                "rows": [["收入", "120 万元"], ["增速", "12%"]],
            },
        ],
    }
    pptx = {
        "schema": office.SCHEMA,
        "title": "季度汇报",
        "subtitle": "2026 Q2",
        "slides": [
            {"title": "核心结论", "bullets": ["增长稳定", "风险可控"], "notes": "内部备注"},
            {
                "title": "关键数据",
                "table": {"headers": ["指标", "Q1", "Q2"], "rows": [["收入", 100, 120]]},
            },
        ],
    }
    xlsx = {
        "schema": office.SCHEMA,
        "sheets": [
            {
                "name": "数据",
                "rows": [["月份", "收入"], ["1月", 100], ["2月", 120]],
                "freeze": "A2",
                "auto_filter": True,
            }
        ],
    }
    cases = (("pdf", common), ("docx", common), ("pptx", pptx), ("xlsx", xlsx))

    for suffix, spec in cases:
        output = tmp_path / f"sample.{suffix}"
        write_result = office.write_document(spec, output)
        assert output.is_file() and output.stat().st_size > 0
        assert write_result
        validation = office.validate_document(output)
        assert validation["valid"] is True
        markdown, metadata = office.read_document(output)
        assert markdown.strip()
        assert metadata["warnings"] == []

    assert "核心结论" in office.read_document(tmp_path / "sample.docx")[0]
    assert "季度汇报" in office.read_document(tmp_path / "sample.pptx")[0]
    assert "120" in office.read_document(tmp_path / "sample.xlsx")[0]
    pdf_markdown, _ = office.read_document(tmp_path / "sample.pdf")
    assert "月度简报" in pdf_markdown

    rendered = office.render_document(
        tmp_path / "sample.pdf", tmp_path / "rendered", dpi=72, overwrite=False
    )
    assert len(rendered) == 1
    assert rendered[0].read_bytes().startswith(b"\x89PNG")

    if office._libreoffice_command() is not None:
        for suffix in ("docx", "pptx", "xlsx"):
            office_pages = office.render_document(
                tmp_path / f"sample.{suffix}",
                tmp_path / f"rendered-{suffix}",
                dpi=72,
                overwrite=False,
            )
            assert office_pages
            assert all(page.read_bytes().startswith(b"\x89PNG") for page in office_pages)


def test_xlsx_read_limits_are_explicit(office, tmp_path: Path) -> None:
    output = tmp_path / "large.xlsx"
    office.write_document(
        {
            "schema": office.SCHEMA,
            "sheets": [{"name": "数据", "rows": [["n"], [1], [2], [3]]}],
        },
        output,
    )

    markdown, metadata = office.read_document(output, max_rows=2, max_columns=1)

    assert "| 1 |" in markdown
    assert "| 2 |" not in markdown
    assert metadata["warnings"] == [
        {
            "code": "sheet_truncated",
            "sheet": "数据",
            "source_rows": 4,
            "source_columns": 1,
        }
    ]


def test_remote_legacy_and_existing_outputs_are_rejected(office, tmp_path: Path) -> None:
    with pytest.raises(office.SkillError, match="本地"):
        office._input_path("https://example.com/a.pdf")
    legacy = tmp_path / "old.xls"
    legacy.write_bytes(b"old")
    with pytest.raises(office.SkillError, match="另存为"):
        office._input_path(str(legacy))
    existing = tmp_path / "existing.pdf"
    existing.write_bytes(b"existing")
    with pytest.raises(office.SkillError, match="--overwrite"):
        office._output_path(str(existing), overwrite=False)


def test_cli_writes_json_errors_and_tsinghua_hint(
    office, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"schema": office.SCHEMA, "title": "PDF", "blocks": []}),
        encoding="utf-8",
    )
    original = office.importlib.util.find_spec
    monkeypatch.setattr(
        office.importlib.util,
        "find_spec",
        lambda name: None if name == "reportlab" else original(name),
    )

    assert office.main(["write", str(spec), str(tmp_path / "out.pdf")]) == 3
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "missing_dependency"
    assert office.TSINGHUA_INDEX in error["install"]["pip"]
