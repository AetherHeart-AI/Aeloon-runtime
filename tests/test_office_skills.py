from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

from aeloon_core import __main__ as cli
from aeloon_core.runtime import skill_runtime
from aeloon_core.runtime.builtin_skills import BUILTIN_SKILL_IDS

RESOURCE_ROOT = Path(__file__).parents[1] / "aeloon_core" / "resources" / "skills"
OFFICE_SKILL_IDS = ("document-reader", "word-docx", "powerpoint-pptx")
RETIRED_SKILL_IDS = (
    "office",
    "ppt",
    "document-writing",
    "reports",
    "markitdown",
    "pdf",
    "paddleocr-doc-parsing",
    "pptx-generator",
    "document-format-skills",
)
RETIRED_REPLACEMENTS = {
    "office": "document-reader, word-docx, or powerpoint-pptx",
    "ppt": "powerpoint-pptx",
    "document-writing": "word-docx",
    "reports": "word-docx",
    "markitdown": "document-reader",
    "pdf": "document-reader",
    "paddleocr-doc-parsing": "document-reader",
    "pptx-generator": "powerpoint-pptx",
    "document-format-skills": "word-docx",
}


def test_only_new_office_skills_are_bundled_and_valid() -> None:
    assert BUILTIN_SKILL_IDS == OFFICE_SKILL_IDS
    assert {path.parent.name for path in RESOURCE_ROOT.glob("*/SKILL.md")} == set(OFFICE_SKILL_IDS)
    for skill_id in OFFICE_SKILL_IDS:
        skill_dir = RESOURCE_ROOT / skill_id
        skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        _, frontmatter, body = skill_text.split("---", 2)
        metadata = yaml.safe_load(frontmatter)
        assert set(metadata) == {"name", "description"}
        assert metadata["name"] == skill_id
        assert "present_files" in body
        assert (skill_dir / "LICENSE.txt").is_file()
        interface = yaml.safe_load(
            (skill_dir / "agents" / "openai.yaml").read_text(encoding="utf-8")
        )["interface"]
        assert f"${skill_id}" in interface["default_prompt"]


def test_old_skill_resources_are_removed() -> None:
    assert all(not (RESOURCE_ROOT / skill_id).exists() for skill_id in RETIRED_SKILL_IDS)


def test_bundled_skill_scripts_compile() -> None:
    for skill_id in OFFICE_SKILL_IDS:
        for script in (RESOURCE_ROOT / skill_id).rglob("*.py"):
            compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_bundled_skill_runtime_prepends_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "document-reader" / "scripts" / "cli.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import sys\n"
        "assert sys.argv[1:] == ['ingest', '--check']\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_runtime, "bundled_skill_root", lambda: tmp_path)
    original_argv = list(sys.argv)

    assert skill_runtime.run_bundled_skill("document-reader", "ingest", ["--check"]) == 7
    assert sys.argv == original_argv


def test_bundled_skill_runtime_reports_unknown_actions() -> None:
    with pytest.raises(ValueError, match="expected one of: preflight, prepare-ocr, ingest"):
        skill_runtime.run_bundled_skill("document-reader", "unknown", [])
    with pytest.raises(ValueError, match="unknown bundled Skill 'missing'"):
        skill_runtime.run_bundled_skill("missing", "run", [])


@pytest.mark.parametrize(("skill_id", "replacement"), RETIRED_REPLACEMENTS.items())
def test_every_retired_skill_reports_its_replacement(skill_id: str, replacement: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"has been retired; use {re.escape(replacement)}",
    ):
        skill_runtime.run_bundled_skill(skill_id, "run", [])


@pytest.mark.asyncio
async def test_cli_dispatches_bundled_skill_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_runner(skill_id: str, action: str, arguments: list[str]) -> int:
        calls.append((skill_id, action, arguments))
        return 9

    monkeypatch.setattr(cli, "run_bundled_skill", fake_runner)
    code = await cli.async_main(
        ["system", "skill", "document-reader", "render-pdf", "--check", "--dpi", "96"]
    )

    assert code == 9
    assert calls == [("document-reader", "render-pdf", ["--check", "--dpi", "96"])]


def test_wheel_packaging_uses_only_python_office_dependencies() -> None:
    manifest = (RESOURCE_ROOT.parents[2] / "pyproject.toml").read_text(encoding="utf-8")

    for dependency in (
        "markitdown",
        "markdown-it-py",
        "lxml",
        "pypdfium2",
        "python-docx",
        "python-pptx",
    ):
        assert dependency in manifest
    for removed in ("nodejs-wheel", "paddleocr", "paddlepaddle", "reportlab"):
        assert removed not in manifest.lower()
    assert "pyinstaller" not in manifest.lower()
    assert not (RESOURCE_ROOT.parents[2] / "aeloon.spec").exists()
    assert 'packages = ["aeloon_core"]' in manifest
    assert not list(RESOURCE_ROOT.rglob("package.json"))
    assert not list(RESOURCE_ROOT.rglob("*.js"))
    assert not list(RESOURCE_ROOT.rglob("*.cjs"))
