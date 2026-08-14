from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from aeloon_core import __main__ as cli
from aeloon_core.runtime import skill_runtime
from aeloon_core.runtime.builtin_skills import BUILTIN_SKILL_IDS

RESOURCE_ROOT = Path(__file__).parents[1] / "aeloon_core" / "resources" / "skills"
OFFICE_SKILL_IDS = ("aeloon-office-lite",)


def test_only_office_lite_is_bundled_and_valid() -> None:
    assert BUILTIN_SKILL_IDS == OFFICE_SKILL_IDS
    assert {path.parent.name for path in RESOURCE_ROOT.glob("*/SKILL.md")} == set(OFFICE_SKILL_IDS)
    skill_dir = RESOURCE_ROOT / "aeloon-office-lite"
    skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, body = skill_text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    assert set(metadata) == {"name", "description"}
    assert metadata["name"] == "aeloon-office-lite"
    assert "present_files" in body
    assert "aeloon-core system skill" in body
    assert "aeloon system skill" not in body
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in body
    interface = yaml.safe_load((skill_dir / "agents" / "openai.yaml").read_text(encoding="utf-8"))[
        "interface"
    ]
    assert "$aeloon-office-lite" in interface["default_prompt"]


def test_office_lite_script_compiles() -> None:
    script = RESOURCE_ROOT / "aeloon-office-lite" / "scripts" / "cli.py"
    compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_bundled_skill_runtime_prepends_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "aeloon-office-lite" / "scripts" / "cli.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import sys\nassert sys.argv[1:] == ['read', '--check']\nraise SystemExit(7)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_runtime, "bundled_skill_root", lambda: tmp_path)
    original_argv = list(sys.argv)

    assert skill_runtime.run_bundled_skill("aeloon-office-lite", "read", ["--check"]) == 7
    assert sys.argv == original_argv


def test_bundled_skill_runtime_reports_unknown_ids_and_actions() -> None:
    with pytest.raises(
        ValueError, match="expected one of: preflight, read, write, render, validate"
    ):
        skill_runtime.run_bundled_skill("aeloon-office-lite", "unknown", [])
    with pytest.raises(ValueError, match="unknown bundled Skill 'missing'"):
        skill_runtime.run_bundled_skill("missing", "run", [])


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
        ["system", "skill", "aeloon-office-lite", "render", "file.pdf", "--dpi", "96"]
    )

    assert code == 9
    assert calls == [("aeloon-office-lite", "render", ["file.pdf", "--dpi", "96"])]


def test_wheel_packaging_uses_only_lite_python_office_dependencies() -> None:
    manifest = (RESOURCE_ROOT.parents[2] / "pyproject.toml").read_text(encoding="utf-8")

    for dependency in (
        "openpyxl",
        "pypdfium2",
        "python-docx",
        "python-pptx",
        "reportlab",
    ):
        assert dependency in manifest
    for removed in (
        "markitdown",
        "markdown-it-py",
        "pdfplumber",
        "lxml",
        "nodejs-wheel",
        "paddleocr",
        "paddlepaddle",
    ):
        assert removed not in manifest.lower()
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in manifest
    assert '[project.scripts]\naeloon-core = "aeloon_core.__main__:main"' in manifest
    assert '\naeloon = "' not in manifest
    assert not list(RESOURCE_ROOT.rglob("package.json"))
    assert not list(RESOURCE_ROOT.rglob("*.js"))
