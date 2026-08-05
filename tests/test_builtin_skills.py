from pathlib import Path

from aeloon_core.runtime.builtin_skills import (
    BUILTIN_SKILL_IDS,
    provision_builtin_skills,
)
from aeloon_core.runtime.resources import ResourceLoader
from aeloon_core.runtime.service import RuntimeService


def test_provision_builtin_skills_is_idempotent_and_preserves_existing(tmp_path: Path) -> None:
    custom = tmp_path / "skills" / "office" / "SKILL.md"
    custom.parent.mkdir(parents=True)
    custom.write_text("custom office skill\n", encoding="utf-8")

    copied = provision_builtin_skills(tmp_path)

    assert copied == ("ppt", "document-writing", "reports")
    assert custom.read_text(encoding="utf-8") == "custom office skill\n"
    assert provision_builtin_skills(tmp_path) == ()
    assert all(
        (tmp_path / "skills" / skill_id / "SKILL.md").is_file()
        for skill_id in BUILTIN_SKILL_IDS
    )
    assert all(
        (tmp_path / "skills" / skill_id / "agents" / "openai.yaml").is_file()
        for skill_id in BUILTIN_SKILL_IDS
        if skill_id != "office"
    )


def test_runtime_bootstrap_copies_and_discovers_builtin_skills(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    RuntimeService(config_path=tmp_path / "config.json", data_dir=data_dir)

    loader = ResourceLoader(cwd=tmp_path, agent_dir=data_dir)
    resources = loader.reload()
    assert {skill.name for skill in resources.skills} == set(BUILTIN_SKILL_IDS)
    assert all(skill.content == "" for skill in resources.skills)
    loaded = loader.load_skill("ppt")
    assert "定义演示策略" in loaded.content
