from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "aeloon_core" / "resources" / "skills" / "aeloon-office-lite" / "SKILL.md"
CASES = ROOT / "tests" / "fixtures" / "office-skill-routing.json"

POSITIVE = re.compile(
    r"(?:读取|提取|创建|生成|写入|制作|渲染|检查|read|extract|create|write|render|validate).*"
    r"(?:pdf|word|docx|powerpoint|pptx|excel|xlsx|工作簿|幻灯片)|"
    r"(?:pdf|word|docx|powerpoint|pptx|excel|xlsx|工作簿|幻灯片).*"
    r"(?:读取|提取|创建|生成|写入|制作|渲染|检查|read|extract|create|write|render|validate)",
    re.IGNORECASE,
)
NEGATIVE = re.compile(
    r"(?:宏|修订|批注|动画|复杂模板|像素级|合并|拆分|旋转|加密|填写表单).*"
    r"(?:pdf|word|docx|pptx|excel)|不要创建.*(?:word|幻灯片)",
    re.IGNORECASE,
)


def route_contract(prompt: str) -> str | None:
    if NEGATIVE.search(prompt):
        return None
    return "aeloon-office-lite" if POSITIVE.search(prompt) else None


def test_office_trigger_and_near_miss_contract() -> None:
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    assert payload["schema"] == "office-skill-routing-cases/v1"
    for case in payload["cases"]:
        assert route_contract(case["prompt"]) == case["expected"], case["prompt"]


def test_frontmatter_description_covers_the_trigger_contract() -> None:
    _, frontmatter, _ = SKILL.read_text(encoding="utf-8").split("---", 2)
    description = yaml.safe_load(frontmatter)["description"]
    for term in ("PDF", "PPTX", "DOCX", "XLSX", "读取", "创建", "视觉", "简单"):
        assert term.casefold() in description.casefold(), term
