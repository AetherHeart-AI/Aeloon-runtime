from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "aeloon_core" / "resources" / "skills"
CASES = ROOT / "tests" / "fixtures" / "office-skill-routing.json"

POSITIVE = {
    "document-reader": re.compile(
        r"(?:读取|提取|extract|read|ocr|render).*(?:pdf|扫描|document)|"
        r"(?:pdf|扫描|document).*(?:读取|提取|extract|read|ocr|render)",
        re.IGNORECASE,
    ),
    "word-docx": re.compile(r"(?:word|docx|修订).*(?:生成|替换|合同|create|edit)?", re.IGNORECASE),
    "powerpoint-pptx": re.compile(
        r"(?:powerpoint|pptx|幻灯片).*(?:制作|update|chart|template)?", re.IGNORECASE
    ),
}
NEGATIVE = re.compile(
    r"(?:合并|拆分|旋转|加密|填写表单).*(?:pdf)|"
    r"(?:写入|创建).*(?:xlsx|工作簿)|不要创建.*(?:word|幻灯片)",
    re.IGNORECASE,
)


def route_contract(prompt: str) -> str | None:
    if NEGATIVE.search(prompt):
        return None
    matches = [skill_id for skill_id, pattern in POSITIVE.items() if pattern.search(prompt)]
    return matches[0] if len(matches) == 1 else None


def test_office_trigger_and_near_miss_contract() -> None:
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    assert payload["schema"] == "office-skill-routing-cases/v1"
    for case in payload["cases"]:
        assert route_contract(case["prompt"]) == case["expected"], case["prompt"]
        assert case["expected"] != "office"


def test_frontmatter_descriptions_cover_the_trigger_contract() -> None:
    required_terms = {
        "document-reader": ("PDF", "OCR", "rendering", "do not use for creating"),
        "word-docx": ("Word", "DOCX", "track changes", "comments"),
        "powerpoint-pptx": ("PowerPoint", "PPTX", "template", "chart"),
    }
    for skill_id, terms in required_terms.items():
        skill = (SKILLS / skill_id / "SKILL.md").read_text(encoding="utf-8")
        _, frontmatter, _ = skill.split("---", 2)
        description = yaml.safe_load(frontmatter)["description"]
        for term in terms:
            assert term.casefold() in description.casefold(), (skill_id, term)
