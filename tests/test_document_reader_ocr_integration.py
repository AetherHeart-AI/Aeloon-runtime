from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "aeloon_core" / "resources" / "skills" / "document-reader"
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "document-reader-ocr" / "golden.json"
RUN_BENCHMARK = os.environ.get("AELOON_RUN_OCR_BENCHMARK") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not RUN_BENCHMARK,
        reason="set AELOON_RUN_OCR_BENCHMARK=1 to prepare models and run OCR",
    ),
]


def load_reader():
    path = SKILL_DIR / "scripts" / "cli.py"
    spec = importlib.util.spec_from_file_location("ocr_benchmark_document_reader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_cjk_font() -> Path:
    configured = os.environ.get("AELOON_OCR_BENCHMARK_FONT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise AssertionError(
        "CJK font missing; install fonts-noto-cjk or set AELOON_OCR_BENCHMARK_FONT"
    )


def draw_golden_pdf(target: Path, golden: dict, font_path: Path) -> None:
    dpi = int(golden["dpi"])
    width, height = round(8.27 * dpi), round(11.69 * dpi)
    regular = ImageFont.truetype(str(font_path), 34)
    small = ImageFont.truetype(str(font_path), 26)
    title = ImageFont.truetype(str(font_path), 52)
    pages: list[Image.Image] = []
    for page_number, page in enumerate(golden["pages"], 1):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((120, 70), "埃隆文档识别黄金样本", font=small, fill="black")
        draw.line((120, 115, width - 120, 115), fill="black", width=2)
        draw.text((120, 190), page["title"], font=title, fill="black")
        y = 310
        for line in page["body"]:
            draw.text((120, y), line, font=regular, fill="black")
            y += 72
        y += 70
        left, right = 180, width - 180
        row_height = 100
        middle = (left + right) // 2
        for row_index, row in enumerate(page["table"]):
            top = y + row_index * row_height
            bottom = top + row_height
            draw.rectangle((left, top, right, bottom), outline="black", width=3)
            draw.line((middle, top, middle, bottom), fill="black", width=3)
            draw.text((left + 30, top + 25), row[0], font=regular, fill="black")
            draw.text((middle + 30, top + 25), row[1], font=regular, fill="black")
        draw.line((120, height - 145, width - 120, height - 145), fill="black", width=2)
        draw.text(
            (120, height - 110),
            f"内部质量基线 第{page_number}页 共{len(golden['pages'])}页",
            font=small,
            fill="black",
        )
        pages.append(image)
    pages[0].save(target, "PDF", save_all=True, append_images=pages[1:], resolution=dpi)


def expected_values(golden: dict) -> tuple[str, list[str]]:
    text: list[str] = []
    cells: list[str] = []
    for page in golden["pages"]:
        text.extend([page["title"], *page["body"]])
        cells.extend(cell for row in page["table"] for cell in row)
    return "".join(text), cells


def assert_quality(reader, golden: dict, markdown: str, metrics: dict) -> None:
    expected_text, expected_cells = expected_values(golden)
    quality = reader.evaluate_chinese_ocr_quality(
        expected_text=expected_text,
        actual_text=markdown,
        expected_table_cells=expected_cells,
        expected_pages=len(golden["pages"]),
        successful_pages=int(metrics["pages"]),
    )
    assert quality["page_coverage"] == 1.0, quality
    assert quality["chinese_character_recall"] >= 0.90, quality
    assert quality["table_cell_recall"] >= 0.85, quality
    assert quality["passed"] is True, quality


def test_three_page_chinese_scan_online_then_fully_offline(tmp_path: Path) -> None:
    reader = load_reader()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert len(golden["pages"]) >= 3
    assert 200 <= golden["dpi"] <= 300
    source = tmp_path / "chinese-scan-golden.pdf"
    draw_golden_pdf(source, golden, find_cjk_font())
    cache_dir = Path(os.environ.get("AELOON_OCR_BENCHMARK_CACHE", tmp_path / "cache"))

    prepared = reader.prepare_ocr(cache_dir)
    assert prepared["complete"] is True

    online = reader.ingest_document(
        str(source), tmp_path / "online", engine="docling", cache_dir=cache_dir
    )
    assert online.status != "failed_for_agent"
    online_evidence = json.loads(online.evidence.read_text(encoding="utf-8"))
    assert_quality(
        reader,
        golden,
        online.markdown.read_text(encoding="utf-8"),
        online_evidence["attempts"][-1]["metrics"],
    )

    offline = reader.ingest_document(
        str(source),
        tmp_path / "offline",
        engine="docling",
        offline=True,
        cache_dir=cache_dir,
    )
    assert offline.status != "failed_for_agent"
    offline_evidence = json.loads(offline.evidence.read_text(encoding="utf-8"))
    offline_metrics = offline_evidence["attempts"][-1]["metrics"]
    assert offline_metrics["offline_network_probe"] == "blocked"
    assert_quality(
        reader,
        golden,
        offline.markdown.read_text(encoding="utf-8"),
        offline_metrics,
    )
