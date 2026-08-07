from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from aeloon_core import __main__ as cli
from aeloon_core.runtime import skill_runtime
from aeloon_core.runtime.builtin_skills import BUILTIN_SKILL_IDS

RESOURCE_ROOT = Path(__file__).parents[1] / "aeloon_core" / "resources" / "skills"
OFFICE_SKILL_IDS = (
    "markitdown",
    "pdf",
    "paddleocr-doc-parsing",
    "pptx-generator",
    "document-format-skills",
)


def load_script(skill_id: str, script_name: str):
    path = RESOURCE_ROOT / skill_id / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{skill_id}_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_office_execution_skills_are_bundled_and_valid() -> None:
    assert set(OFFICE_SKILL_IDS).issubset(BUILTIN_SKILL_IDS)
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


def test_office_router_names_every_execution_skill() -> None:
    office = (RESOURCE_ROOT / "office" / "SKILL.md").read_text(encoding="utf-8")
    assert all(skill_id in office for skill_id in OFFICE_SKILL_IDS)
    assert ".wps/.dps/.et" in office


def test_bundled_office_scripts_compile() -> None:
    for skill_id in (*OFFICE_SKILL_IDS, "office"):
        for script in (RESOURCE_ROOT / skill_id / "scripts").glob("*.py"):
            compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_bundled_skill_runtime_dispatches_python_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "pdf" / "scripts" / "render_pdf.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(7)\n", encoding="utf-8")
    monkeypatch.setattr(skill_runtime, "bundled_skill_root", lambda: tmp_path)
    original_argv = list(sys.argv)

    assert skill_runtime.run_bundled_skill("pdf", "render", ["--check"]) == 7
    assert sys.argv == original_argv


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
        ["system", "skill", "pdf", "render", "--check", "--dpi", "96"]
    )

    assert code == 9
    assert calls == [("pdf", "render", ["--check", "--dpi", "96"])]


def test_packaging_includes_skill_scripts_and_runtime_dependencies() -> None:
    manifest = (RESOURCE_ROOT.parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    spec = (RESOURCE_ROOT.parents[2] / "aeloon.spec").read_text(encoding="utf-8")
    package_lock = (
        RESOURCE_ROOT / "pptx-generator" / "runtime" / "package-lock.json"
    ).read_text(encoding="utf-8")

    for dependency in (
        "markitdown",
        "nodejs-wheel",
        "paddleocr",
        "paddlepaddle",
        "pypdfium2",
        "python-docx",
    ):
        assert dependency in manifest
    assert "include_py_files=True" in spec
    assert '"nodejs-wheel-binaries"' in spec
    assert '"pptxgenjs": "4.0.1"' in package_lock


def test_office_preflight_can_require_available_python(monkeypatch) -> None:
    preflight = load_script("office", "preflight.py")
    checks = {
        "python": preflight.Check(True, "test"),
        "markitdown": preflight.Check(False, "missing", "install it"),
    }
    monkeypatch.setattr(preflight, "collect_checks", lambda: checks)
    monkeypatch.setattr(
        preflight,
        "COMPONENTS",
        {"python-only": ("python",), "reader": ("markitdown",)},
    )
    assert preflight.main(["--require", "python-only"]) == 0
    assert preflight.main(["--require", "reader"]) == 2


def test_markitdown_converter_rejects_remote_and_native_wps(tmp_path: Path) -> None:
    converter = load_script("markitdown", "convert.py")
    with pytest.raises(ValueError, match="URI"):
        converter._local_file("https://example.com/report.docx")
    wps = tmp_path / "report.wps"
    wps.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="native WPS"):
        converter._local_file(str(wps))


def test_markitdown_converter_uses_local_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    converter = load_script("markitdown", "convert.py")
    calls = []

    class FakeMarkItDown:
        def convert_local(self, source):
            calls.append(source)
            return SimpleNamespace(markdown="# 本地内容")

    monkeypatch.setitem(sys.modules, "markitdown", SimpleNamespace(MarkItDown=FakeMarkItDown))
    source = tmp_path / "report.docx"
    source.write_bytes(b"docx")
    assert converter.convert_file(source) == "# 本地内容"
    assert calls == [source]


def test_paddleocr_wrapper_is_local_only() -> None:
    script = RESOURCE_ROOT / "paddleocr-doc-parsing" / "scripts" / "local_parse.py"
    text = script.read_text(encoding="utf-8")
    assert "PADDLEOCR_ACCESS_TOKEN" not in text
    assert "paddleocr api" not in text
    assert "requests" not in text
    parser = load_script("paddleocr-doc-parsing", "local_parse.py")
    with pytest.raises(ValueError, match="local files"):
        parser.local_input("https://example.com/scan.pdf")
    with parser.deny_network(True):
        with pytest.raises(RuntimeError, match="network access is disabled"):
            parser.socket.create_connection(("example.com", 443))


def test_paddleocr_wrapper_writes_local_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = load_script("paddleocr-doc-parsing", "local_parse.py")

    class FakeImage:
        def save(self, path: Path) -> None:
            path.write_bytes(b"image")

    class FakeResult:
        def __init__(self, page: int) -> None:
            self.page = page
            self.markdown = {
                "markdown_text": f"## 第 {page} 页",
                "markdown_images": {f"assets/page-{page}.png": FakeImage()},
            }

        def save_to_json(self, save_path: str) -> None:
            Path(save_path, "result.json").write_text("{}", encoding="utf-8")

        def save_to_markdown(self, save_path: str) -> None:
            Path(save_path, "result.md").write_text(
                self.markdown["markdown_text"], encoding="utf-8"
            )

    class FakePipeline:
        init_kwargs = None

        def __init__(self, **kwargs) -> None:
            type(self).init_kwargs = kwargs

        def predict(self, *, input: str):
            assert input.endswith("scan.pdf")
            return [FakeResult(1), FakeResult(2)]

        def concatenate_markdown_pages(self, pages):
            return "\n\n".join(page["markdown_text"] for page in pages)

    monkeypatch.setattr(parser, "import_pipeline", lambda: FakePipeline)
    monkeypatch.setattr(parser.os, "environ", {})
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"pdf")
    output_dir = tmp_path / "output"
    args = SimpleNamespace(
        input=str(source),
        output_dir=str(output_dir),
        model_cache=str(tmp_path / "models"),
        device="cpu",
        offline=True,
        use_doc_orientation=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )

    output, pages = parser.parse_document(args)

    assert pages == 2
    assert output.read_text(encoding="utf-8") == "## 第 1 页\n\n## 第 2 页"
    assert (output_dir / "assets" / "page-1.png").read_bytes() == b"image"
    assert (output_dir / "assets" / "page-2.png").read_bytes() == b"image"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["offline"] is True
    assert parser.os.environ["HF_HUB_OFFLINE"] == "1"
    assert parser.os.environ["PADDLE_PDX_CACHE_HOME"] == str(tmp_path / "models")
    assert FakePipeline.init_kwargs["device"] == "cpu"


def test_paddleocr_wrapper_rejects_asset_path_escape(tmp_path: Path) -> None:
    parser = load_script("paddleocr-doc-parsing", "local_parse.py")
    with pytest.raises(RuntimeError, match="outside the output directory"):
        parser.safe_asset_path(tmp_path, "../outside.png")


def test_third_party_license_families_are_preserved() -> None:
    assert "MIT License" in (
        RESOURCE_ROOT / "markitdown" / "LICENSE.txt"
    ).read_text(encoding="utf-8")
    assert "Apache License" in (
        RESOURCE_ROOT / "pdf" / "LICENSE.txt"
    ).read_text(encoding="utf-8")
    assert "Apache License" in (
        RESOURCE_ROOT / "paddleocr-doc-parsing" / "LICENSE.txt"
    ).read_text(encoding="utf-8")
    for skill_id in ("pptx-generator", "document-format-skills"):
        assert "MIT License" in (RESOURCE_ROOT / skill_id / "LICENSE.txt").read_text(
            encoding="utf-8"
        )
