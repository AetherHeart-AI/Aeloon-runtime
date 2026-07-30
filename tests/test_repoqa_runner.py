from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from benchmarks.adapters.repoqa import RepoQAAdapter
from benchmarks.harness.base import (
    HarnessInvocation,
    HarnessResult,
    ProcessOutcome,
)
from benchmarks.repoqa.runner import (
    RESULT_PREFIX,
    RepoQACase,
    WorkspaceCache,
    build_prompt,
    changed_files,
    evaluate_answer,
    load_cases,
    parse_answer,
)


class FakeRepoQAHarness:
    version = "fake@1"
    model = "fake-model"

    def __init__(self, name: str, *, mutate: bool = False) -> None:
        self.name = name
        self.mutate = mutate

    def run(self, request) -> HarnessResult:
        if self.mutate:
            (request.workspace / "collateral.txt").write_text("changed\n", encoding="utf-8")
        invocation = HarnessInvocation(
            command=[self.name, "<prompt>"],
            cwd=request.workspace,
            prompt_argument=True,
        )
        return HarnessResult(
            harness=self.name,
            version=self.version,
            invocation=invocation,
            process=ProcessOutcome(
                returncode=0,
                stdout="completed",
                stderr="",
                duration_ms=1,
            ),
            status="completed",
            final_content=(f'{RESULT_PREFIX} {{"path":"src/search.py","symbol":"find_target"}}'),
        )


def _repository(
    name: str,
    *,
    first_symbol: str = "find_target",
    second_symbol: str = "find_other",
) -> dict[str, object]:
    return {
        "repo": name,
        "commit_sha": "a" * 40,
        "content": {
            "src/search.py": (
                f"def {first_symbol}():\n    return 1\n\ndef {second_symbol}():\n    return 2\n"
            )
        },
        "needles": [
            {
                "path": "src/search.py",
                "name": first_symbol,
                "start_line": 0,
                "description": f"Return the primary value for {name}.",
            },
            {
                "path": "src/search.py",
                "name": second_symbol,
                "start_line": 3,
                "description": f"Return the secondary value for {name}.",
            },
        ],
    }


def _write_dataset(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def _case() -> RepoQACase:
    return RepoQACase(
        instance_id="python::example/repo::find_target",
        language="python",
        repository="example/repo",
        commit_sha="a" * 40,
        description="Return the primary value.",
        target_path="src/search.py",
        target_symbol="find_target",
        repository_files={
            ".gitignore": "ignored.txt\n",
            "src/search.py": "def find_target():\n    return 1\n",
        },
    )


def test_load_cases_interleaves_languages_and_repositories(tmp_path: Path) -> None:
    dataset = tmp_path / "repoqa.json.gz"
    _write_dataset(
        dataset,
        {
            "rust": [_repository("rust/one"), _repository("rust/two")],
            "python": [_repository("python/one"), _repository("python/two")],
        },
    )

    cases = load_cases(dataset)

    assert [(case.language, case.repository) for case in cases[:4]] == [
        ("python", "python/one"),
        ("rust", "rust/one"),
        ("python", "python/two"),
        ("rust", "rust/two"),
    ]
    assert [case.target_symbol for case in cases[:4]] == ["find_target"] * 4
    assert [case.target_symbol for case in cases[4:]] == ["find_other"] * 4


def test_load_cases_rejects_repository_path_traversal(tmp_path: Path) -> None:
    dataset = tmp_path / "repoqa.json.gz"
    repository = _repository("unsafe/repo")
    repository["content"] = {"../secret.py": "SECRET = True\n"}
    _write_dataset(dataset, {"python": [repository]})

    with pytest.raises(RuntimeError, match="repository-relative"):
        load_cases(dataset)


def test_prompt_requires_tools_and_does_not_reveal_target() -> None:
    case = _case()

    prompt = build_prompt(case)

    assert "local search/read tools" in prompt
    assert "Do not use the network" in prompt
    assert RESULT_PREFIX in prompt
    assert case.target_path not in prompt
    assert case.target_symbol not in prompt


def test_answer_parser_and_exact_read_only_grading() -> None:
    case = _case()
    final_content = (
        "Located it after searching the repository.\n"
        f'{RESULT_PREFIX} {{"path":"src/search.py","symbol":"find_target"}}'
    )

    answer, error = parse_answer(final_content)
    evaluation = evaluate_answer(case, final_content, [])
    collateral = evaluate_answer(case, final_content, ["notes.txt"])

    assert error is None
    assert answer is not None
    assert (answer.path, answer.symbol) == ("src/search.py", "find_target")
    assert evaluation == {
        "matched_target": True,
        "path_match": True,
        "symbol_match": True,
        "clean_worktree": True,
        "reported_path": "src/search.py",
        "reported_symbol": "find_target",
        "parse_error": None,
    }
    assert collateral["matched_target"] is True
    assert collateral["clean_worktree"] is False


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("nothing structured", "did not contain"),
        (f"{RESULT_PREFIX} not-json", "JSON was invalid"),
        (
            f'{RESULT_PREFIX} {{"path":"../secret","symbol":"find_target"}}',
            "repository-relative",
        ),
        (
            f'{RESULT_PREFIX} {{"path":"src/search.py","symbol":""}}',
            "must not be empty",
        ),
    ],
)
def test_answer_parser_rejects_invalid_results(content: str, message: str) -> None:
    answer, error = parse_answer(content)

    assert answer is None
    assert error is not None
    assert message in error


def test_workspace_cache_restores_read_only_repository(tmp_path: Path) -> None:
    case = _case()
    cache = WorkspaceCache(tmp_path / "cache")
    workspace, baseline = cache.prepare(case, "fake")
    (workspace / "src" / "search.py").write_text("broken\n", encoding="utf-8")
    (workspace / "extra.txt").write_text("extra\n", encoding="utf-8")
    (workspace / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    assert changed_files(workspace) == ["extra.txt", "ignored.txt", "src/search.py"]

    same_workspace, same_baseline = cache.prepare(case, "fake")

    assert same_workspace == workspace
    assert same_baseline == baseline
    assert (
        (workspace / "src" / "search.py").read_text(encoding="utf-8").startswith("def find_target")
    )
    assert not (workspace / "extra.txt").exists()
    assert not (workspace / "ignored.txt").exists()
    assert changed_files(workspace) == []


def test_repoqa_adapter_grades_location_and_collateral_damage(tmp_path: Path) -> None:
    adapter = RepoQAAdapter(project_root=tmp_path, limit=1)
    _write_dataset(adapter.dataset_path, {"python": [_repository("example/repo")]})
    harnesses = [
        FakeRepoQAHarness("clean"),
        FakeRepoQAHarness("mutating", mutate=True),
    ]

    summary = adapter.execute(harnesses)  # type: ignore[arg-type]

    assert summary["recorded_cases"] == 2
    assert summary["passed"] == 1
    assert summary["harnesses"]["clean"]["passed"] == 1
    assert summary["harnesses"]["mutating"]["passed"] == 0
    mutating_record = json.loads(
        (adapter.run.output_dir / "mutating" / "results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert mutating_record["evaluation"]["matched_target"] is True
    assert mutating_record["evaluation"]["clean_worktree"] is False
    assert mutating_record["changed_files"] == ["collateral.txt"]
    manifest = json.loads((adapter.run.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["grading"]["target"] == "exact-path-and-symbol"
