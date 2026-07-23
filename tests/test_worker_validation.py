from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelRetry

from aeloon_core.pydantic_runtime import ToolObservation
from aeloon_core.worker_state import WorkerReport
from aeloon_core.worker_terminal_tools import CompleteWorkArgs, RequestMasterArgs
from aeloon_core.worker_validation import validate_worker_terminal_output


def _ctx(*, tools: list[str], observations: list[ToolObservation]) -> SimpleNamespace:
    return SimpleNamespace(
        deps=SimpleNamespace(
            tools_used=tools,
            tool_observations=observations,
        )
    )


def _exec(command: str, exit_code: int = 0) -> ToolObservation:
    return ToolObservation(
        name="exec",
        arguments={"command": command},
        result=f"result\n\nExit code: {exit_code}",
    )


def _evidence(
    kind: str,
    locator: str,
    *,
    status: str,
    method: str | None = None,
    finding_id: str | None = None,
) -> dict[str, str]:
    item = {
        "kind": kind,
        "locator": locator,
        "claim": f"claim for {locator}",
        "status": status,
    }
    if method is not None:
        item["method"] = method
    if finding_id is not None:
        item["finding_id"] = finding_id
    return item


def test_terminal_output_rejects_legacy_evidence_strings() -> None:
    with pytest.raises(ValidationError):
        CompleteWorkArgs.model_validate(
            {"summary": "done", "evidence": ["tests passed"]}
        )


def test_terminal_evidence_enforces_field_lengths() -> None:
    with pytest.raises(ValidationError, match="at most 1000 characters"):
        CompleteWorkArgs.model_validate(
            {
                "summary": "done",
                "evidence": [
                    {
                        **_evidence("file", "module.py:1", status="observed"),
                        "claim": "x" * 1_001,
                    }
                ],
            }
        )


def test_persisted_reports_normalize_legacy_evidence() -> None:
    report = WorkerReport(summary="old result", evidence=("tests passed",))

    assert report.evidence[0].kind == "legacy"
    assert report.evidence[0].locator == "tests passed"
    assert report.evidence[0].status == "observed"


def test_builder_completion_accepts_real_checks_and_file_lines(tmp_path) -> None:
    target = tmp_path / "module.py"
    target.write_text("first\nsecond\n", encoding="utf-8")
    output = CompleteWorkArgs.model_validate(
        {
            "summary": "implemented and verified",
            "evidence": [
                _evidence(
                    "file",
                    "module.py:2",
                    status="observed",
                ),
                _evidence(
                    "test",
                    "tests/test_module.py::test_behavior",
                    status="passed",
                    method="pytest tests/test_module.py",
                ),
                _evidence(
                    "typecheck",
                    "not configured",
                    status="not_applicable",
                ),
                _evidence(
                    "lint",
                    "not configured",
                    status="not_applicable",
                ),
            ],
        }
    )

    validated = validate_worker_terminal_output(
        _ctx(
            tools=["write", "exec"],
            observations=[_exec("pytest tests/test_module.py")],
        ),
        output,
        worker_type_id="builder",
        workspace=tmp_path,
    )

    assert validated is output


def test_builder_completion_rejects_unexecuted_pass_claim(tmp_path) -> None:
    output = CompleteWorkArgs.model_validate(
        {
            "summary": "claimed success",
            "evidence": [
                _evidence(
                    "test",
                    "tests/test_module.py",
                    status="passed",
                    method="pytest tests/test_module.py",
                ),
                _evidence("typecheck", "not configured", status="not_applicable"),
                _evidence("lint", "not configured", status="not_applicable"),
            ],
        }
    )

    with pytest.raises(ModelRetry, match="not backed by an exec call"):
        validate_worker_terminal_output(
            _ctx(tools=["write"], observations=[]),
            output,
            worker_type_id="builder",
            workspace=tmp_path,
        )


def test_builder_completion_rejects_failed_command_claimed_as_passed(tmp_path) -> None:
    output = CompleteWorkArgs.model_validate(
        {
            "summary": "claimed success",
            "evidence": [
                _evidence(
                    "test",
                    "tests/test_module.py",
                    status="passed",
                    method="pytest tests/test_module.py",
                ),
                _evidence("typecheck", "not configured", status="not_applicable"),
                _evidence("lint", "not configured", status="not_applicable"),
            ],
        }
    )

    with pytest.raises(ModelRetry, match="did not exit successfully"):
        validate_worker_terminal_output(
            _ctx(
                tools=["exec"],
                observations=[_exec("pytest tests/test_module.py", exit_code=1)],
            ),
            output,
            worker_type_id="builder",
            workspace=tmp_path,
        )


def test_builder_exec_is_mutation_capable_and_requires_the_full_gate(tmp_path) -> None:
    output = CompleteWorkArgs(summary="changed files through a generator")

    with pytest.raises(ModelRetry, match="verification categories"):
        validate_worker_terminal_output(
            _ctx(
                tools=["exec"],
                observations=[_exec("python -m project.generate")],
            ),
            output,
            worker_type_id="builder",
            workspace=tmp_path,
        )


def test_file_evidence_requires_an_existing_valid_line(tmp_path) -> None:
    target = tmp_path / "module.py"
    target.write_text("only line\n", encoding="utf-8")
    output = RequestMasterArgs.model_validate(
        {
            "summary": "blocked",
            "question": "What should win?",
            "evidence": [
                _evidence("file", "module.py:2", status="observed"),
            ],
        }
    )

    with pytest.raises(ModelRetry, match="line range is invalid"):
        validate_worker_terminal_output(
            _ctx(tools=[], observations=[]),
            output,
            worker_type_id="builder",
            workspace=tmp_path,
        )


def test_reviewer_requires_runtime_evidence_and_finding_links(tmp_path) -> None:
    output = CompleteWorkArgs.model_validate(
        {
            "summary": "[F-1] high: reproduced incorrect behavior",
            "evidence": [
                _evidence(
                    "test",
                    "tests/test_bug.py::test_reproduction",
                    status="failed",
                    method="pytest tests/test_bug.py",
                    finding_id="F-1",
                )
            ],
        }
    )
    validated = validate_worker_terminal_output(
        _ctx(
            tools=["exec"],
            observations=[_exec("pytest tests/test_bug.py", exit_code=1)],
        ),
        output,
        worker_type_id="reviewer",
        workspace=tmp_path,
    )
    assert validated is output

    missing_link = output.model_copy(
        update={"evidence": [output.evidence[0].model_copy(update={"finding_id": None})]}
    )
    with pytest.raises(ModelRetry, match="linked evidence"):
        validate_worker_terminal_output(
            _ctx(
                tools=["exec"],
                observations=[_exec("pytest tests/test_bug.py", exit_code=1)],
            ),
            missing_link,
            worker_type_id="reviewer",
            workspace=tmp_path,
        )


def test_reviewer_cannot_complete_from_code_reading_alone(tmp_path) -> None:
    output = CompleteWorkArgs(
        summary="No findings after reading the diff.",
        evidence=[
            _evidence("file", "module.py:1", status="observed"),
        ],
    )
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ModelRetry, match="actual test or runtime check"):
        validate_worker_terminal_output(
            _ctx(tools=["read"], observations=[]),
            output,
            worker_type_id="reviewer",
            workspace=tmp_path,
        )


def test_continuation_inherits_validated_evidence_only_without_new_mutation(
    tmp_path,
) -> None:
    inherited = WorkerReport(
        summary="prior verified checkpoint",
        evidence=(
            _evidence(
                "test",
                "tests/test_prior.py::test_behavior",
                status="passed",
                method="pytest tests/test_prior.py",
            ),
            _evidence("typecheck", "not configured", status="not_applicable"),
            _evidence("lint", "not configured", status="not_applicable"),
        ),
    ).evidence
    output = CompleteWorkArgs(summary="continued without changing files")

    validated = validate_worker_terminal_output(
        _ctx(tools=[], observations=[]),
        output,
        worker_type_id="builder",
        workspace=tmp_path,
        inherited_evidence=inherited,
    )

    assert len(validated.evidence) == 3
    with pytest.raises(ModelRetry, match="verification categories"):
        validate_worker_terminal_output(
            _ctx(tools=["write"], observations=[]),
            output,
            worker_type_id="builder",
            workspace=tmp_path,
            inherited_evidence=inherited,
        )
