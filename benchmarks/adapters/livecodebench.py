"""Thin integration of LiveCodeBench v6 loading and official evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from benchmarks.adapters.base import BenchmarkAdapter, run_checked
from benchmarks.harness.base import (
    Harness,
    HarnessRequest,
    HarnessResult,
)
from benchmarks.livecodebench import runner as official
from benchmarks.progress import ProgressBar, info

_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT")


@dataclass(frozen=True)
class _Generation:
    prompt: str
    code: str
    result: HarnessResult


def _parallel_map_ordered(
    function: Callable[[_InputT], _OutputT],
    items: list[_InputT],
    *,
    max_workers: int,
    thread_name_prefix: str,
) -> list[_OutputT]:
    if max_workers <= 1:
        return [function(item) for item in items]

    results: dict[int, _OutputT] = {}
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
    ) as executor:
        futures = {executor.submit(function, item): index for index, item in enumerate(items)}
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [results[index] for index in range(len(items))]


class LiveCodeBenchAdapter(BenchmarkAdapter):
    name = "livecodebench"
    repository_url = "https://github.com/LiveCodeBench/LiveCodeBench.git"
    release_version = "v6"
    scenarios = ("code-generation", "self-repair")

    @property
    def environment_dir(self) -> Path:
        return self.run.workspace_root / "environments" / self.name

    @property
    def environment_python(self) -> Path:
        return self.environment_dir / "bin" / "python"

    def install_dependencies(self) -> None:
        requirements = self.project_root / "benchmarks" / "livecodebench" / "requirements.txt"
        fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
        marker = self.environment_dir / ".aeloon-dependencies.json"
        source_revision = self.source_revision()
        if self.environment_python.is_file() and marker.is_file():
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if (
                payload.get("requirements_sha256") == fingerprint
                and payload.get("source_revision") == source_revision
            ):
                info(
                    "[%s] Reusing evaluator environment: %s",
                    self.name,
                    self.environment_dir,
                )
                return

        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError(
                "Preparing LiveCodeBench requires 'uv' on PATH. Install uv and rerun the benchmark."
            )
        if not self.environment_python.is_file():
            info(
                "[%s] Creating evaluator environment: %s",
                self.name,
                self.environment_dir,
            )
            self.environment_dir.parent.mkdir(parents=True, exist_ok=True)
            run_checked(
                [
                    uv,
                    "venv",
                    "--python",
                    "3.11",
                    str(self.environment_dir),
                ],
                cwd=self.project_root,
            )
        info("[%s] Installing/updating evaluator dependencies", self.name)
        run_checked(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(self.environment_python),
                "--requirement",
                str(requirements),
            ],
            cwd=self.project_root,
        )
        marker.write_text(
            json.dumps(
                {
                    "requirements_sha256": fingerprint,
                    "source_revision": source_revision,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        info("[%s] Evaluator dependencies are ready", self.name)

    def load_cases(self) -> list[official.LiveCodeBenchCase]:
        return self.selected(
            official.load_cases(
                self.run.source_dir,
                self.environment_python,
                self.release_version,
            )
        )

    def evaluate(
        self,
        *,
        generations: dict[str, tuple[str, HarnessResult]],
    ) -> dict[str, dict[str, Any]]:
        if not generations:
            return {}
        info(
            "[%s] Running official evaluator for %d generations",
            self.name,
            len(generations),
        )
        payload = [
            {"instance_id": instance_id, "code": code}
            for instance_id, (code, _result) in generations.items()
        ]
        self.run.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=".livecodebench-eval-",
            dir=self.run.output_dir,
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False)
            input_path = Path(stream.name)
        try:
            response = official._run_adapter(
                livecodebench_root=self.run.source_dir,
                livecodebench_python=self.environment_python,
                release_version=self.release_version,
                command="evaluate",
                input_path=input_path,
                num_processes=4,
                timeout=6,
                process_timeout=7200.0,
            )
        finally:
            input_path.unlink(missing_ok=True)

        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError("LiveCodeBench evaluator returned no results.")
        evaluations: dict[str, dict[str, Any]] = {}
        for item in raw_results:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("instance_id"), str)
                or not isinstance(item.get("passed"), bool)
                or not isinstance(item.get("metadata"), dict)
            ):
                raise RuntimeError("LiveCodeBench evaluator returned an invalid result.")
            instance_id = item["instance_id"]
            generation = generations.get(instance_id)
            if generation is None:
                raise RuntimeError(
                    f"LiveCodeBench evaluator returned unexpected case: {instance_id}"
                )
            if instance_id in evaluations:
                raise RuntimeError(f"LiveCodeBench evaluator duplicated case: {instance_id}")
            _code, result = generation
            oracle_passed = item["passed"]
            evaluations[instance_id] = {
                "passed": result.status == "completed" and oracle_passed,
                "oracle_passed": oracle_passed,
                "metadata": item["metadata"],
            }
        missing = sorted(set(generations) - evaluations.keys())
        if missing:
            raise RuntimeError(f"LiveCodeBench evaluator omitted cases: {', '.join(missing)}")
        info("[%s] Official evaluation completed", self.name)
        return evaluations

    def execute(self, harnesses: list[Harness]) -> dict[str, Any]:
        info("[%s] Loading official %s cases", self.name, self.release_version)
        cases = self.load_cases()
        if not cases:
            raise RuntimeError("LiveCodeBench contains no selected cases.")
        info("[%s] Loaded %d cases", self.name, len(cases))

        manifest_path = self.run.output_dir / "manifest.json"
        manifest = self._start_manifest(harnesses, cases)
        self.write_json(manifest_path, manifest)
        summaries: dict[str, dict[str, Any]] = {}
        try:
            for harness in harnesses:
                info(
                    "[%s/%s] Starting code-generation and self-repair",
                    self.name,
                    harness.name,
                )
                summaries[harness.name] = self._run_harness(harness, cases)
                info("[%s/%s] All scenarios completed", self.name, harness.name)
        except BaseException:
            manifest["status"] = "interrupted"
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            self.write_json(manifest_path, manifest)
            raise

        summary = {
            "schema_version": 1,
            "benchmark": self.name,
            "run_id": self.run.run_id,
            "workers": self.workers,
            "release_version": self.release_version,
            "selected_cases": len(cases),
            "archive": str(self.run.output_dir),
            "harnesses": summaries,
        }
        self.write_json(self.run.output_dir / "summary.json", summary)
        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        self.write_json(manifest_path, manifest)
        return summary

    def _start_manifest(
        self,
        harnesses: list[Harness],
        cases: list[official.LiveCodeBenchCase],
    ) -> dict[str, Any]:
        return {
            **self.manifest(harnesses, status="running"),
            "created_at": datetime.now(UTC).isoformat(),
            "release_version": self.release_version,
            "scenarios": list(self.scenarios),
            "cases": [case.instance_id for case in cases],
        }

    def _run_harness(
        self,
        harness: Harness,
        cases: list[official.LiveCodeBenchCase],
    ) -> dict[str, Any]:
        records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        baseline_records: dict[str, dict[str, Any]] = {}
        pending_baselines = cases

        worker_count = min(self.workers, len(pending_baselines))
        if worker_count > 1:
            info(
                "[%s/%s] Parallel case execution enabled: workers=%d",
                self.name,
                harness.name,
                worker_count,
            )

        def generate_baseline(
            item: tuple[int, official.LiveCodeBenchCase],
        ) -> tuple[str, _Generation]:
            index, case = item
            baseline_progress.set_detail(f"running {case.instance_id}")
            info(
                "[%s/%s] code-generation %d/%d: %s",
                self.name,
                harness.name,
                index,
                len(pending_baselines),
                case.instance_id,
            )
            prompt = official._code_generation_prompt(case)
            result = self._invoke(
                harness,
                case=case,
                scenario="code-generation",
                prompt=prompt,
            )
            code, _error = official._extract_python_code(result.final_content)
            generation = _Generation(prompt=prompt, code=code, result=result)
            baseline_progress.advance(detail=f"completed {case.instance_id}")
            return case.instance_id, generation

        indexed_cases = list(enumerate(pending_baselines, start=1))
        with ProgressBar(
            f"{self.name}/{harness.name} code-generation",
            total=len(indexed_cases),
        ) as baseline_progress:
            generated_baselines = _parallel_map_ordered(
                generate_baseline,
                indexed_cases,
                max_workers=worker_count,
                thread_name_prefix=f"livecodebench-{harness.name}",
            )
        new_baseline_generations = {
            instance_id: generation for instance_id, generation in generated_baselines
        }
        baseline_evaluations = (
            self.evaluate(
                generations={
                    instance_id: (generation.code, generation.result)
                    for instance_id, generation in new_baseline_generations.items()
                }
            )
            if new_baseline_generations
            else {}
        )
        self._log_score(
            harness=harness,
            scenario="code-generation",
            evaluations=baseline_evaluations,
        )
        for case in pending_baselines:
            generation = new_baseline_generations[case.instance_id]
            record = self._record(
                harness=harness,
                case=case,
                scenario="code-generation",
                prompt=generation.prompt,
                code=generation.code,
                result=generation.result,
                evaluation=baseline_evaluations[case.instance_id],
            )
            self._archive_record(record, generation.result)
            records_by_key[("code-generation", case.instance_id)] = record
            baseline_records[case.instance_id] = record

        repair_records: dict[str, dict[str, Any]] = {}
        pending_repairs = [
            case
            for case in cases
            if not baseline_records[case.instance_id]["evaluation"]["oracle_passed"]
        ]

        def generate_repair(
            item: tuple[int, official.LiveCodeBenchCase],
        ) -> tuple[str, _Generation]:
            index, case = item
            baseline = baseline_records[case.instance_id]
            repair_progress.set_detail(f"running {case.instance_id}")
            info(
                "[%s/%s] self-repair %d/%d: %s",
                self.name,
                harness.name,
                index,
                len(pending_repairs),
                case.instance_id,
            )
            prompt = official._self_repair_prompt(
                case,
                code=baseline["generation"]["code"],
                metadata=baseline["evaluation"]["metadata"],
            )
            result = self._invoke(
                harness,
                case=case,
                scenario="self-repair",
                prompt=prompt,
            )
            code, _error = official._extract_python_code(result.final_content)
            generation = _Generation(prompt=prompt, code=code, result=result)
            repair_progress.advance(detail=f"completed {case.instance_id}")
            return case.instance_id, generation

        indexed_repairs = list(enumerate(pending_repairs, start=1))
        repair_worker_count = min(self.workers, len(pending_repairs))
        with ProgressBar(
            f"{self.name}/{harness.name} self-repair",
            total=len(indexed_repairs),
        ) as repair_progress:
            generated_repairs = _parallel_map_ordered(
                generate_repair,
                indexed_repairs,
                max_workers=repair_worker_count,
                thread_name_prefix=f"livecodebench-repair-{harness.name}",
            )
        new_repair_generations = {
            instance_id: generation for instance_id, generation in generated_repairs
        }
        repair_evaluations = (
            self.evaluate(
                generations={
                    instance_id: (generation.code, generation.result)
                    for instance_id, generation in new_repair_generations.items()
                }
            )
            if new_repair_generations
            else {}
        )
        for case in cases:
            baseline = baseline_records[case.instance_id]
            attempted = case.instance_id in new_repair_generations
            if attempted:
                generation = new_repair_generations[case.instance_id]
                code = generation.code
                result = generation.result
                evaluation = repair_evaluations[case.instance_id]
                prompt = generation.prompt
            else:
                code = baseline["generation"]["code"]
                result = new_baseline_generations[case.instance_id].result
                evaluation = baseline["evaluation"]
                prompt = None
            record = self._record(
                harness=harness,
                case=case,
                scenario="self-repair",
                prompt=prompt,
                code=code,
                result=result,
                evaluation=evaluation,
                baseline={
                    "code": baseline["generation"]["code"],
                    "evaluation": baseline["evaluation"],
                },
                repair_attempted=attempted,
            )
            if attempted:
                self._archive_record(record, result)
            else:
                self._attach_existing_logs(record)
                self.write_result(
                    self.run.output_dir / harness.name / "results.jsonl",
                    record,
                )
            records_by_key[("self-repair", case.instance_id)] = record
            repair_records[case.instance_id] = record

        generation_records = [
            records_by_key[("code-generation", case.instance_id)] for case in cases
        ]
        ordered_repair_records = [
            records_by_key[("self-repair", case.instance_id)] for case in cases
        ]
        self._log_score(
            harness=harness,
            scenario="self-repair",
            evaluations={
                case.instance_id: repair_records[case.instance_id]["evaluation"] for case in cases
            },
        )
        return {
            "harness": harness.name,
            "version": harness.version,
            "recorded_cases": len(cases),
            "code_generation_passed": sum(
                bool(record["evaluation"]["passed"]) for record in generation_records
            ),
            "self_repair_passed": sum(
                bool(record["evaluation"]["passed"]) for record in ordered_repair_records
            ),
            "repaired": sum(bool(record.get("repaired")) for record in ordered_repair_records),
            "results": str(self.run.output_dir / harness.name / "results.jsonl"),
        }

    def _log_score(
        self,
        *,
        harness: Harness,
        scenario: str,
        evaluations: dict[str, dict[str, Any]],
    ) -> None:
        total = len(evaluations)
        passed = sum(bool(evaluation.get("passed")) for evaluation in evaluations.values())
        percentage = (100.0 * passed / total) if total else 0.0
        info(
            "[%s/%s] %s score: %d/%d (%.2f%%)",
            self.name,
            harness.name,
            scenario,
            passed,
            total,
            percentage,
        )

    def _invoke(
        self,
        harness: Harness,
        *,
        case: official.LiveCodeBenchCase,
        scenario: str,
        prompt: str,
    ) -> HarnessResult:
        workspace_parent = self.run.workspace_root / "cases" / self.name
        workspace_parent.mkdir(parents=True, exist_ok=True)
        safe_id = official._safe_id(case.instance_id)
        with tempfile.TemporaryDirectory(
            prefix=f".{scenario}-",
            dir=workspace_parent,
        ) as temporary_workspace:
            return harness.run(
                HarnessRequest(
                    prompt=prompt,
                    workspace=Path(temporary_workspace),
                    session_dir=(
                        self.run.output_dir / harness.name / "sessions" / scenario / safe_id
                    ),
                    project_root=self.project_root,
                    config_path=self.config_path,
                )
            )

    def _record(
        self,
        *,
        harness: Harness,
        case: official.LiveCodeBenchCase,
        scenario: str,
        prompt: str | None,
        code: str,
        result: HarnessResult,
        evaluation: dict[str, Any],
        baseline: dict[str, Any] | None = None,
        repair_attempted: bool | None = None,
    ) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "benchmark": self.name,
            "release_version": self.release_version,
            "scenario": scenario,
            "instance_id": case.instance_id,
            "case": official._case_metadata(case),
            "harness": harness.name,
            "harness_version": harness.version,
            "prompt": prompt,
            "agent": result.to_record(),
            "generation": {"code": code},
            "evaluation": evaluation,
        }
        if baseline is not None:
            record["baseline"] = baseline
            record["repair_attempted"] = bool(repair_attempted)
            record["repaired"] = bool(
                repair_attempted
                and not baseline["evaluation"]["oracle_passed"]
                and evaluation["oracle_passed"]
            )
        return record

    def _archive_record(
        self,
        record: dict[str, Any],
        result: HarnessResult,
    ) -> None:
        harness = str(record["harness"])
        scenario = str(record["scenario"])
        safe_id = official._safe_id(str(record["instance_id"]))
        log_root = self.run.output_dir / harness / "logs" / scenario
        log_root.mkdir(parents=True, exist_ok=True)
        stdout_path = log_root / f"{safe_id}.stdout.log"
        stderr_path = log_root / f"{safe_id}.stderr.log"
        stdout_path.write_text(result.process.stdout, encoding="utf-8")
        stderr_path.write_text(result.process.stderr, encoding="utf-8")
        record["agent"]["stdout_path"] = str(stdout_path.relative_to(self.run.output_dir))
        record["agent"]["stderr_path"] = str(stderr_path.relative_to(self.run.output_dir))
        self.write_result(
            self.run.output_dir / harness / "results.jsonl",
            record,
        )

    def _attach_existing_logs(self, record: dict[str, Any]) -> None:
        harness = str(record["harness"])
        safe_id = official._safe_id(str(record["instance_id"]))
        log_root = self.run.output_dir / harness / "logs" / "code-generation"
        stdout_path = log_root / f"{safe_id}.stdout.log"
        stderr_path = log_root / f"{safe_id}.stderr.log"
        if stdout_path.is_file():
            record["agent"]["stdout_path"] = str(stdout_path.relative_to(self.run.output_dir))
        if stderr_path.is_file():
            record["agent"]["stderr_path"] = str(stderr_path.relative_to(self.run.output_dir))
