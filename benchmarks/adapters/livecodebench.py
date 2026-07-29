"""Thin integration of LiveCodeBench v6 loading and official evaluation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from benchmarks.adapters.base import BenchmarkAdapter, run_checked
from benchmarks.harness.base import (
    Harness,
    HarnessInvocation,
    HarnessRequest,
    HarnessResult,
    ProcessOutcome,
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
    supports_resume = True
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
        with self._run_lock():
            return self._execute_locked(harnesses)

    @contextmanager
    def _run_lock(self) -> Iterator[None]:
        self.run.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.run.output_dir / ".run.lock"
        with lock_path.open("a+", encoding="utf-8") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"Benchmark run is already active: {self.run.run_id}"
                ) from None
            stream.seek(0)
            stream.truncate()
            stream.write(f"{os.getpid()}\n")
            stream.flush()
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _execute_locked(self, harnesses: list[Harness]) -> dict[str, Any]:
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
        now = datetime.now(UTC).isoformat()
        if not self.resuming:
            return {
                **self.manifest(harnesses, status="running"),
                "created_at": now,
                "release_version": self.release_version,
                "scenarios": list(self.scenarios),
                "cases": [case.instance_id for case in cases],
            }

        manifest_path = self.run.output_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise RuntimeError(f"Resume manifest does not exist: {manifest_path}") from None
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Resume manifest is invalid: {manifest_path}: {exc}") from None
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Resume manifest is not an object: {manifest_path}")

        expected_harnesses = [harness.name for harness in harnesses]
        recorded_harnesses = [
            item.get("id")
            for item in manifest.get("harnesses", [])
            if isinstance(item, dict)
        ]
        checks = {
            "benchmark": (manifest.get("benchmark"), self.name),
            "run_id": (manifest.get("run_id"), self.run.run_id),
            "release_version": (manifest.get("release_version"), self.release_version),
            "scenarios": (manifest.get("scenarios"), list(self.scenarios)),
            "cases": (
                manifest.get("cases"),
                [case.instance_id for case in cases],
            ),
            "harnesses": (recorded_harnesses, expected_harnesses),
            "source revision": (
                manifest.get("source", {}).get("revision")
                if isinstance(manifest.get("source"), dict)
                else None,
                self.source_revision(),
            ),
        }
        mismatches = [
            label for label, (recorded, expected) in checks.items() if recorded != expected
        ]
        if mismatches:
            raise RuntimeError(
                "Cannot resume benchmark run because the manifest changed: "
                + ", ".join(mismatches)
            )

        previous_finished_at = manifest.pop("finished_at", None)
        resume_events = manifest.setdefault("resume_events", [])
        if not isinstance(resume_events, list):
            raise RuntimeError(f"Resume manifest has invalid resume_events: {manifest_path}")
        resume_events.append(
            {
                "resumed_at": now,
                "previous_status": manifest.get("status"),
                "previous_finished_at": previous_finished_at,
                "workers": self.workers,
                "harnesses": [
                    {"id": harness.name, "version": harness.version}
                    for harness in harnesses
                ],
            }
        )
        manifest["status"] = "running"
        return manifest

    def _run_harness(
        self,
        harness: Harness,
        cases: list[official.LiveCodeBenchCase],
    ) -> dict[str, Any]:
        records_by_key = self._load_records(harness, cases)
        baseline_records = {
            instance_id: record
            for (scenario, instance_id), record in records_by_key.items()
            if scenario == "code-generation"
        }
        pending_baselines = [
            case for case in cases if case.instance_id not in baseline_records
        ]
        if baseline_records:
            info(
                "[%s/%s] Reusing %d code-generation records; pending=%d",
                self.name,
                harness.name,
                len(baseline_records),
                len(pending_baselines),
            )

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
            instance_id: generation
            for instance_id, generation in generated_baselines
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

        repair_records = {
            instance_id: record
            for (scenario, instance_id), record in records_by_key.items()
            if scenario == "self-repair"
        }
        self._validate_repair_records(
            repair_records=repair_records,
            baseline_records=baseline_records,
            harness=harness,
        )
        pending_repairs = [
            case
            for case in cases
            if case.instance_id not in repair_records
            and not baseline_records[case.instance_id]["evaluation"]["oracle_passed"]
        ]
        if repair_records:
            info(
                "[%s/%s] Reusing %d self-repair records; pending attempts=%d",
                self.name,
                harness.name,
                len(repair_records),
                len(pending_repairs),
            )

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
            instance_id: generation
            for instance_id, generation in generated_repairs
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
            if case.instance_id in repair_records:
                continue
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
                result = self._result_from_record(baseline)
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
            "repaired": sum(
                bool(record.get("repaired")) for record in ordered_repair_records
            ),
            "results": str(self.run.output_dir / harness.name / "results.jsonl"),
        }

    def _validate_repair_records(
        self,
        *,
        repair_records: dict[str, dict[str, Any]],
        baseline_records: dict[str, dict[str, Any]],
        harness: Harness,
    ) -> None:
        for instance_id, repair in repair_records.items():
            baseline = baseline_records.get(instance_id)
            snapshot = repair.get("baseline")
            if (
                baseline is None
                or not isinstance(snapshot, dict)
                or snapshot.get("code") != baseline["generation"].get("code")
                or snapshot.get("evaluation") != baseline.get("evaluation")
            ):
                raise RuntimeError(
                    "Cannot resume inconsistent self-repair record: "
                    f"{harness.name}/{instance_id}"
                )

    def _load_records(
        self,
        harness: Harness,
        cases: list[official.LiveCodeBenchCase],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        selected_ids = {case.instance_id for case in cases}
        results_path = self.run.output_dir / harness.name / "results.jsonl"
        records: dict[tuple[str, str], dict[str, Any]] = {}
        if results_path.is_file():
            for line_number, line in enumerate(
                results_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid benchmark result at {results_path}:{line_number}: {exc}"
                    ) from None
                key = self._record_key(
                    record,
                    harness=harness,
                    selected_ids=selected_ids,
                    source=f"{results_path}:{line_number}",
                )
                records[key] = record

        if not self.resuming:
            return records

        recovery_path = (
            self.run.output_dir
            / harness.name
            / "recovered-code-generation.json"
        )
        if not recovery_path.is_file():
            return records
        try:
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid recovery artifact at {recovery_path}: {exc}") from None
        recovered_records = recovery.get("records") if isinstance(recovery, dict) else None
        if not isinstance(recovered_records, list):
            raise RuntimeError(f"Recovery artifact contains no records: {recovery_path}")

        imported = 0
        for index, record in enumerate(recovered_records, start=1):
            key = self._record_key(
                record,
                harness=harness,
                selected_ids=selected_ids,
                source=f"{recovery_path}:record {index}",
            )
            if key in records:
                continue
            self.write_result(results_path, record)
            records[key] = record
            imported += 1
        if imported:
            info(
                "[%s/%s] Imported %d recovered records into the resume ledger",
                self.name,
                harness.name,
                imported,
            )
        return records

    def _record_key(
        self,
        record: Any,
        *,
        harness: Harness,
        selected_ids: set[str],
        source: str,
    ) -> tuple[str, str]:
        if not isinstance(record, dict):
            raise RuntimeError(f"Benchmark result is not an object: {source}")
        scenario = record.get("scenario")
        instance_id = record.get("instance_id")
        if (
            record.get("benchmark") != self.name
            or record.get("release_version") != self.release_version
            or record.get("harness") != harness.name
            or scenario not in self.scenarios
            or not isinstance(instance_id, str)
            or instance_id not in selected_ids
            or not isinstance(record.get("generation"), dict)
            or not isinstance(record.get("evaluation"), dict)
            or not isinstance(record.get("agent"), dict)
        ):
            raise RuntimeError(f"Benchmark result is incompatible with this run: {source}")
        return scenario, instance_id

    def _result_from_record(self, record: dict[str, Any]) -> HarnessResult:
        agent = record.get("agent")
        if not isinstance(agent, dict):
            raise RuntimeError(
                "Cannot resume invalid agent record for "
                f"{record.get('scenario')}/{record.get('instance_id')}"
            )
        return self._result_from_agent(agent)

    def _result_from_agent(
        self,
        agent: dict[str, Any],
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> HarnessResult:
        command = agent.get("command")
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            command = ["resumed-benchmark-generation"]
        wall_time_ms = agent.get("wall_time_ms")
        if (
            not isinstance(wall_time_ms, int | float)
            or isinstance(wall_time_ms, bool)
        ):
            wall_time_ms = 0
        returncode = agent.get("returncode")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            returncode = None
        if not stdout and isinstance(agent.get("stdout"), str):
            stdout = agent["stdout"]
        if not stderr and isinstance(agent.get("stderr"), str):
            stderr = agent["stderr"]

        harness_name = agent.get("harness")
        status = agent.get("status")
        if not isinstance(harness_name, str) or not isinstance(status, str):
            raise RuntimeError("Cannot resume an invalid harness result.")
        version = agent.get("version")
        final_content = agent.get("final_content")
        session_id = agent.get("session_id")
        turn_id = agent.get("turn_id")
        payload_error = agent.get("payload_error")
        duration_ms = agent.get("duration_ms")
        cost_usd = agent.get("cost_usd")
        return HarnessResult(
            harness=harness_name,
            version=version if isinstance(version, str) else None,
            invocation=HarnessInvocation(
                command=command,
                cwd=self.project_root,
            ),
            process=ProcessOutcome(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=round(wall_time_ms),
                timed_out=bool(agent.get("timed_out")),
            ),
            status=status,
            final_content=(
                final_content if isinstance(final_content, str) else None
            ),
            session_id=session_id if isinstance(session_id, str) else None,
            turn_id=turn_id if isinstance(turn_id, str) else None,
            usage=agent["usage"] if isinstance(agent.get("usage"), dict) else {},
            models=agent["models"] if isinstance(agent.get("models"), dict) else {},
            tools_used=(
                agent["tools_used"]
                if isinstance(agent.get("tools_used"), list)
                else []
            ),
            transitions=(
                agent["transitions"]
                if isinstance(agent.get("transitions"), list)
                else []
            ),
            cost_usd=(
                cost_usd
                if isinstance(cost_usd, int | float)
                and not isinstance(cost_usd, bool)
                else None
            ),
            payload_error=(
                payload_error if isinstance(payload_error, str) else None
            ),
            duration_ms=(
                duration_ms
                if isinstance(duration_ms, int | float)
                and not isinstance(duration_ms, bool)
                else None
            ),
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
