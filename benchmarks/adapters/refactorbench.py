"""Thin integration of the official RefactorBench mappings and AST tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.adapters.base import BenchmarkAdapter
from benchmarks.harness.base import Harness, HarnessRequest
from benchmarks.progress import ProgressBar, info
from benchmarks.refactorbench import runner as official


class RefactorBenchAdapter(BenchmarkAdapter):
    name = "refactorbench"
    repository_url = "https://github.com/microsoft/RefactorBench.git"
    instruction_set = "base"

    def install_dependencies(self) -> None:
        # RefactorBench's published AST tests use Python's standard library.
        info("[%s] No additional evaluator dependencies required", self.name)
        return

    def load_cases(self) -> list[official.RefactorCase]:
        return self.selected(official.load_cases(self.run.source_dir, self.instruction_set))

    def evaluate(
        self,
        *,
        case: official.RefactorCase,
        workspace: Path,
    ) -> official.ProcessOutcome:
        return official._run_official_test(
            test_path=case.test_path,
            workspace=workspace,
            timeout=60.0,
        )

    def execute(self, harnesses: list[Harness]) -> dict[str, Any]:
        info("[%s] Loading official %s cases", self.name, self.instruction_set)
        cases = self.load_cases()
        if not cases:
            raise RuntimeError("RefactorBench contains no selected cases.")
        info("[%s] Loaded %d cases", self.name, len(cases))

        output_dir = self.run.output_dir
        manifest_path = output_dir / "manifest.json"
        manifest = {
            **self.manifest(harnesses, status="running"),
            "created_at": datetime.now(UTC).isoformat(),
            "instruction_set": self.instruction_set,
            "cases": [case.instance_id for case in cases],
        }
        self.write_json(manifest_path, manifest)

        source_revision = self.source_revision() or "unknown"
        cache = official.WorkspaceCache(
            self.run.workspace_root / "cases" / self.name / source_revision[:12],
            refactorbench_root=self.run.source_dir,
        )
        records_by_harness: dict[str, list[dict[str, Any]]] = {
            harness.name: [] for harness in harnesses
        }
        try:
            with ProgressBar(
                self.name,
                total=len(cases) * len(harnesses),
            ) as progress:
                if self.workers == 1:
                    for harness in harnesses:
                        records_by_harness[harness.name] = self._run_harness_cases(
                            harness=harness,
                            cases=cases,
                            cache=cache,
                            progress=progress,
                        )
                else:
                    records_by_harness = self._run_parallel(
                        harnesses=harnesses,
                        cases=cases,
                        cache=cache,
                        progress=progress,
                    )
        except BaseException:
            manifest["status"] = "interrupted"
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            self.write_json(manifest_path, manifest)
            raise

        harness_summaries = {
            harness.name: self._summarize(
                harness=harness,
                records=records_by_harness[harness.name],
            )
            for harness in harnesses
        }
        passed = sum(item["passed"] for item in harness_summaries.values())
        recorded = sum(item["recorded_cases"] for item in harness_summaries.values())
        summary = {
            "schema_version": 1,
            "benchmark": self.name,
            "run_id": self.run.run_id,
            "workers": self.workers,
            "selected_cases": len(cases),
            "recorded_cases": recorded,
            "passed": passed,
            "pass_rate": passed / recorded if recorded else None,
            "archive": str(output_dir),
            "harnesses": harness_summaries,
        }
        self.write_json(output_dir / "summary.json", summary)
        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        self.write_json(manifest_path, manifest)
        return summary

    def _run_harness_cases(
        self,
        *,
        harness: Harness,
        cases: list[official.RefactorCase],
        cache: official.WorkspaceCache,
        progress: ProgressBar,
    ) -> list[dict[str, Any]]:
        info(
            "[%s/%s] Starting %d cases",
            self.name,
            harness.name,
            len(cases),
        )
        records = [
            self._run_case(
                case=case,
                harness=harness,
                cache=cache,
                progress=progress,
            )
            for case in cases
        ]
        info("[%s/%s] Completed %d cases", self.name, harness.name, len(cases))
        return records

    def _run_parallel(
        self,
        *,
        harnesses: list[Harness],
        cases: list[official.RefactorCase],
        cache: official.WorkspaceCache,
        progress: ProgressBar,
    ) -> dict[str, list[dict[str, Any]]]:
        cases_by_repository: dict[str, list[official.RefactorCase]] = {}
        for case in cases:
            cases_by_repository.setdefault(case.repository, []).append(case)

        # WorkspaceCache owns one writable checkout per repository. Initialize
        # them serially, then assign each repository to exactly one worker.
        for repository_cases in cases_by_repository.values():
            cache.prepare(repository_cases[0])

        max_workers = min(self.workers, len(cases_by_repository))
        info(
            "[%s] Parallel execution enabled: workers=%d repository_lanes=%d",
            self.name,
            max_workers,
            len(cases_by_repository),
        )
        records_by_harness = {harness.name: [] for harness in harnesses}
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="refactorbench",
        ) as executor:
            futures = {
                executor.submit(
                    self._run_repository_lane,
                    harnesses=harnesses,
                    cases=repository_cases,
                    cache=cache,
                    progress=progress,
                ): repository
                for repository, repository_cases in cases_by_repository.items()
            }
            try:
                for future in as_completed(futures):
                    lane_records = future.result()
                    for harness_name, records in lane_records.items():
                        records_by_harness[harness_name].extend(records)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

        case_order = {case.instance_id: index for index, case in enumerate(cases)}
        for records in records_by_harness.values():
            records.sort(key=lambda record: case_order[str(record["instance_id"])])
        return records_by_harness

    def _run_repository_lane(
        self,
        *,
        harnesses: list[Harness],
        cases: list[official.RefactorCase],
        cache: official.WorkspaceCache,
        progress: ProgressBar,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            harness.name: self._run_harness_cases(
                harness=harness,
                cases=cases,
                cache=cache,
                progress=progress,
            )
            for harness in harnesses
        }

    def _run_case(
        self,
        *,
        case: official.RefactorCase,
        harness: Harness,
        cache: official.WorkspaceCache,
        progress: ProgressBar,
    ) -> dict[str, Any]:
        progress.set_detail(f"{harness.name} running {case.instance_id}")
        info("[%s/%s] Running case %s", self.name, harness.name, case.instance_id)
        workspace, baseline = cache.prepare(case)
        prompt = case.prompt_path.read_text(encoding="utf-8")
        safe_id = case.instance_id.replace("/", "__")
        harness_root = self.run.output_dir / harness.name
        session_dir = harness_root / "sessions" / safe_id
        result = harness.run(
            HarnessRequest(
                prompt=prompt,
                workspace=workspace,
                session_dir=session_dir,
                project_root=self.project_root,
                config_path=self.config_path,
            )
        )

        log_root = harness_root / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        stdout_path = log_root / f"{safe_id}.stdout.log"
        stderr_path = log_root / f"{safe_id}.stderr.log"
        stdout_path.write_text(result.process.stdout, encoding="utf-8")
        stderr_path.write_text(result.process.stderr, encoding="utf-8")
        agent = result.to_record()
        agent["stdout_path"] = str(stdout_path.relative_to(self.run.output_dir))
        agent["stderr_path"] = str(stderr_path.relative_to(self.run.output_dir))

        patch, changed_files, patch_error = official._capture_patch(workspace, baseline)
        patch_path = harness_root / "patches" / f"{safe_id}.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch, encoding="utf-8")

        outcome = self.evaluate(case=case, workspace=workspace)
        oracle_passed = outcome.returncode == 0 and not outcome.timed_out
        passed = result.status == "completed" and oracle_passed
        record = {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "benchmark": self.name,
            "harness": harness.name,
            "harness_version": harness.version,
            "instruction_set": self.instruction_set,
            "instance_id": case.instance_id,
            "repository": case.repository,
            "prompt": prompt,
            "workspace": str(workspace),
            "baseline_commit": baseline,
            "agent": agent,
            "evaluation": {
                "passed": passed,
                "oracle_passed": oracle_passed,
                "returncode": outcome.returncode,
                "timed_out": outcome.timed_out,
                "duration_ms": outcome.duration_ms,
                "stdout": official._bounded(outcome.stdout),
                "stderr": official._bounded(outcome.stderr),
            },
            "changed_files": changed_files,
            "patch_path": str(patch_path.relative_to(self.run.output_dir)),
            "patch_error": patch_error,
            "false_completed": result.status == "completed" and not oracle_passed,
        }
        self.write_result(harness_root / "results.jsonl", record)
        verdict = "PASS" if passed else "FAIL"
        info(
            "[%s/%s] Case %s: %s",
            self.name,
            harness.name,
            case.instance_id,
            verdict,
        )
        progress.advance(detail=f"{harness.name} {case.instance_id} {verdict}")
        return record

    def _summarize(
        self,
        *,
        harness: Harness,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        passed = sum(bool(record["evaluation"]["passed"]) for record in records)
        input_tokens = _usage_total(records, "input_tokens", "inputTokens")
        output_tokens = _usage_total(records, "output_tokens", "outputTokens")
        return {
            "harness": harness.name,
            "version": harness.version,
            "recorded_cases": len(records),
            "passed": passed,
            "pass_rate": passed / len(records) if records else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "results": str(self.run.output_dir / harness.name / "results.jsonl"),
        }


def _usage_total(
    records: list[dict[str, Any]],
    *keys: str,
) -> int | float | None:
    values: list[int | float] = []
    for record in records:
        usage = record.get("agent", {}).get("usage", {})
        if not isinstance(usage, dict):
            continue
        value = next(
            (
                usage[key]
                for key in keys
                if isinstance(usage.get(key), int | float) and not isinstance(usage.get(key), bool)
            ),
            None,
        )
        if value is not None:
            values.append(value)
    return sum(values) if values else None
