"""Agentized RepoQA integration with repository tools and exact location grading."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.adapters.base import BenchmarkAdapter
from benchmarks.harness.base import Harness, HarnessRequest
from benchmarks.progress import ProgressBar, info
from benchmarks.repoqa import runner as official


class RepoQAAdapter(BenchmarkAdapter):
    """Locate official RepoQA needle functions by searching local repositories."""

    name = "repoqa"
    repository_url = "https://github.com/evalplus/repoqa.git"

    @property
    def dataset_path(self) -> Path:
        return self.run.workspace_root / "datasets" / self.name / official.DATASET_FILENAME

    def prepare_dataset(self) -> None:
        super().prepare_dataset()
        if self.dataset_path.is_file():
            info("[%s] Reusing pinned dataset: %s", self.name, self.dataset_path)
        else:
            info(
                "[%s] Downloading official dataset version %s",
                self.name,
                official.DATASET_VERSION,
            )
        official.download_dataset(self.dataset_path)

    def install_dependencies(self) -> None:
        # The agentized task grades exact official path/symbol pairs, so it
        # deliberately avoids RepoQA's model backends, tokenizers, and BLEU stack.
        info("[%s] No additional evaluator dependencies required", self.name)

    def load_cases(self) -> list[official.RepoQACase]:
        return self.selected(official.load_cases(self.dataset_path))

    def evaluate(
        self,
        *,
        case: official.RepoQACase,
        final_content: str | None,
        changed_files: list[str],
    ) -> dict[str, Any]:
        return official.evaluate_answer(case, final_content, changed_files)

    def execute(self, harnesses: list[Harness]) -> dict[str, Any]:
        info(
            "[%s] Loading official dataset version %s",
            self.name,
            official.DATASET_VERSION,
        )
        cases = self.load_cases()
        if not cases:
            raise RuntimeError("RepoQA contains no selected cases.")
        info("[%s] Loaded %d cases", self.name, len(cases))

        output_dir = self.run.output_dir
        manifest_path = output_dir / "manifest.json"
        manifest = {
            **self.manifest(harnesses, status="running"),
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": {
                "version": official.DATASET_VERSION,
                "url": official.DATASET_URL,
                "sha256": official.DATASET_SHA256,
                "path": str(self.dataset_path),
            },
            "task": "search-needle-function-agent",
            "grading": {
                "target": "exact-path-and-symbol",
                "requires_clean_worktree": True,
                "result_prefix": official.RESULT_PREFIX,
            },
            "languages": sorted({case.language for case in cases}),
            "cases": [case.instance_id for case in cases],
        }
        self.write_json(manifest_path, manifest)

        cache = official.WorkspaceCache(
            self.run.workspace_root / "cases" / self.name / official.DATASET_VERSION
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
        recorded = sum(item["recorded_cases"] for item in harness_summaries.values())
        passed = sum(item["passed"] for item in harness_summaries.values())
        summary = {
            "schema_version": 1,
            "benchmark": self.name,
            "task": "search-needle-function-agent",
            "dataset_version": official.DATASET_VERSION,
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
        cases: list[official.RepoQACase],
        cache: official.WorkspaceCache,
        progress: ProgressBar,
    ) -> list[dict[str, Any]]:
        info("[%s/%s] Starting %d cases", self.name, harness.name, len(cases))
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
        cases: list[official.RepoQACase],
        cache: official.WorkspaceCache,
        progress: ProgressBar,
    ) -> dict[str, list[dict[str, Any]]]:
        cases_by_repository: dict[tuple[str, str], list[official.RepoQACase]] = {}
        for case in cases:
            cases_by_repository.setdefault(
                (case.language, case.repository),
                [],
            ).append(case)

        # Each lane owns all tasks and harness workspaces for one repository.
        # Materializing the immutable sources first avoids setup races.
        for repository_cases in cases_by_repository.values():
            cache.materialize(repository_cases[0])

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
            thread_name_prefix="repoqa",
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
        cases: list[official.RepoQACase],
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
        case: official.RepoQACase,
        harness: Harness,
        cache: official.WorkspaceCache,
        progress: ProgressBar,
    ) -> dict[str, Any]:
        progress.set_detail(f"{harness.name} running {case.instance_id}")
        info("[%s/%s] Running case %s", self.name, harness.name, case.instance_id)
        workspace, baseline = cache.prepare(case, harness.name)
        prompt = official.build_prompt(case)
        safe_id = hashlib.sha256(case.instance_id.encode()).hexdigest()[:20]
        harness_root = self.run.output_dir / harness.name
        result = harness.run(
            HarnessRequest(
                prompt=prompt,
                workspace=workspace,
                session_dir=harness_root / "sessions" / safe_id,
                project_root=self.project_root,
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

        changed_files = official.changed_files(workspace)
        evaluation = self.evaluate(
            case=case,
            final_content=result.final_content,
            changed_files=changed_files,
        )
        passed = (
            result.status == "completed"
            and bool(evaluation["matched_target"])
            and bool(evaluation["clean_worktree"])
        )
        evaluation.update(
            {
                "passed": passed,
                "agent_completed": result.status == "completed",
            }
        )
        record = {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "benchmark": self.name,
            "task": "search-needle-function-agent",
            "dataset_version": official.DATASET_VERSION,
            "harness": harness.name,
            "harness_version": harness.version,
            "instance_id": case.instance_id,
            "language": case.language,
            "repository": case.repository,
            "repository_commit": case.commit_sha,
            "prompt": prompt,
            "workspace": str(workspace),
            "baseline_commit": baseline,
            "agent": agent,
            "target": {
                "path": case.target_path,
                "symbol": case.target_symbol,
            },
            "evaluation": evaluation,
            "changed_files": changed_files,
            "false_completed": result.status == "completed" and not passed,
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
        matched = sum(bool(record["evaluation"]["matched_target"]) for record in records)
        clean = sum(bool(record["evaluation"]["clean_worktree"]) for record in records)
        parse_failures = sum(record["evaluation"]["parse_error"] is not None for record in records)
        language_records: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            language_records.setdefault(str(record["language"]), []).append(record)
        languages = {
            language: {
                "recorded_cases": len(items),
                "passed": sum(bool(item["evaluation"]["passed"]) for item in items),
                "pass_rate": (
                    sum(bool(item["evaluation"]["passed"]) for item in items) / len(items)
                    if items
                    else None
                ),
            }
            for language, items in sorted(language_records.items())
        }
        return {
            "harness": harness.name,
            "version": harness.version,
            "recorded_cases": len(records),
            "passed": passed,
            "pass_rate": passed / len(records) if records else None,
            "target_matches": matched,
            "clean_worktrees": clean,
            "parse_failures": parse_failures,
            "input_tokens": _usage_total(records, "input_tokens", "inputTokens"),
            "output_tokens": _usage_total(records, "output_tokens", "outputTokens"),
            "languages": languages,
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
