"""Build-plane profile CLI operations."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from aeloon_core.config import Config
from aeloon_core.orchestrator import AeloonCoreOrchestrator
from aeloon_core.profile_artifacts import ProfileArtifactError
from aeloon_core.profiles import canonical_profile_hash, parse_profile


async def run_profile(
    args: Namespace,
    *,
    path_overrides: Callable[..., Config],
) -> None:
    """Execute one explicit profile build-plane operation."""
    try:
        if args.profile_command == "validate":
            parsed = parse_profile(args.source.read_text(encoding="utf-8"))
            _print_json(
                {
                    "valid": True,
                    "profile_id": parsed.id,
                    "revision": parsed.revision,
                    "canonical_profile_hash": canonical_profile_hash(parsed),
                    "default_agent": parsed.default_agent,
                    "agents": [agent.id for agent in parsed.agents],
                }
            )
            return

        config = path_overrides(
            args.config,
            workspace=getattr(args, "workspace", None),
            data_dir=getattr(args, "data_dir", None),
        )
        orchestrator = AeloonCoreOrchestrator(config)
        store = orchestrator.profile_store
        if args.profile_command == "compile":
            result = await store.compile(
                args.source,
                compiler=args.compiler,
                provider=orchestrator.provider if args.compiler == "llm" else None,
                model=(args.model or config.agents.defaults.model)
                if args.compiler == "llm"
                else None,
            )
        elif args.profile_command == "inspect":
            result = store.inspect(args.artifact_id)
        elif args.profile_command == "approve":
            result = store.approve(args.artifact_id, approved_by=args.approved_by)
        elif args.profile_command == "activate":
            result = store.activate(args.artifact_id)
        elif args.profile_command == "status":
            profile_id = args.profile_id or config.agents.defaults.profile_id
            result = (
                store.status(profile_id)
                if profile_id is not None
                else {
                    "profiles": [
                        store.status(path.stem)
                        for path in sorted(store.active_dir.glob("*.json"))
                    ]
                }
            )
        elif args.profile_command == "rollback":
            result = store.rollback(args.artifact_id)
        else:
            raise SystemExit(f"Unknown profile command: {args.profile_command}")
        _print_json(result)
    except (OSError, ValueError, ProfileArtifactError) as exc:
        raise SystemExit(f"Profile operation failed: {exc}") from exc


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
