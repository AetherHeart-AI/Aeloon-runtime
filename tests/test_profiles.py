from __future__ import annotations

import hashlib
from textwrap import dedent

import pytest
from pydantic import ValidationError

from aeloon_core.profiles import (
    MAX_COMPILED_SOURCE_CHARS,
    MAX_PROFILE_SOURCE_CHARS,
    ProfileValidationError,
    RuntimeProfileSpec,
    canonical_profile_bytes,
    canonical_profile_hash,
    emit_compiled_profile,
    parse_compiled_profile,
    parse_profile,
)

VALID_PROFILE = dedent(
    """\
    ---
    schema_version: 1
    id: coding-team
    revision: 1
    description: Coding and review team
    default_agent: implementer
    max_handoffs: 8
    agents:
      - id: planner
        description: Analyze requirements
        tools: [read, glob, grep]
      - id: implementer
        description: Implement and verify changes
        tools: [read, write, edit, exec]
    ---

    ## Shared
    Keep changes small and verified.

    ## Master
    Choose the role best suited to the next step.

    ## Agent: planner
    Inspect the repository and make a concrete plan.

    ## Agent: implementer
    Implement the plan and run focused tests.
    """
)


def test_parse_profile_returns_frozen_exact_source_model() -> None:
    profile = parse_profile(VALID_PROFILE)

    assert profile.schema_version == 1
    assert profile.id == "coding-team"
    assert profile.default_agent == "implementer"
    assert profile.max_handoffs == 8
    assert profile.shared_prompt == "Keep changes small and verified."
    assert profile.master_prompt == "Choose the role best suited to the next step."
    assert [agent.id for agent in profile.agents] == ["planner", "implementer"]
    assert profile.agent("planner").tools == ("read", "glob", "grep")
    assert profile.agent_map["implementer"].prompt == (
        "Implement the plan and run focused tests."
    )

    with pytest.raises(ValidationError, match="frozen"):
        profile.revision = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.agent_map["reviewer"] = profile.agents[0]  # type: ignore[index]


def test_v1_profile_keeps_preexisting_delegate_tasks_role_identifier_valid() -> None:
    profile = parse_profile(VALID_PROFILE.replace("planner", "delegate_tasks"))

    assert profile.agent("delegate_tasks").id == "delegate_tasks"


@pytest.mark.parametrize(
    "bad_source, expected",
    [
        (VALID_PROFILE.replace("revision: 1\n", "revision: 1\nrevision: 2\n"), "duplicate"),
        (
            VALID_PROFILE.replace(
                "description: Analyze requirements\n",
                "description: Analyze requirements\n    description: Duplicate\n",
            ),
            "duplicate",
        ),
        (
            VALID_PROFILE.replace(
                "description: Coding and review team",
                "description: !!python/object/apply:os.system ['touch /tmp/profile-owned']",
            ),
            "invalid profile YAML",
        ),
        (
            VALID_PROFILE.replace("revision: 1\n", "revision: 1\nunknown: value\n"),
            "extra_forbidden",
        ),
        (
            VALID_PROFILE.replace(
                "description: Analyze requirements\n",
                "description: Analyze requirements\n    unknown: value\n",
            ),
            "extra_forbidden",
        ),
    ],
)
def test_profile_yaml_rejects_duplicates_custom_tags_and_unknown_fields(
    bad_source: str,
    expected: str,
) -> None:
    with pytest.raises(ProfileValidationError, match=expected):
        parse_profile(bad_source)


def test_profile_yaml_rejects_aliases_and_oversized_sources() -> None:
    aliased = VALID_PROFILE.replace(
        "description: Coding and review team",
        "description: &shared_description Coding and review team",
    ).replace(
        "description: Analyze requirements",
        "description: *shared_description",
    )
    with pytest.raises(ProfileValidationError, match="aliases are not allowed"):
        parse_profile(aliased)

    with pytest.raises(ProfileValidationError, match="profile source exceeds"):
        parse_profile(VALID_PROFILE + ("x" * MAX_PROFILE_SOURCE_CHARS))


@pytest.mark.parametrize(
    "bad_id",
    [
        "Coding-Team",
        "1coding-team",
        "coding.team",
        "a" * 65,
    ],
)
def test_profile_identifier_must_match_contract(bad_id: str) -> None:
    with pytest.raises(ProfileValidationError, match="string_pattern_mismatch"):
        parse_profile(VALID_PROFILE.replace("id: coding-team", f"id: {bad_id}", 1))


@pytest.mark.parametrize(
    "reserved_id",
    [
        "master",
        "worker",
        "tool",
        "temporary_guard",
        "control",
        "done",
        "handoff_agent",
        "complete_task",
    ],
)
def test_agent_identifier_rejects_reserved_runtime_and_control_names(
    reserved_id: str,
) -> None:
    source = VALID_PROFILE.replace("default_agent: implementer", f"default_agent: {reserved_id}")
    source = source.replace("id: implementer", f"id: {reserved_id}")
    source = source.replace("Agent: implementer", f"Agent: {reserved_id}")

    with pytest.raises(ProfileValidationError, match="reserved runtime/control name"):
        parse_profile(source)


def test_profile_requires_one_to_sixteen_unique_agents() -> None:
    no_agents = VALID_PROFILE.replace(
        dedent(
            """\
            agents:
              - id: planner
                description: Analyze requirements
                tools: [read, glob, grep]
              - id: implementer
                description: Implement and verify changes
                tools: [read, write, edit, exec]
            """
        ),
        "agents: []\n",
    )
    with pytest.raises(ProfileValidationError, match="too_short"):
        parse_profile(no_agents)

    declarations = "\n".join(
        f"  - id: role-{index}\n    description: Role {index}\n    tools: []"
        for index in range(17)
    )
    sections = "\n\n".join(
        f"## Agent: role-{index}\nInstructions for role {index}." for index in range(17)
    )
    too_many = f"""---
schema_version: 1
id: many-agents
revision: 1
description: Too many agents
default_agent: role-0
max_handoffs: 8
agents:
{declarations}
---

## Shared
Shared instructions.

## Master
Master instructions.

{sections}
"""
    with pytest.raises(ProfileValidationError, match="too_long"):
        parse_profile(too_many)

    duplicate = VALID_PROFILE.replace("id: implementer", "id: planner")
    with pytest.raises(ProfileValidationError, match="agent ids must be unique"):
        parse_profile(duplicate)


def test_default_agent_must_name_a_declared_agent() -> None:
    source = VALID_PROFILE.replace("default_agent: implementer", "default_agent: reviewer")

    with pytest.raises(ProfileValidationError, match="default agent must name"):
        parse_profile(source)


def test_tools_must_be_unique_within_each_agent() -> None:
    source = VALID_PROFILE.replace(
        "tools: [read, glob, grep]",
        "tools: [read, glob, read]",
    )

    with pytest.raises(ProfileValidationError, match="tool names must be unique"):
        parse_profile(source)


@pytest.mark.parametrize(
    "bad_source, expected",
    [
        (VALID_PROFILE.replace("## Shared", "## Shared constraints"), "unknown or malformed"),
        (
            VALID_PROFILE.replace("## Shared\nKeep changes small and verified.\n\n", ""),
            "missing required",
        ),
        (VALID_PROFILE.replace("## Agent: planner", "## Agent: reviewer"), "missing agent"),
        (
            VALID_PROFILE
            + "\n## Agent: reviewer\nReview the implementation without modifying it.\n",
            "undeclared agent",
        ),
        (
            VALID_PROFILE.replace(
                "## Master\nChoose the role best suited to the next step.",
                "## Master\nChoose the role.\n\n## Master\nChoose again.",
            ),
            "duplicate Markdown section",
        ),
        ("unexpected prose\n" + VALID_PROFILE, "must begin with one YAML frontmatter"),
        (VALID_PROFILE.replace("Keep changes small and verified.", "   "), "must not be empty"),
    ],
)
def test_markdown_sections_must_match_profile_exactly(
    bad_source: str,
    expected: str,
) -> None:
    with pytest.raises(ProfileValidationError, match=expected):
        parse_profile(bad_source)


def test_markdown_rejects_content_between_frontmatter_and_first_section() -> None:
    source = VALID_PROFILE.replace("\n## Shared", "\nUnexpected content.\n\n## Shared")

    with pytest.raises(ProfileValidationError, match="content before"):
        parse_profile(source)


def test_canonical_profile_bytes_and_hash_ignore_yaml_order_and_line_endings() -> None:
    reordered = dedent(
        """\
        ---
        description: Coding and review team
        max_handoffs: 8
        revision: 1
        schema_version: 1
        default_agent: implementer
        agents:
          - tools: [read, glob, grep]
            description: Analyze requirements
            id: planner
          - tools: [read, write, edit, exec]
            id: implementer
            description: Implement and verify changes
        id: coding-team
        ---

        ## Shared
        Keep changes small and verified.

        ## Master
        Choose the role best suited to the next step.

        ## Agent: planner
        Inspect the repository and make a concrete plan.

        ## Agent: implementer
        Implement the plan and run focused tests.
        """
    ).replace("\n", "\r\n")
    first = parse_profile(VALID_PROFILE)
    second = parse_profile(reordered)

    assert first == second
    assert canonical_profile_bytes(first) == canonical_profile_bytes(second)
    assert canonical_profile_hash(first) == canonical_profile_hash(second)
    assert canonical_profile_hash(first) == hashlib.sha256(
        canonical_profile_bytes(first)
    ).hexdigest()


def test_compiled_profile_emitter_is_deterministic_and_round_trips() -> None:
    profile = parse_profile(VALID_PROFILE)

    first = emit_compiled_profile(profile)
    second = emit_compiled_profile(profile)
    runtime = parse_compiled_profile(first, artifact_id="artifact-sha256", generation=7)

    assert first == second
    assert first.startswith('"""Generated Aeloon profile artifact. Do not edit."""\n')
    assert "import " not in first
    assert runtime.profile_schema_version == 1
    assert runtime.compiled_api_version == 1
    assert runtime.profile_id == profile.id
    assert runtime.revision == profile.revision
    assert runtime.default_agent_id == profile.default_agent
    assert runtime.max_handoffs == profile.max_handoffs
    assert runtime.artifact_id == "artifact-sha256"
    assert runtime.generation == 7
    assert runtime.agent("planner").tools == ("read", "glob", "grep")
    assert runtime.agent_map["implementer"].prompt == profile.agent("implementer").prompt


def test_runtime_profile_is_frozen_forbids_extras_and_supports_provenance_copy() -> None:
    runtime = parse_compiled_profile(emit_compiled_profile(parse_profile(VALID_PROFILE)))

    with pytest.raises(ValidationError, match="frozen"):
        runtime.generation = 2  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeProfileSpec.model_validate({**runtime.model_dump(), "unknown": "value"})

    pinned = runtime.model_copy(update={"artifact_id": "artifact-1", "generation": 3})
    assert pinned.artifact_id == "artifact-1"
    assert pinned.generation == 3
    assert runtime.artifact_id is None
    assert runtime.generation == 0


def test_compiled_parser_accepts_only_an_optional_module_docstring_and_one_class() -> None:
    emitted = emit_compiled_profile(parse_profile(VALID_PROFILE))
    without_docstring = emitted.split("\n", 2)[2]

    assert parse_compiled_profile(without_docstring).profile_id == "coding-team"
    with pytest.raises(ProfileValidationError, match="only an optional docstring"):
        parse_compiled_profile("value = 1\n" + emitted)
    with pytest.raises(ProfileValidationError, match="only an optional docstring"):
        parse_compiled_profile(emitted + "\nclass Other:\n    value = 1\n")
    with pytest.raises(ProfileValidationError, match="must be named"):
        parse_compiled_profile(emitted.replace("class CompiledProfile:", "class Other:"))


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda source: "import os\n" + source, "only an optional docstring"),
        (
            lambda source: source.replace(
                "class CompiledProfile:",
                "class CompiledProfile(object):",
            ),
            "must not have bases",
        ),
        (
            lambda source: source.replace(
                "class CompiledProfile:",
                "@staticmethod\nclass CompiledProfile:",
            ),
            "must not have bases",
        ),
        (
            lambda source: source.replace(
                "class CompiledProfile:",
                "class CompiledProfile(metaclass=type):",
            ),
            "must not have bases",
        ),
        (
            lambda source: source.replace(
                "    profile_schema_version = 1",
                "    def steal(self):\n        return open('/tmp/profile-owned').read()\n"
                "    profile_schema_version = 1",
            ),
            "only plain literal assignments",
        ),
        (
            lambda source: source.replace(
                '    profile_id = "coding-team"',
                '    profile_id = __import__("os").environ["HOME"]',
            ),
            "Subscript is not allowed",
        ),
        (
            lambda source: source.replace(
                '    profile_id = "coding-team"',
                '    profile_id = os.environ["HOME"]',
            ),
            "Subscript is not allowed",
        ),
        (
            lambda source: source.replace(
                '    profile_id = "coding-team"',
                "    profile_id = os.environ",
            ),
            "Attribute is not allowed",
        ),
        (
            lambda source: source.replace(
                '    profile_id = "coding-team"',
                '    profile_id = [item for item in ("coding-team",)]',
            ),
            "ListComp is not allowed",
        ),
        (
            lambda source: source.replace(
                "    agents = (",
                "    agents = tuple(item for item in (",
            ).replace("    )\n", "    ))\n", 1),
            "Call is not allowed",
        ),
        (
            lambda source: source.replace(
                '    profile_id = "coding-team"',
                '    profile_id = (lambda: "coding-team")',
            ),
            "Lambda is not allowed",
        ),
        (
            lambda source: source.replace(
                "    revision = 1",
                "    revision: int = 1",
            ),
            "only plain literal assignments",
        ),
        (
            lambda source: source.replace(
                "    revision = 1",
                "    revision = 1\n    unknown = 1",
            ),
            "field is not allowed",
        ),
        (
            lambda source: source.replace(
                "    revision = 1",
                "    revision = 1\n    revision = 2",
            ),
            "duplicate compiled field",
        ),
        (
            lambda source: source.replace("    revision = 1\n", ""),
            "missing compiled fields",
        ),
        (
            lambda source: source.replace("    revision = 1", "    revision = True"),
            "only strings and integers",
        ),
        (
            lambda source: source.replace("    revision = 1", "    revision = {1}"),
            "Set is not allowed",
        ),
        (
            lambda source: source.replace(
                "class CompiledProfile:\n",
                'class CompiledProfile:\n    "class docstrings are not allowed"\n',
            ),
            "only plain literal assignments",
        ),
        (
            lambda source: source.replace(
                "    profile_schema_version = 1",
                "    if True:\n        profile_schema_version = 1",
            ),
            "only plain literal assignments",
        ),
    ],
)
def test_compiled_ast_rejects_executable_or_non_contract_syntax(
    mutate,
    expected: str,
) -> None:
    emitted = emit_compiled_profile(parse_profile(VALID_PROFILE))

    with pytest.raises(ProfileValidationError, match=expected):
        parse_compiled_profile(mutate(emitted))


def test_compiled_parser_never_executes_malicious_source(tmp_path) -> None:
    target = tmp_path / "owned"
    emitted = emit_compiled_profile(parse_profile(VALID_PROFILE))
    malicious = emitted.replace(
        '    profile_id = "coding-team"',
        f"    profile_id = open({str(target)!r}, 'w').write('owned')",
    )

    with pytest.raises(ProfileValidationError, match="Call is not allowed"):
        parse_compiled_profile(malicious)

    assert not target.exists()


def test_compiled_parser_rejects_oversized_source_before_ast_parsing() -> None:
    with pytest.raises(ProfileValidationError, match="compiled profile exceeds"):
        parse_compiled_profile("#" * (MAX_COMPILED_SOURCE_CHARS + 1))


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda source: source.replace(
                '    default_agent_id = "implementer"',
                '    default_agent_id = "reviewer"',
            ),
            "default agent must name",
        ),
        (
            lambda source: source.replace(
                '            "tools": ("read", "glob", "grep"),',
                '            "tools": ("read", "read"),',
            ),
            "tool names must be unique",
        ),
        (
            lambda source: source.replace(
                '            "prompt": "Inspect the repository and make a concrete plan.",',
                '            "prompt": "Inspect the repository and make a concrete plan.",\n'
                '            "unknown": "value",',
            ),
            "extra_forbidden",
        ),
        (
            lambda source: source.replace(
                '            "id": "planner",',
                '            "id": "planner",\n            "id": "implementer",',
            ),
            "duplicate compiled mapping key",
        ),
    ],
)
def test_compiled_literal_values_still_pass_runtime_semantic_validation(
    mutate,
    expected: str,
) -> None:
    emitted = emit_compiled_profile(parse_profile(VALID_PROFILE))

    with pytest.raises(ProfileValidationError, match=expected):
        parse_compiled_profile(mutate(emitted))


def test_compiled_parser_rejects_invalid_artifact_provenance() -> None:
    emitted = emit_compiled_profile(parse_profile(VALID_PROFILE))

    with pytest.raises(ProfileValidationError, match="artifact_id"):
        parse_compiled_profile(emitted, artifact_id="")
    with pytest.raises(ProfileValidationError, match="generation"):
        parse_compiled_profile(emitted, generation=-1)
