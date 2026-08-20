from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from aeloon_runtime.bootstrap import create_runtime_service
from aeloon_runtime.rpc.protocol import RpcError
from aeloon_runtime.runtime_server_v3 import (
    FILE_BYTES,
    IMAGE_BYTES,
    MAX_FRAME_BYTES,
    RuntimeV3Server,
    _range_contains,
    pack_frame,
)


def test_rpc_source_and_manifest_are_strict_and_complete() -> None:
    source = json.loads(Path("docs/rpc-v3.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        Path("aeloon_runtime/rpc/aeloon-rpc-v3.manifest.json").read_text(encoding="utf-8")
    )
    assert source["$schema"].endswith("draft/2020-12/schema")
    assert source["frame_max_bytes"] == MAX_FRAME_BYTES
    assert source["file_max_bytes"] == FILE_BYTES
    assert source["image_max_bytes"] == IMAGE_BYTES
    assert len(manifest["methods"]) == 69
    assert len(manifest["plugin_methods"]) == 9
    assert len(manifest["events"]) >= 30
    assert len(manifest["errors"]) >= 15
    assert "plugins.configure" in manifest["methods"]
    assert "plugin.cloud.account_login" in manifest["plugin_methods"]
    Draft202012Validator.check_schema(source)

    defs = manifest["$defs"]
    thread_create = Draft202012Validator(
        {"$defs": defs, "$ref": "#/$defs/Params_thread_create_v3"}
    )
    thread_create.validate({"project_id": "p", "kind": "standard"})
    thread_create.validate({"workspace": "/tmp/workspace", "kind": "worktree"})
    with pytest.raises(ValidationError):
        thread_create.validate({"kind": "standard"})
    with pytest.raises(ValidationError):
        thread_create.validate(
            {"project_id": "p", "workspace": "/tmp/workspace", "kind": "standard"}
        )

    fs_list = Draft202012Validator({"$defs": defs, "$ref": "#/$defs/Params_fs_list_v3"})
    fs_list.validate({"thread_id": "t", "path": "src"})
    fs_list.validate({"root": "/tmp/workspace"})
    with pytest.raises(ValidationError):
        fs_list.validate({"thread_id": "t", "root": "/tmp/workspace"})

    settings_update = Draft202012Validator(
        {"$defs": defs, "$ref": "#/$defs/Params_settings_update_v3"}
    )
    settings_update.validate(
        {
            "patch": {},
            "revision": 1,
            "workspace": "/tmp/workspace",
            "secret_actions": [
                {"path": "providers.deepseek.api_key", "action": "set", "value": "secret"}
            ],
        }
    )
    provider_list = Draft202012Validator(
        {"$defs": defs, "$ref": "#/$defs/Params_provider_list_v3"}
    )
    provider_list.validate({"workspace": "/tmp/workspace"})
    cloud_login = Draft202012Validator(
        {"$defs": defs, "$ref": "#/$defs/Params_plugin_cloud_account_login_v3"}
    )
    cloud_login.validate({"username": "user", "password": "pass", "workspace": "/tmp"})


def test_handshake_negotiates_a_window_not_an_exact_version() -> None:
    from aeloon_runtime.runtime_server_v3 import SUPPORTED_PROTOCOLS, _range_contains

    # Newest first: a client that speaks both must be answered with the newer one.
    assert SUPPORTED_PROTOCOLS[0] == "3.1.0"
    assert "3.0.0" in SUPPORTED_PROTOCOLS

    def negotiate(minimum: str, maximum: str) -> str | None:
        return next(
            (v for v in SUPPORTED_PROTOCOLS if _range_contains(minimum, maximum, v)),
            None,
        )

    # A client pinned to the previous minor keeps working after the Runtime moves
    # on; that is the whole point of versioning the two sides independently.
    assert negotiate("3.0.0", "3.0.0") == "3.0.0"
    assert negotiate("3.0.0", "3.1.0") == "3.1.0"
    assert negotiate("3.1.0", "3.1.0") == "3.1.0"
    assert negotiate("4.0.0", "4.0.0") is None


def test_semver_endpoint_range_and_frame_limit() -> None:
    assert _range_contains("3.0.0", "3.1.0", "3.0.0")
    # A prerelease sorts below the release it precedes, so a client that only
    # speaks the final 3.0.0 must not be matched by a 3.0.0-rc Runtime.
    assert not _range_contains("3.0.0", "3.0.0", "3.0.0-rc.1")
    assert _range_contains("3.0.0-rc.1", "3.0.0", "3.0.0-rc.1")
    payload = {"data": "x" * (MAX_FRAME_BYTES - len(b'{"data":""}'))}
    assert len(pack_frame(payload)) > MAX_FRAME_BYTES
    with pytest.raises(RpcError, match="40 MiB"):
        pack_frame({"data": "x" * MAX_FRAME_BYTES})


@pytest.mark.asyncio
async def test_workspace_symlink_and_attachment_id_boundaries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    data_dir = tmp_path / "data"
    runtime = create_runtime_service(config_path=data_dir / "config.json", data_dir=data_dir)
    server = RuntimeV3Server(runtime, tmp_path / "runtime.sock", (workspace,), data_dir)
    try:
        project = await server.dispatch("project.add", {"path": str(workspace)})
        created = await server.dispatch(
            "thread.create", {"project_id": project["project"]["id"], "kind": "standard"}
        )
        thread_id = created["thread"]["id"]
        settings = await server.dispatch("settings.get", {})
        assert settings["settings"]["plugins"] == {}
        updated_settings = await server.dispatch(
            "settings.update",
            {
                "revision": settings["revision"],
                "patch": {"plugins": {"demo": {"enabled": True, "api_key": "secret"}}},
            },
        )
        assert updated_settings["settings"]["plugins"] == {
            "demo": {"enabled": True, "api_key": "***"}
        }
        with pytest.raises(RpcError, match="outside"):
            await server.dispatch(
                "fs.read", {"thread_id": thread_id, "path": "escape/secret.txt"}
            )
        (workspace / "inside-link").symlink_to(workspace, target_is_directory=True)
        with pytest.raises(RpcError, match="Symbolic links"):
            await server.dispatch(
                "fs.read", {"thread_id": thread_id, "path": "inside-link/README.md"}
            )
        with pytest.raises(RpcError, match="outside"):
            await server.dispatch("fs.list", {"thread_id": thread_id, "path": "."})
        with pytest.raises(RpcError, match="Relative path"):
            await server.dispatch(
                "artifact.resolve", {"thread_id": thread_id, "paths": ["../secret.txt"]}
            )
        with pytest.raises(RpcError, match="outside"):
            await server.dispatch(
                "git.diff",
                {"thread_id": thread_id, "scope": "changes", "path": "escape/secret.txt"},
            )
        with pytest.raises(RpcError, match="outside"):
            await server.dispatch(
                "git.stage", {"thread_id": thread_id, "paths": ["escape/secret.txt"]}
            )
        with pytest.raises(RpcError, match="not found"):
            await server.dispatch(
                "attachment.download", {"attachment_id": "00000000-0000-0000-0000-000000000000"}
            )
        attachment = server.store.add_attachment(
            name="secret.txt",
            mime_type="text/plain",
            data=b"secret",
            root=data_dir / "attachments",
        )
        with server.store.transaction() as db:
            db.execute(
                "UPDATE attachments SET storage_path = ? WHERE id = ?",
                (str(outside / "secret.txt"), attachment["id"]),
            )
        with pytest.raises(RpcError, match="outside Runtime storage"):
            await server.dispatch(
                "attachment.download", {"attachment_id": attachment["id"]}
            )
        # A blob that remains inside Runtime storage but no longer matches its
        # content address must not be served either.
        with server.store.transaction() as db:
            db.execute(
                "UPDATE attachments SET storage_path = ? WHERE id = ?",
                (str(data_dir / "attachments" / "tampered.blob"), attachment["id"]),
            )
        (data_dir / "attachments" / "tampered.blob").write_bytes(b"wrong")
        with pytest.raises(RpcError, match="content address is invalid"):
            await server.dispatch(
                "attachment.download", {"attachment_id": attachment["id"]}
            )
        with pytest.raises(RpcError, match="exceeds"):
            await server.dispatch(
                "attachment.upload",
                {
                    "name": "large.bin",
                    "mime_type": "application/octet-stream",
                    "data_base64": base64.b64encode(b"x" * (FILE_BYTES + 1)).decode("ascii"),
                },
            )
        with pytest.raises(RpcError, match="exceeds"):
            await server.dispatch(
                "attachment.upload",
                {
                    "name": "large.png",
                    "mime_type": "image/png",
                    "data_base64": base64.b64encode(b"x" * (IMAGE_BYTES + 1)).decode("ascii"),
                },
            )
        preview_source = ("a" * 20_001).encode("utf-8")
        preview_attachment = server.store.add_attachment(
            name="preview.txt",
            mime_type="text/plain",
            data=preview_source,
            root=data_dir / "attachments",
        )
        preview = await server.dispatch(
            "attachment.preview", {"attachment_id": preview_attachment["id"]}
        )
        assert preview == {"kind": "text", "preview": "a" * 20_000}
    finally:
        await runtime.close()
        server.store.close()
