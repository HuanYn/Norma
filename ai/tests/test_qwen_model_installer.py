from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "install_qwen3vl_model.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("install_qwen3vl_model", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(name: str, payload: bytes, source: str) -> dict[str, object]:
    return {
        "name": name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": source,
    }


def _tiny_manifest(module: ModuleType, weight: bytes) -> dict[str, object]:
    metadata = module.GENERATED_ASSETS["configuration.json"]
    return {
        "schema": module.EXPECTED_SCHEMA,
        "model_id": module.EXPECTED_MODEL_ID,
        "revision": module.EXPECTED_REVISION,
        "files": [
            _record("model.safetensors", weight, "huggingface"),
            _record("configuration.json", metadata, "modelscope-snapshot-metadata"),
        ],
    }


def test_checked_installer_publishes_only_after_complete_snapshot(
    tmp_path: Path,
) -> None:
    module = _load_script()
    weight = b"tiny-test-weight"
    manifest = _tiny_manifest(module, weight)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.safetensors").write_bytes(weight)
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(cache / str(kwargs["filename"]))

    output = tmp_path / "models" / "qwen"
    result = module._install_snapshot(
        manifest=manifest,
        output=output,
        offline=True,
        downloader=download,
    )

    assert result == "installed"
    assert {item.name for item in output.iterdir()} == {
        "configuration.json",
        "model.safetensors",
    }
    assert (output / "model.safetensors").read_bytes() == weight
    assert (output / "configuration.json").read_bytes() == module.GENERATED_ASSETS[
        "configuration.json"
    ]
    assert calls == [
        {
            "repo_id": module.EXPECTED_MODEL_ID,
            "filename": "model.safetensors",
            "revision": module.EXPECTED_REVISION,
            "local_files_only": True,
        }
    ]

    def unexpected_download(**kwargs: object) -> str:
        raise AssertionError(f"valid installed snapshot should be reused: {kwargs}")

    assert (
        module._install_snapshot(
            manifest=manifest,
            output=output,
            offline=False,
            downloader=unexpected_download,
        )
        == "reused"
    )


def test_failed_download_never_publishes_partial_snapshot(tmp_path: Path) -> None:
    module = _load_script()
    manifest = _tiny_manifest(module, b"expected")
    bad_cache = tmp_path / "bad.safetensors"
    bad_cache.write_bytes(b"different")
    output = tmp_path / "models" / "qwen"

    with pytest.raises(ValueError, match="content pin"):
        module._install_snapshot(
            manifest=manifest,
            output=output,
            offline=False,
            downloader=lambda **_: str(bad_cache),
        )

    assert not output.exists()
    assert not list((tmp_path / "models").glob(".qwen.install-*"))


def test_existing_non_closed_world_snapshot_is_rejected(tmp_path: Path) -> None:
    module = _load_script()
    weight = b"expected"
    manifest = _tiny_manifest(module, weight)
    output = tmp_path / "qwen"
    output.mkdir()
    (output / "model.safetensors").write_bytes(weight)
    (output / "configuration.json").write_bytes(
        module.GENERATED_ASSETS["configuration.json"]
    )
    (output / "unexpected.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="closed-world"):
        module._install_snapshot(
            manifest=manifest,
            output=output,
            offline=True,
            downloader=lambda **_: "unused",
        )


def test_existing_snapshot_may_include_non_runtime_documentation(
    tmp_path: Path,
) -> None:
    module = _load_script()
    weight = b"expected"
    manifest = _tiny_manifest(module, weight)
    output = tmp_path / "qwen"
    output.mkdir()
    (output / "model.safetensors").write_bytes(weight)
    (output / "configuration.json").write_bytes(
        module.GENERATED_ASSETS["configuration.json"]
    )
    (output / "README.md").write_text("documentation only", encoding="utf-8")
    (output / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")

    assert (
        module._install_snapshot(
            manifest=manifest,
            output=output,
            offline=True,
            downloader=lambda **_: "unused",
        )
        == "reused"
    )


def test_repository_manifest_and_generated_asset_are_self_consistent() -> None:
    module = _load_script()
    manifest = module._load_manifest(module.DEFAULT_MANIFEST)

    assert manifest["model_id"] == module.EXPECTED_MODEL_ID
    assert manifest["revision"] == module.EXPECTED_REVISION
    generated = {
        item["name"]: item
        for item in manifest["files"]
        if item["source"] == "modelscope-snapshot-metadata"
    }
    assert set(generated) == {"configuration.json"}
    payload = module.GENERATED_ASSETS["configuration.json"]
    assert generated["configuration.json"]["size"] == len(payload)
    assert (
        generated["configuration.json"]["sha256"] == hashlib.sha256(payload).hexdigest()
    )
