"""Install Norma's pinned Qwen3-VL snapshot outside request handling.

The runtime is deliberately offline-only.  This command is the explicit,
auditable provisioning step: it downloads the exact Hugging Face revision,
checks every byte against Norma's version-controlled manifest, adds the one
small ModelScope compatibility metadata file captured by that manifest, and
publishes the directory only after the complete snapshot passes verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "ai" / "rag" / "qwen3_vl_2b_manifest.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / ".norma"
    / "data"
    / "models"
    / "qwen3-vl"
    / "Qwen3-VL-2B-Instruct-modelscope"
)
EXPECTED_SCHEMA = "norma-pinned-model-manifest-v1"
EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
EXPECTED_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
GENERATED_ASSETS = {
    "configuration.json": b'{"framework":"Pytorch","task":"image-text-to-text"}',
}
IGNORED_DOCUMENTATION_ASSETS = frozenset({".gitattributes", "README.md"})
COPY_CHUNK_BYTES = 8 * 1024 * 1024


DownloadFunction = Callable[..., str]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in model manifest: {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
    )
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "model_id", "revision", "files"}
        or document["schema"] != EXPECTED_SCHEMA
        or document["model_id"] != EXPECTED_MODEL_ID
        or document["revision"] != EXPECTED_REVISION
        or not isinstance(document["files"], list)
    ):
        raise ValueError("the Qwen3-VL model manifest is invalid")

    seen: set[str] = set()
    for item in document["files"]:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "size",
            "sha256",
            "source",
        }:
            raise ValueError("the Qwen3-VL model manifest is invalid")
        name = item["name"]
        size = item["size"]
        sha256 = item["sha256"]
        source = item["source"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or source not in {"huggingface", "modelscope-snapshot-metadata"}
        ):
            raise ValueError("the Qwen3-VL model manifest is invalid")
        if source == "modelscope-snapshot-metadata" and name not in GENERATED_ASSETS:
            raise ValueError("the manifest contains an unknown generated model asset")
        seen.add(name)
    if not seen or "model.safetensors" not in seen:
        raise ValueError("the Qwen3-VL model manifest is invalid")
    return document


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _verify_file(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"model asset is missing or unsafe: {path.name}")
    actual_size, actual_sha256 = _sha256_file(path)
    if actual_size != size or actual_sha256 != sha256:
        raise ValueError(f"model asset failed its content pin: {path.name}")


def _verify_directory(target: Path, manifest: dict[str, Any]) -> None:
    if not target.is_dir() or target.is_symlink():
        raise ValueError("the Qwen3-VL target is not a regular directory")
    expected = {item["name"] for item in manifest["files"]}
    entries = {entry.name: entry for entry in target.iterdir()}
    actual_runtime = set(entries) - IGNORED_DOCUMENTATION_ASSETS
    if actual_runtime != expected:
        raise ValueError(
            "the Qwen3-VL directory is not the pinned closed-world snapshot"
        )
    for name in set(entries) & IGNORED_DOCUMENTATION_ASSETS:
        if not entries[name].is_file() or entries[name].is_symlink():
            raise ValueError("the Qwen3-VL documentation asset is unsafe")
    for item in manifest["files"]:
        _verify_file(
            target / item["name"],
            size=item["size"],
            sha256=item["sha256"],
        )


def _copy_and_verify(
    source: Path,
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    if not source.is_file():
        raise ValueError(f"download cache did not return a file: {target.name}")
    digest = hashlib.sha256()
    copied = 0
    before = source.stat()
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        while chunk := input_handle.read(COPY_CHUNK_BYTES):
            copied += len(chunk)
            digest.update(chunk)
            output_handle.write(chunk)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"download cache changed while copying: {target.name}")
    if copied != expected_size or digest.hexdigest() != expected_sha256:
        raise ValueError(
            f"downloaded model asset failed its content pin: {target.name}"
        )


def _install_snapshot(
    *,
    manifest: dict[str, Any],
    output: Path,
    offline: bool,
    downloader: DownloadFunction,
) -> str:
    target = output.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _verify_directory(target, manifest)
        return "reused"

    staging_prefix = f".{target.name}.install-"
    with tempfile.TemporaryDirectory(prefix=staging_prefix, dir=target.parent) as raw:
        staging = Path(raw)
        for item in manifest["files"]:
            destination = staging / item["name"]
            if item["source"] == "huggingface":
                cached = Path(
                    downloader(
                        repo_id=manifest["model_id"],
                        filename=item["name"],
                        revision=manifest["revision"],
                        local_files_only=offline,
                    )
                )
                _copy_and_verify(
                    cached,
                    destination,
                    expected_size=item["size"],
                    expected_sha256=item["sha256"],
                )
            else:
                payload = GENERATED_ASSETS[item["name"]]
                destination.write_bytes(payload)
                _verify_file(
                    destination,
                    size=item["size"],
                    sha256=item["sha256"],
                )
        _verify_directory(staging, manifest)
        try:
            staging.rename(target)
        except FileExistsError:
            _verify_directory(target, manifest)
            return "reused-after-race"
    _verify_directory(target, manifest)
    return "installed"


def _load_huggingface_downloader() -> DownloadFunction:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            'install the multimodal dependencies first: python -m pip install -e ".[multimodal]"'
        ) from error
    return hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify Norma's exact local Qwen3-VL model snapshot; "
            "this explicit installer is the only step that may access the Hub"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use an existing Hugging Face cache only; never access the network",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest.resolve(strict=True))
    result = _install_snapshot(
        manifest=manifest,
        output=args.output,
        offline=args.offline,
        downloader=_load_huggingface_downloader(),
    )
    print(
        json.dumps(
            {
                "file_count": len(manifest["files"]),
                "model_id": manifest["model_id"],
                "output": str(args.output.absolute()),
                "result": result,
                "revision": manifest["revision"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
