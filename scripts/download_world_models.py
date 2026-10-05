"""Explicitly fetch pinned public LingBot-v2 1.3B weights and shared assets.

This downloads models only. It does not run upstream code, allocate a GPU,
install dependencies, accept a gated agreement, or start an interactive service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

MODELS = (
    (
        "robbyant/lingbot-world-v2-1.3b-causal-fast",
        "7e36a5f919f86cb4255cc9bfc30adb44963fbde1",
        "transformers",
        ["model-0000*-of-00006.safetensors", "model.safetensors.index.json", "README.md"],
    ),
    (
        "robbyant/lingbot-world-v2-14b-causal-fast",
        "5c33dd40b213598c418fd25bff30fdbd23fd38a7",
        "assets",
        ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl/*", "README.md"],
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--acknowledge-noncommercial", action="store_true",
                        help="Acknowledge v2's CC BY-NC-SA 4.0 research/demo restriction")
    args = parser.parse_args()
    if not args.acknowledge_noncommercial:
        parser.error("Review upstream CC BY-NC-SA 4.0 terms and explicitly acknowledge the restriction")
    from huggingface_hub import HfApi, snapshot_download
    import fnmatch

    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=False)
    manifests = []
    total_bytes = 0
    for model_id, revision, subdir, patterns in MODELS:
        info = api.model_info(model_id, revision=revision, files_metadata=True)
        if info.sha != revision or info.gated or info.private:
            raise RuntimeError("Expected public ungated pinned model; no agreement bypass is supported")
        files = [item for item in info.siblings
                 if any(fnmatch.fnmatchcase(item.rfilename, pattern) for pattern in patterns)]
        if not files:
            raise RuntimeError("The pinned release has no matching files")
        total_bytes += sum(item.size or 0 for item in files)
        manifests.append((model_id, revision, subdir, patterns, files))
    if shutil.disk_usage(root).free < total_bytes + 8 * 1024**3:
        raise RuntimeError("Insufficient free disk for a full snapshot plus 8 GiB margin")
    print(f"Pinned LingBot-v2 assets: {total_bytes / 1024**3:.2f} GiB; inference not yet verified", flush=True)
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Download and integrity check only; no inference or latency claim",
              "license": "CC BY-NC-SA 4.0; upstream non-commercial restrictions apply",
              "models": []}
    for model_id, revision, subdir, patterns, files in manifests:
        target = root / subdir
        snapshot_download(model_id, revision=revision, local_dir=target,
                          allow_patterns=patterns, token=False, max_workers=2)
        records = []
        for item in files:
            path = target / item.rfilename
            with path.open("rb") as content:
                actual = hashlib.file_digest(content, "sha256").hexdigest()
            expected = getattr(item.lfs, "sha256", None) if item.lfs else None
            if path.stat().st_size != item.size or (expected and actual != expected):
                raise RuntimeError(f"Integrity check failed: {item.rfilename}")
            records.append({"file": item.rfilename, "bytes": item.size,
                            "sha256": actual, "matches_upstream_lfs": bool(expected)})
        report["models"].append({"model": model_id, "revision": revision,
                                 "subdir": subdir, "files": records})
        print(f"Verified {model_id}: {len(records)} files", flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (root / "download-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("All pinned files verified. This is NOT an interactive model readiness result.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
