"""Run the private single-GPU worker; model load starts only after job submission."""
from __future__ import annotations

import argparse
import os
import secrets
import stat
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.video.generator import MODEL_ID  # noqa: E402 - standalone script path setup
from ai.video.worker import WorkerSettings, create_app  # noqa: E402


def read_private_token(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size > 4096:
            raise ValueError("Token file must be a small regular file")
        if os.name == "posix":
            if details.st_mode & 0o077 or details.st_uid != os.getuid():
                raise ValueError("Token file must be owned by this user and accessible only to its owner")
        token = os.read(descriptor, 4097).decode("ascii").strip()
        if not 24 <= len(token) <= 4096 or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise ValueError("Token file contains an invalid token")
        return token
    finally:
        os.close(descriptor)


def create_private_token(path: Path) -> str:
    """Exclusive creation: no overwrite, symlink following, logging or shell token."""
    token = secrets.token_urlsafe(32)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        content = (token + "\n").encode("ascii")
        while content:
            written = os.write(descriptor, content)
            content = content[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return token


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--offload", choices=("model", "sequential"), default="model")
    parser.add_argument("--allow-model-downloads", action="store_true")
    tokens = parser.add_mutually_exclusive_group()
    tokens.add_argument("--token-file", type=Path, help="Existing owner-only token file")
    tokens.add_argument("--create-token-file", type=Path, help="Create a new owner-only token file exclusively")
    args = parser.parse_args(argv)
    token = os.environ.get("NORMA_VIDEO_WORKER_TOKEN", "")
    if token and (args.token_file or args.create_token_file):
        parser.error("Use one token source: environment OR private token file")
    try:
        if args.token_file:
            token = read_private_token(args.token_file)
        elif args.create_token_file:
            token = create_private_token(args.create_token_file)
    except (OSError, ValueError, UnicodeError):
        parser.error("Could not read/create a private token file; check ownership, permissions and existence")
    if args.host not in {"127.0.0.1", "::1", "localhost"} and not token:
        parser.error("Non-loopback binding requires NORMA_VIDEO_WORKER_TOKEN; prefer an SSH tunnel")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    try:
        settings = WorkerSettings(
            data_dir=args.data_dir, token=token, model_path=args.model_path,
            revision=args.revision, allow_downloads=args.allow_model_downloads, offload=args.offload,
        )
    except ValueError as exc:
        parser.error(str(exc))
    import uvicorn
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
