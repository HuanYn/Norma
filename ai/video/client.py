from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from PIL import Image

from .models import MAX_IMAGE_BYTES, VideoOptions, decode_image


class VideoWorkerError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class VideoWorkerClient:
    """For the local backend. Never accept a worker URL from the browser payload."""

    def __init__(self, base_url: str, *, token: str = "", timeout: float = 30):
        address = urlsplit(base_url)
        if (
            address.scheme not in {"http", "https"}
            or not address.hostname
            or address.username
            or address.password
        ):
            raise ValueError("Worker URL must be an HTTP(S) origin without credentials")
        if address.query or address.fragment or address.path not in {"", "/"}:
            raise ValueError("Worker URL must be an origin, without a path or query")
        if address.scheme == "http" and address.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError(
                "Use an SSH loopback tunnel or HTTPS for remote image upload"
            )
        if not 0 < timeout <= 120:
            raise ValueError("Invalid HTTP timeout")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._opener = build_opener(ProxyHandler({}), NoRedirect())

    @staticmethod
    def _job_id(job_id: str) -> str:
        import re

        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Invalid job ID")
        return job_id

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read(128 * 1024 + 1)
                if len(body) > 128 * 1024:
                    raise VideoWorkerError("Worker response exceeds size limit")
                result = json.loads(body)
                if not isinstance(result, dict):
                    raise VideoWorkerError("Invalid worker response")
                return result
        except HTTPError as exc:
            raise VideoWorkerError(f"Video worker returned HTTP {exc.code}") from exc
        except (URLError, OSError, ValueError) as exc:
            raise VideoWorkerError(
                "Video worker is unreachable or returned an invalid response"
            ) from exc

    def health(self) -> dict:
        return self._request("GET", "/health")

    def submit(
        self,
        image_bytes: bytes,
        options: VideoOptions,
        *,
        upload_confirmed: bool = False,
    ) -> dict:
        if upload_confirmed is not True:
            raise ValueError(
                "Confirm this image's remote upload before creating a video job"
            )
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds 8 MiB; resize locally before upload")
        image = decode_image(base64.b64encode(image_bytes).decode("ascii"))
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=93)
        payload = options.model_dump()
        payload.update(
            image_base64=base64.b64encode(encoded.getvalue()).decode("ascii"),
            upload_confirmed=True,
        )
        return self._request("POST", "/v1/video/jobs", payload)

    def status(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/video/jobs/{self._job_id(job_id)}")

    def cancel(self, job_id: str) -> dict:
        return self._request("POST", f"/v1/video/jobs/{self._job_id(job_id)}/cancel")

    def download(
        self, job_id: str, destination: Path, *, max_bytes: int = 256 * 1024 * 1024
    ) -> None:
        """Save only to a caller-owned local path, never overwrite an existing file."""
        job_id = self._job_id(job_id)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        request = Request(
            f"{self.base_url}/v1/video/jobs/{job_id}/artifact", headers=headers
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                content = response.read(max_bytes + 1)
                if (
                    len(content) > max_bytes
                    or len(content) < 32
                    or content[4:8] != b"ftyp"
                ):
                    raise VideoWorkerError("Worker artifact is not a bounded MP4")
            # Never expose a partially written file as a completed cached MP4.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{job_id}-",
                    suffix=".part",
                    dir=destination.parent,
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Hard-link publication is atomic and refuses to overwrite an
                # existing destination on both NTFS and the Linux worker host.
                os.link(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        except HTTPError as exc:
            raise VideoWorkerError(f"Video artifact returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise VideoWorkerError("Could not download video artifact") from exc
        except OSError as exc:
            raise VideoWorkerError(
                "Could not safely publish the local video artifact"
            ) from exc
