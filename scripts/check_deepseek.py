"""One explicit, bounded DeepSeek vision smoke using a synthetic image.

Credentials come only from NORMA_VLM_API_KEY. Never log response bodies on error.
No private album is read. The script does not retry or switch models silently.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from time import monotonic
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.rag import (  # noqa: E402
    GroundedRAGRequest,
    ProviderFailureError,
    RetrievalEvidence,
    build_evidence_bundle,
    generate_grounded,
    snapshot_image_bytes,
)
from ai.rag.cloud_runtime import (  # noqa: E402
    CloudVLMUnavailableError,
    _NoRedirect,
    _read_response_body,
    create_cloud_vlm_provider,
)


MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
MAX_MODELS_BYTES = 256 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-on-success", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get("NORMA_VLM_API_KEY", "")
    if (
        not api_key
        or len(api_key) > 4096
        or any(ord(char) < 33 or ord(char) > 126 for char in api_key)
    ):
        print('{"status":"configuration_required"}')
        return 2
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_requested": MODEL,
        "private_images_uploaded": 0,
        "vision_attempts": 0,
        "full_retrieval_test": False,
    }
    output_dir = PROJECT_ROOT / ".norma" / "deepseek-smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    stage = "models"
    try:
        request = Request(
            BASE_URL + "/models", headers={"Authorization": "Bearer " + api_key}
        )
        with build_opener(_NoRedirect()).open(request, timeout=20) as response:
            raw = _read_response_body(response, started + 20, MAX_MODELS_BYTES)
        document = json.loads(raw)
        models = [
            item["id"]
            for item in document.get("data", [])
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", item["id"])
            and api_key not in item["id"]
        ]
        report["models"] = models
        report["models_elapsed_ms"] = round((monotonic() - started) * 1000)
        if MODEL not in models:
            report["status"] = "vision_model_not_listed"
        else:
            stage = "vision"
            buffer = BytesIO()
            with Image.new("RGB", (320, 180), "white") as picture:
                draw = ImageDraw.Draw(picture)
                draw.rectangle((30, 45, 120, 135), fill="red")
                draw.ellipse((200, 45, 290, 135), fill="blue")
                picture.save(buffer, format="PNG")
            content = buffer.getvalue()
            (output_dir / "synthetic-shapes.png").write_bytes(content)
            snapshot = snapshot_image_bytes(
                content, display_name="synthetic-shapes.png", media_type="image/png"
            )
            bundle = build_evidence_bundle(
                query="请用一句中文描述图片中两个形状的颜色、形状和左右位置。",
                retrieval_provider_fingerprint="synthetic-fixture-no-retrieval-v1",
                candidate_photo_ids=("synthetic-01",),
                evidence=(
                    RetrievalEvidence(
                        photo_id="synthetic-01",
                        image=snapshot,
                        thumbnail=None,
                        retrieval_score=1.0,
                        rank=1,
                        provider_fingerprint="synthetic-fixture-no-retrieval-v1",
                    ),
                ),
            )
            provider = create_cloud_vlm_provider(
                BASE_URL,
                MODEL,
                api_key,
                thinking_mode="disabled",
                json_response_format=True,
                max_new_tokens=384,
                timeout_seconds=60,
            )
            report["vision_attempts"] = 1
            generation_started = monotonic()
            answer = generate_grounded(
                provider,
                GroundedRAGRequest(
                    query=bundle.query,
                    evidence=bundle,
                    retrieval_provider_fingerprint=bundle.provenance.retrieval_provider_fingerprint,
                    candidate_digest=bundle.provenance.candidate_digest,
                ),
            )
            report.update(
                status="vision_citation_smoke_passed",
                generation_elapsed_ms=round((monotonic() - generation_started) * 1000),
                image_sha256=snapshot.sha256,
                result=asdict(answer),
            )
    except HTTPError as error:
        report.update(status="http_error", stage=stage, http_status=error.code)
        error.close()
    except CloudVLMUnavailableError as error:
        report.update(status="provider_error", stage=stage, error_code=error.code)
    except ProviderFailureError as error:
        if isinstance(error.__cause__, CloudVLMUnavailableError):
            report.update(
                status="provider_error", stage=stage, error_code=error.__cause__.code
            )
        else:
            report.update(status="check_failed", stage=stage)
    except Exception:
        report.update(status="check_failed", stage=stage)
    report["total_elapsed_ms"] = round((monotonic() - started) * 1000)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if api_key in serialized:
        print('{"status":"report_redaction_failed"}')
        return 2
    (output_dir / "result.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)
    if report["status"] != "vision_citation_smoke_passed":
        return 1
    if args.serve_on_success:
        os.environ.update(
            NORMA_HOST="127.0.0.1",
            NORMA_PORT="8765",
            NORMA_PREFERENCE_MODE="record-only",
            NORMA_VLM_PROVIDER="openai-compatible",
            NORMA_VLM_BASE_URL=BASE_URL,
            NORMA_VLM_MODEL=MODEL,
            NORMA_VLM_THINKING_MODE="disabled",
            NORMA_VLM_JSON_RESPONSE_FORMAT="1",
            NORMA_VLM_MAX_NEW_TOKENS="1024",
            NORMA_VLM_TIMEOUT_SECONDS="60",
        )
        from ai.cli import main as web_main

        return web_main(["web"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
