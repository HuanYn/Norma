"""Start Norma with a locally entered DeepSeek key, without saving the key.

Run from an interactive terminal: python scripts/start_deepseek.py
No API request is sent by this launcher. Cloud analysis still requires a click.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print(
        "Enter the key locally. Rotating any key previously shared in chat is recommended."
    )
    try:
        # getpass may otherwise fall back to echoing the input on an unsafe TTY.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("DeepSeek API Key (hidden): ")
    except (EOFError, getpass.GetPassWarning):
        print("Hidden input unavailable. Run this in a local interactive terminal.")
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    if (
        not api_key
        or len(api_key) > 4096
        or any(ord(character) < 33 or ord(character) > 126 for character in api_key)
    ):
        print("Key is empty or has invalid characters; nothing was started.")
        return 2

    child_environment = os.environ.copy()
    child_environment.update(
        {
            "NORMA_HOST": "127.0.0.1",
            "NORMA_PREFERENCE_MODE": "record-only",
            "NORMA_VLM_PROVIDER": "openai-compatible",
            "NORMA_VLM_BASE_URL": "https://api.deepseek.com",
            "NORMA_VLM_MODEL": "deepseek-flash",
            "NORMA_VLM_API_KEY": api_key,
            "NORMA_VLM_THINKING_MODE": "disabled",
            "NORMA_VLM_JSON_RESPONSE_FORMAT": "1",
            "NORMA_VLM_MAX_NEW_TOKENS": "1024",
            "NORMA_VLM_TIMEOUT_SECONDS": "60",
        }
    )
    print(
        "Starting Norma on loopback with DeepSeek vision; preference training is off."
    )
    print("Only clicking cloud analysis sends selected images and may incur API costs.")
    try:
        return subprocess.run(
            [sys.executable, "-m", "ai", "web"],
            cwd=PROJECT_ROOT,
            env=child_environment,
            check=False,
        ).returncode
    except KeyboardInterrupt:
        return 130
    except OSError:
        print("Could not start Python. Check the local environment and try again.")
        return 2
    finally:
        # Best effort only: Python/OS memory is not a secure erase facility.
        child_environment.pop("NORMA_VLM_API_KEY", None)
        api_key = ""


if __name__ == "__main__":
    raise SystemExit(main())
