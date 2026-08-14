from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    log_level: str
    embedding_provider: str = "lightweight"
    face_provider: str = "opencv-haar"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "norma.db"


def load_settings() -> Settings:
    return Settings(
        host=os.getenv("NORMA_HOST", "127.0.0.1"),
        port=int(os.getenv("NORMA_PORT", "8765")),
        data_dir=Path(os.getenv("NORMA_DATA_DIR", ".norma/data")).resolve(),
        log_level=os.getenv("NORMA_LOG_LEVEL", "INFO").upper(),
        embedding_provider=os.getenv("NORMA_EMBEDDING_PROVIDER", "lightweight"),
        face_provider=os.getenv("NORMA_FACE_PROVIDER", "opencv-haar"),
    )
