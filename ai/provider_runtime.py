from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone

from ai.index.embedding import EmbeddingProvider
from ai.schemas import EmbeddingProviderStatusResponse


class EmbeddingWarmupManager:
    """Idempotent daemon-thread warmup with pollable local state."""

    def __init__(self, provider_factory: Callable[[], EmbeddingProvider]) -> None:
        self.provider_factory = provider_factory
        self.lock = threading.Lock()
        self.state = "idle"
        self.error: str | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.thread: threading.Thread | None = None

    def submit(self) -> EmbeddingProviderStatusResponse:
        provider = self.provider_factory()
        with self.lock:
            if provider.is_loaded:
                self.state = "ready"
                self.error = None
                if self.finished_at is None:
                    self.finished_at = _now()
                return self._status_locked(provider)
            if self.state == "loading" and self.thread and self.thread.is_alive():
                return self._status_locked(provider)
            self.state = "loading"
            self.error = None
            self.started_at = _now()
            self.finished_at = None
            self.thread = threading.Thread(
                target=self._run,
                args=(provider,),
                name="norma-embedding-warmup",
                daemon=True,
            )
            self.thread.start()
            return self._status_locked(provider)

    def status(self) -> EmbeddingProviderStatusResponse:
        provider = self.provider_factory()
        with self.lock:
            if provider.is_loaded and self.state == "idle":
                self.state = "ready"
                self.error = None
                self.finished_at = self.finished_at or _now()
            return self._status_locked(provider)

    def _run(self, provider: EmbeddingProvider) -> None:
        try:
            provider.warmup()
        except Exception as error:
            with self.lock:
                self.state = "failed"
                self.error = str(error)
                self.finished_at = _now()
        else:
            with self.lock:
                self.state = "ready"
                self.error = None
                self.finished_at = _now()

    def _status_locked(
        self, provider: EmbeddingProvider
    ) -> EmbeddingProviderStatusResponse:
        return EmbeddingProviderStatusResponse(
            provider=provider.name,
            dimension=provider.dimension,
            model_backed=provider.model_backed,
            loaded=provider.is_loaded,
            device=provider.runtime_device,
            warmup_state=self.state,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
