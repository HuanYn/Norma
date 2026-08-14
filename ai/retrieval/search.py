from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ai.index.embedding import EmbeddingProvider
from ai.schemas import (
    AlbumEmbeddingResponse,
    AlbumSearchRequest,
    AlbumSearchResponse,
    SearchMatch,
)
from ai.storage import Database


class RetrievalService:
    def __init__(
        self,
        database: Database,
        data_dir: Path,
        provider: EmbeddingProvider,
    ) -> None:
        self.database = database
        self.data_dir = data_dir
        self.provider = provider

    def embed_album(self, album_id: str) -> AlbumEmbeddingResponse:
        started = time.perf_counter()
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id, absolute_path FROM photos WHERE album_id = ? ORDER BY id",
                (album_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"album not found or empty: {album_id}")

        directory = self.data_dir / "embeddings" / self.provider.name / album_id
        directory.mkdir(parents=True, exist_ok=True)
        updates: list[tuple[str, str]] = []
        for row in rows:
            vector = self.provider.embed_image(Path(row["absolute_path"]))
            if vector.shape != (self.provider.dimension,):
                raise ValueError(
                    f"provider returned {vector.shape}, expected {(self.provider.dimension,)}"
                )
            target = (directory / f"{row['id']}.npy").resolve()
            np.save(target, vector.astype(np.float32), allow_pickle=False)
            updates.append((str(target), row["id"]))

        with self.database.connect() as connection:
            connection.executemany(
                "UPDATE photos SET embedding_path = ? WHERE id = ?", updates
            )
        return AlbumEmbeddingResponse(
            album_id=album_id,
            count=len(updates),
            provider=self.provider.name,
            dimension=self.provider.dimension,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )

    def search(self, request: AlbumSearchRequest) -> AlbumSearchResponse:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, absolute_path, thumbnail_path, quality_score,
                       auto_reject, similarity_group, embedding_path
                FROM photos
                WHERE album_id = ? AND embedding_path IS NOT NULL
                """,
                (request.album_id,),
            ).fetchall()
        if not rows:
            raise KeyError("album has no cached embeddings; call the embed endpoint first")

        if request.reference_photo_id:
            reference = next(
                (row for row in rows if row["id"] == request.reference_photo_id), None
            )
            if reference is None:
                raise KeyError(f"reference photo not found: {request.reference_photo_id}")
            query_vector = _load_vector(
                reference["embedding_path"], self.provider.dimension
            )
            mode = "image"
        else:
            query_vector = self.provider.embed_text(request.query or "")
            mode = "text"

        if request.subset_photo_ids is not None:
            allowed = set(request.subset_photo_ids)
            rows = [row for row in rows if row["id"] in allowed]

        matches: list[SearchMatch] = []
        for row in rows:
            if mode == "image" and row["id"] == request.reference_photo_id:
                continue
            vector = _load_vector(row["embedding_path"], self.provider.dimension)
            score = float(np.dot(query_vector, vector))
            matches.append(
                SearchMatch(
                    photo_id=row["id"],
                    filename=Path(row["absolute_path"]).name,
                    thumbnail_url=_thumbnail_url(request.album_id, row["thumbnail_path"]),
                    score=round(score, 6),
                    quality_score=float(row["quality_score"] or 0.0),
                    auto_reject=bool(row["auto_reject"]),
                    similarity_group=row["similarity_group"],
                )
            )
        matches.sort(key=lambda match: (-match.score, -match.quality_score, match.photo_id))
        return AlbumSearchResponse(
            album_id=request.album_id,
            mode=mode,
            provider=self.provider.name,
            matches=matches[: request.limit],
        )


def _load_vector(path: str, expected_dimension: int) -> np.ndarray:
    vector = np.load(path, allow_pickle=False).astype(np.float32)
    if vector.shape != (expected_dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"invalid cached embedding at {path}; re-run the embed endpoint"
        )
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 1e-12 else vector / norm


def _thumbnail_url(album_id: str, thumbnail_path: str) -> str:
    return f"/media/thumbnails/{album_id}/{Path(thumbnail_path).name}"
