from __future__ import annotations

import json
from pathlib import Path

from ai.schemas import (
    AlbumListResponse,
    AlbumPhotoListResponse,
    AlbumSummary,
    PhotoSummary,
    SelectionHistoryItem,
    SelectionHistoryResponse,
)
from ai.storage import Database


class AlbumCatalogService:
    """Read-only catalog over persisted albums and their derived state."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def list_albums(self, *, limit: int, offset: int) -> AlbumListResponse:
        self.database.initialize()
        with self.database.connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM albums").fetchone()[0])
            rows = connection.execute(
                f"{_ALBUM_SUMMARY_SQL} ORDER BY a.indexed_at DESC, a.created_at DESC "
                "LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return AlbumListResponse(
            items=[_album_summary(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_album(self, album_id: str) -> AlbumSummary:
        self.database.initialize()
        with self.database.connect() as connection:
            row = connection.execute(
                f"{_ALBUM_SUMMARY_SQL} WHERE a.id = ?", (album_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"album not found: {album_id}")
        return _album_summary(row)

    def list_photos(
        self,
        album_id: str,
        *,
        limit: int,
        offset: int,
        include_rejects: bool,
        sort: str,
    ) -> AlbumPhotoListResponse:
        album = self.get_album(album_id)
        where = "album_id = ?"
        parameters: list[object] = [album_id]
        if not include_rejects:
            where += " AND auto_reject = 0"
        order = {
            "path": "absolute_path ASC",
            "quality": "quality_score DESC, absolute_path ASC",
            "capture_time": "capture_time IS NULL, capture_time ASC, absolute_path ASC",
        }[sort]
        with self.database.connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM photos WHERE {where}", parameters
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT id, absolute_path, thumbnail_path, width, height, file_size,
                       capture_time, quality_score, blur_score, similarity_group,
                       auto_reject, reject_reason, metadata_json
                FROM photos WHERE {where}
                ORDER BY {order} LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()
        return AlbumPhotoListResponse(
            album=album,
            items=[_photo_summary(album_id, row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    def list_selections(
        self, album_id: str, *, limit: int, offset: int
    ) -> SelectionHistoryResponse:
        self.get_album(album_id)
        with self.database.connect() as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM selections WHERE album_id = ?", (album_id,)
                ).fetchone()[0]
            )
            rows = connection.execute(
                """
                SELECT id, album_id, raw_prompt, result_json, created_at
                FROM selections WHERE album_id = ?
                ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?
                """,
                (album_id, limit, offset),
            ).fetchall()
        items: list[SelectionHistoryItem] = []
        for row in rows:
            result = json.loads(row["result_json"] or "{}")
            items.append(
                SelectionHistoryItem(
                    id=row["id"],
                    album_id=row["album_id"],
                    prompt=row["raw_prompt"],
                    created_at=row["created_at"],
                    feasible=bool(result.get("feasible", False)),
                    selected_count=len(result.get("selected", [])),
                    solver=result.get("solver"),
                    solver_status=result.get("solver_status"),
                )
            )
        return SelectionHistoryResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )


_ALBUM_SUMMARY_SQL = """
SELECT a.id, a.name, a.source_path, a.created_at, a.indexed_at,
       (SELECT COUNT(*) FROM photos p WHERE p.album_id = a.id) AS photo_count,
       (SELECT COUNT(*) FROM photos p WHERE p.album_id = a.id AND p.auto_reject = 1)
           AS rejected_count,
       (SELECT COUNT(*) FROM photos p
        WHERE p.album_id = a.id AND p.embedding_path IS NOT NULL
          AND p.embedding_source_size = p.file_size
          AND p.embedding_source_mtime_ns = p.source_mtime_ns) AS embedded_count,
       (SELECT CASE WHEN COUNT(DISTINCT p.embedding_provider) = 1
                    THEN MAX(p.embedding_provider) ELSE NULL END
        FROM photos p WHERE p.album_id = a.id AND p.embedding_path IS NOT NULL
          AND p.embedding_source_size = p.file_size
          AND p.embedding_source_mtime_ns = p.source_mtime_ns)
           AS embedding_provider,
       (SELECT COUNT(*) FROM faces f JOIN photos p ON p.id = f.photo_id
        WHERE p.album_id = a.id) AS face_count,
       (SELECT COUNT(*) FROM selections s WHERE s.album_id = a.id) AS selection_count
FROM albums a
"""


def _album_summary(row: object) -> AlbumSummary:
    return AlbumSummary(
        id=row["id"],
        name=row["name"],
        source_path=row["source_path"],
        created_at=row["created_at"],
        indexed_at=row["indexed_at"],
        photo_count=int(row["photo_count"]),
        rejected_count=int(row["rejected_count"]),
        embedded_count=int(row["embedded_count"]),
        embedding_provider=row["embedding_provider"],
        face_count=int(row["face_count"]),
        selection_count=int(row["selection_count"]),
    )


def _photo_summary(album_id: str, row: object) -> PhotoSummary:
    metadata = json.loads(row["metadata_json"] or "{}")
    thumbnail_path = Path(row["thumbnail_path"])
    return PhotoSummary(
        id=row["id"],
        filename=Path(row["absolute_path"]).name,
        absolute_path=row["absolute_path"],
        thumbnail_path=str(thumbnail_path),
        thumbnail_url=f"/media/thumbnails/{album_id}/{thumbnail_path.name}",
        width=int(row["width"]),
        height=int(row["height"]),
        file_size=int(row["file_size"]),
        capture_time=row["capture_time"],
        quality_score=float(row["quality_score"]),
        blur_score=float(row["blur_score"]),
        similarity_group=row["similarity_group"],
        auto_reject=bool(row["auto_reject"]),
        reject_reason=row["reject_reason"],
        quality_flags=list(metadata.get("quality_flags", [])),
    )
