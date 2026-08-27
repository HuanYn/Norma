from __future__ import annotations

import math
import sqlite3
import uuid
from datetime import datetime, timezone

from ai.retrieval import RetrievalService
from ai.schemas import (
    AlbumSearchRequest,
    EvaluationCandidate,
    EvaluationCandidateResponse,
    EvaluationQueryCreateRequest,
    EvaluationQueryListResponse,
    EvaluationQueryMetrics,
    EvaluationQuerySummary,
    EvaluationRunRequest,
    EvaluationRunResponse,
    RelevanceJudgmentBatchRequest,
    RelevanceJudgmentBatchResponse,
)
from ai.storage import Database


class EvaluationService:
    """Persist human labels and compute reproducible retrieval metrics."""

    def __init__(self, database: Database, retrieval: RetrievalService) -> None:
        self.database = database
        self.retrieval = retrieval

    def create_query(
        self, request: EvaluationQueryCreateRequest
    ) -> EvaluationQuerySummary:
        query_text = " ".join(request.query_text.split())
        notes = request.notes.strip() if request.notes else None
        query_id = uuid.uuid4().hex
        try:
            with self.database.connect() as connection:
                album = connection.execute(
                    "SELECT 1 FROM albums WHERE id = ?", (request.album_id,)
                ).fetchone()
                if album is None:
                    raise KeyError(f"album not found: {request.album_id}")
                connection.execute(
                    """
                    INSERT INTO evaluation_queries(id, album_id, query_text, notes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (query_id, request.album_id, query_text, notes),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"evaluation query already exists in album: {query_text}"
            ) from error
        return self.get_query(query_id)

    def get_query(self, query_id: str) -> EvaluationQuerySummary:
        with self.database.connect() as connection:
            row = connection.execute(
                _QUERY_SUMMARY_SQL + " WHERE q.id = ?", (query_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"evaluation query not found: {query_id}")
        return _query_summary(row)

    def list_queries(self, album_id: str) -> EvaluationQueryListResponse:
        with self.database.connect() as connection:
            album = connection.execute(
                "SELECT 1 FROM albums WHERE id = ?", (album_id,)
            ).fetchone()
            if album is None:
                raise KeyError(f"album not found: {album_id}")
            rows = connection.execute(
                _QUERY_SUMMARY_SQL
                + " WHERE q.album_id = ? ORDER BY q.created_at, q.id",
                (album_id,),
            ).fetchall()
        items = [_query_summary(row) for row in rows]
        return EvaluationQueryListResponse(items=items, total=len(items))

    def upsert_judgments(
        self, query_id: str, request: RelevanceJudgmentBatchRequest
    ) -> RelevanceJudgmentBatchResponse:
        photo_ids = [item.photo_id for item in request.judgments]
        if len(set(photo_ids)) != len(photo_ids):
            raise ValueError("each photo_id may appear only once per judgment batch")
        with self.database.connect() as connection:
            query = connection.execute(
                "SELECT album_id FROM evaluation_queries WHERE id = ?", (query_id,)
            ).fetchone()
            if query is None:
                raise KeyError(f"evaluation query not found: {query_id}")
            placeholders = ",".join("?" for _ in photo_ids)
            found = {
                row["id"]
                for row in connection.execute(
                    f"SELECT id FROM photos WHERE album_id = ? AND id IN ({placeholders})",
                    (query["album_id"], *photo_ids),
                )
            }
            missing = sorted(set(photo_ids) - found)
            if missing:
                raise ValueError(
                    "photos do not belong to the evaluation query album: "
                    + ", ".join(missing)
                )
            for item in request.judgments:
                connection.execute(
                    """
                    INSERT INTO relevance_judgments(
                        query_id, photo_id, relevance, annotator, updated_at
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(query_id, photo_id) DO UPDATE SET
                        relevance = excluded.relevance,
                        annotator = excluded.annotator,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (query_id, item.photo_id, item.relevance, request.annotator),
                )
            counts = connection.execute(
                """
                SELECT COUNT(*) AS judgment_count,
                       COALESCE(SUM(relevance > 0), 0) AS relevant_count
                FROM relevance_judgments WHERE query_id = ?
                """,
                (query_id,),
            ).fetchone()
        return RelevanceJudgmentBatchResponse(
            query_id=query_id,
            upserted_count=len(request.judgments),
            judgment_count=int(counts["judgment_count"]),
            relevant_count=int(counts["relevant_count"]),
        )

    def candidates(
        self, query_id: str, *, limit: int = 50
    ) -> EvaluationCandidateResponse:
        if limit < 1 or limit > 50:
            raise ValueError("candidate limit must be between 1 and 50")
        query = self.get_query(query_id)
        search = self.retrieval.search(
            AlbumSearchRequest(
                album_id=query.album_id,
                query=query.query_text,
                limit=limit,
            )
        )
        with self.database.connect() as connection:
            judgments = {
                row["photo_id"]: (int(row["relevance"]), row["annotator"])
                for row in connection.execute(
                    """
                    SELECT photo_id, relevance, annotator
                    FROM relevance_judgments WHERE query_id = ?
                    """,
                    (query_id,),
                )
            }
        items = []
        for rank, match in enumerate(search.matches, start=1):
            judgment = judgments.get(match.photo_id)
            items.append(
                EvaluationCandidate(
                    rank=rank,
                    photo_id=match.photo_id,
                    filename=match.filename,
                    thumbnail_url=match.thumbnail_url,
                    score=match.score,
                    relevance=judgment[0] if judgment else None,
                    annotator=judgment[1] if judgment else None,
                )
            )
        return EvaluationCandidateResponse(
            query=query,
            provider=search.provider,
            items=items,
        )

    def run(
        self, album_id: str, request: EvaluationRunRequest
    ) -> EvaluationRunResponse:
        cutoffs = sorted(set(request.cutoffs))
        if not cutoffs or cutoffs[0] < 1 or cutoffs[-1] > 50:
            raise ValueError("evaluation cutoffs must be between 1 and 50")
        listed = self.list_queries(album_id).items
        if request.query_ids is not None:
            requested = set(request.query_ids)
            selected = [query for query in listed if query.id in requested]
            missing = sorted(requested - {query.id for query in selected})
            if missing:
                raise ValueError(
                    "evaluation queries do not belong to album: " + ", ".join(missing)
                )
        else:
            selected = listed
        if not selected:
            raise ValueError("album has no evaluation queries to run")

        metrics: list[EvaluationQueryMetrics] = []
        skipped = 0
        for query in selected:
            grades = self._judgments(query.id)
            if not grades:
                skipped += 1
                continue
            ranked = self.retrieval.search(
                AlbumSearchRequest(
                    album_id=album_id,
                    query=query.query_text,
                    limit=cutoffs[-1],
                )
            )
            metrics.append(
                _query_metrics(
                    query.id,
                    query.query_text,
                    [item.photo_id for item in ranked.matches],
                    grades,
                    cutoffs,
                )
            )
        if not metrics:
            raise ValueError("no selected evaluation query has human judgments")

        run_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        response = EvaluationRunResponse(
            run_id=run_id,
            album_id=album_id,
            provider=self.retrieval.provider.name,
            cutoffs=cutoffs,
            query_count=len(metrics),
            skipped_query_count=skipped,
            macro_mrr=_mean([item.reciprocal_rank for item in metrics]),
            macro_precision_at=_macro(metrics, "precision_at", cutoffs),
            macro_recall_at=_macro(metrics, "recall_at", cutoffs),
            macro_ndcg_at=_macro(metrics, "ndcg_at", cutoffs),
            queries=metrics,
            created_at=created_at,
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs(
                    id, album_id, embedding_provider, request_json, result_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    album_id,
                    self.retrieval.provider.name,
                    request.model_dump_json(),
                    response.model_dump_json(),
                    created_at,
                ),
            )
        return response

    def get_run(self, run_id: str) -> EvaluationRunResponse:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM evaluation_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"evaluation run not found: {run_id}")
        return EvaluationRunResponse.model_validate_json(row["result_json"])

    def _judgments(self, query_id: str) -> dict[str, int]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT photo_id, relevance FROM relevance_judgments WHERE query_id = ?",
                (query_id,),
            ).fetchall()
        return {row["photo_id"]: int(row["relevance"]) for row in rows}


_QUERY_SUMMARY_SQL = """
SELECT q.id, q.album_id, q.query_text, q.notes, q.created_at, q.updated_at,
       (SELECT COUNT(*) FROM relevance_judgments j WHERE j.query_id = q.id)
           AS judgment_count,
       (SELECT COUNT(*) FROM relevance_judgments j
        WHERE j.query_id = q.id AND j.relevance > 0) AS relevant_count
FROM evaluation_queries q
"""


def _query_summary(row: sqlite3.Row) -> EvaluationQuerySummary:
    return EvaluationQuerySummary(
        id=row["id"],
        album_id=row["album_id"],
        query_text=row["query_text"],
        notes=row["notes"],
        judgment_count=int(row["judgment_count"]),
        relevant_count=int(row["relevant_count"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _query_metrics(
    query_id: str,
    query_text: str,
    ranked_photo_ids: list[str],
    grades: dict[str, int],
    cutoffs: list[int],
) -> EvaluationQueryMetrics:
    relevant_count = sum(grade > 0 for grade in grades.values())
    ranked_grades = [grades.get(photo_id, 0) for photo_id in ranked_photo_ids]
    first_relevant = next(
        (rank for rank, grade in enumerate(ranked_grades, start=1) if grade > 0),
        None,
    )
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    ndcg: dict[str, float] = {}
    ideal = sorted(grades.values(), reverse=True)
    for cutoff in cutoffs:
        key = str(cutoff)
        observed = ranked_grades[:cutoff]
        found = sum(grade > 0 for grade in observed)
        precision[key] = round(found / cutoff, 6)
        recall[key] = round(found / relevant_count, 6) if relevant_count else 0.0
        dcg = _dcg(observed)
        ideal_dcg = _dcg(ideal[:cutoff])
        ndcg[key] = round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0
    return EvaluationQueryMetrics(
        query_id=query_id,
        query_text=query_text,
        judgment_count=len(grades),
        relevant_count=relevant_count,
        ranked_photo_ids=ranked_photo_ids,
        relevance_by_photo=grades,
        reciprocal_rank=round(1 / first_relevant, 6) if first_relevant else 0.0,
        precision_at=precision,
        recall_at=recall,
        ndcg_at=ndcg,
    )


def _dcg(grades: list[int]) -> float:
    return sum(
        ((2**grade) - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6)


def _macro(
    metrics: list[EvaluationQueryMetrics], field: str, cutoffs: list[int]
) -> dict[str, float]:
    return {
        str(cutoff): _mean([getattr(item, field)[str(cutoff)] for item in metrics])
        for cutoff in cutoffs
    }
