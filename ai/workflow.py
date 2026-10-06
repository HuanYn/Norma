"""Photo-first demo orchestration. Originals are read-only; curation is a view."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import uuid

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from ai.demo_api import _image_bytes, _parser
from ai.index.embedding import EmbeddingProviderUnavailableError
from ai.selection.learned_quality import snapshot
from ai.selection.parser import COUNT_PATTERNS, parse_selection_prompt, extract_category_counts
from ai.selection.structured import StructuredIntentError, StructuredProviderError
from ai.schemas import SelectionRequest


class CurationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    min_aesthetic: float = Field(default=5.0, ge=1, le=10)
    min_technical: float = Field(default=40.0, ge=0, le=100)


class PickRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=1000)
    use_preference_memory: bool = False
    allow_proxy_memory: bool = False
    use_language_model: bool = False
    allow_cloud: bool = False
    default_count: int = Field(default=9, ge=1, le=50)
    required_photo_ids: list[str] = Field(default_factory=list, max_length=50)
    representative_photo_ids: list[str] = Field(default_factory=list, max_length=2)


class SelectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=1000)
    curation_id: str
    required_photo_ids: list[str] = Field(default_factory=list, max_length=50)


REPRESENTATIVE_CLAUSE = re.compile(
    r"(?:两|2)个不同的人(?:物)?(?:最少|至少)(?:分别|各)(?:有)?(?:一|1)张"
)


def resolve_representatives(request, allowed):
    """Use explicit user-chosen representative photos, never facial identity inference."""
    matches = list(REPRESENTATIVE_CLAUSE.finditer(request.prompt))
    representatives = request.representative_photo_ids
    if matches:
        if len(representatives) != 2 or len(set(representatives)) != 2:
            raise ValueError("请在照片上各指定一张“人物A代表照”和“人物B代表照”；两张人像不等于两个不同的人。")
    elif representatives:
        raise ValueError("代表照需要对应的两位人物选片要求，请检查本次提示词。")
    if not set(representatives) <= allowed:
        raise ValueError("代表照不在本次美学保留范围，请调整门槛或重新选择。")
    return REPRESENTATIVE_CLAUSE.sub("", request.prompt), {
        "source": "user-selected-representative-photos-not-face-recognition",
        "source_clauses": [m.group() for m in matches],
        "photo_ids": representatives,
        "identity_verified_by_model": False,
    } if matches else None


def curate(database, assessment, album_id, request, provider=None):
    evidence = snapshot(assessment, album_id)
    with database.connect() as c:
        rows = c.execute(
            "SELECT * FROM photos WHERE album_id=? ORDER BY id", (album_id,)
        ).fetchall()
    if any(r["phash"] is None or r["dhash"] is None for r in rows):
        raise ValueError("先完成基础质量与相似分析，再筛选最佳照片。")
    groups = {}
    for row in rows:
        groups.setdefault(row["similarity_group"] or "photo:" + row["id"], []).append(
            row
        )
    kept, hidden = [], []

    def utility(row):
        score = evidence["items"][row["id"]]
        return 0.65 * (score["aesthetic_quality"] - 1) / 9 + 0.35 * max(
            0, min(1, score["technical_quality"] / 100)
        )

    for group, members in sorted(groups.items()):
        ranked = sorted(members, key=lambda r: (-utility(r), r["id"]))
        winner = None
        for row in ranked:
            score = evidence["items"][row["id"]]
            reasons = []
            if row["auto_reject"]:
                reasons.append("基础技术质量未通过")
            if score["aesthetic_quality"] < request.min_aesthetic:
                reasons.append("美学分低于门槛")
            if score["technical_quality"] < request.min_technical:
                reasons.append("技术质量低于门槛")
            if not reasons and winner is None:
                winner = row["id"]
            elif not reasons:
                reasons.append("相似组已保留综合评分更高的一张")
            photo = {
                "photo_id": row["id"],
                "filename": Path(row["absolute_path"]).name,
                "thumbnail_url": f"/media/thumbnails/{album_id}/{Path(row['thumbnail_path']).name}",
                "aesthetic_quality": score["aesthetic_quality"],
                "technical_quality": score["technical_quality"],
                "curation_score": utility(row),
                "group": group,
                "reasons": reasons,
                "source_sha256": score["source_sha256"],
            }
            (hidden if reasons else kept).append(photo)
    kept.sort(key=lambda p: (-p["curation_score"], p["photo_id"]))
    visual_evidence = None
    if provider is not None:
        from ai.selection.visual_duplicates import fold_similar
        kept, visually_hidden, visual_evidence = fold_similar(kept, rows, provider)
        hidden.extend(visually_hidden)
    identity = {
        "assessment": evidence["snapshot_sha256"],
        "thresholds": request.model_dump(),
        "visual_duplicates": visual_evidence,
        "groups": [
            [
                r["id"],
                r["similarity_group"],
                bool(r["auto_reject"]),
                r["phash"],
                r["dhash"],
            ]
            for r in rows
        ],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "album_id": album_id,
        "snapshot_sha256": digest,
        "assessment_provider": evidence["provider"],
        "thresholds": request.model_dump(),
        "total": len(rows),
        "kept": kept,
        "hidden": hidden,
        "group_count": sum(len(v) > 1 for v in groups.values()),
        "originals_deleted": 0,
        "algorithm": "musiq-dual-best-per-near-duplicate-v1" + ("+openclip-spatial-v1" if visual_evidence else ""),
        "visual_duplicates": visual_evidence,
        "similarity_method": "hash groups + learned visual similarity with spatial verification; greedy quality-first non-transitive suppression" if visual_evidence else "conservative pHash<=7 AND dHash<=9 connected groups; not semantic topic deduplication",
    }


def normalize_query(prompt, default_count, available):
    # Narrow visible typo correction, not an LLM inference claim.
    normalized = re.sub(r"((?:挑|选|找)\s*\d+\s*)长(?=\s*\S)", r"\1张", prompt.strip())
    normalized = re.sub(r"(挑|选|找)出(?=\s*\d+\s*张)", r"\1", normalized)
    # Normalize only a command's explicit Chinese total; retain the original in
    # provenance. Category numerals remain exact original evidence spans.
    from ai.selection.structured import _grammar_text
    normalized = re.sub(
        r"(?:选|挑|找|保留)(?:出)?\s*[零一二两三四五六七八九十]+\s*张",
        lambda m: _grammar_text(m.group()).replace("挑出", "挑").replace("找出", "找"),
        normalized,
    )
    categories, residual = extract_category_counts(normalized)
    explicit = any(p.search(residual) for p in COUNT_PATTERNS)
    # A noun phrase such as "挑出最适合这趟旅游展示的9张照片" has an
    # explicit total too. Compile this bounded clause grammar;
    # preserve the original wording for semantic and hard-condition validation.
    descriptive_count = re.search(
        r"(?:挑|选|找|保留)(?:出)?[^\d，,。;；\n]{1,60}的\s*(\d+)\s*张(?:照片|图片)(?=$|[\s，,。;；！!])",
        residual,
    )
    if descriptive_count:
        literal_counts = {int(m.group(1)) for p in COUNT_PATTERNS for m in p.finditer(residual)}
        if literal_counts and literal_counts != {int(descriptive_count.group(1))}:
            raise ValueError("存在冲突的选片总数，请明确一个总数")
    if not explicit and descriptive_count:
        normalized = f"选{descriptive_count.group(1)}张。{normalized}"
        explicit = True
    if not explicit and categories:
        normalized = f"选{sum(categories.values())}张。{normalized}"
        explicit = True
    intent = parse_selection_prompt(normalized)
    if not explicit:
        count = min(default_count, available)
        if count == 0:
            raise ValueError("没有达到门槛的照片，请先调整美学评估门槛。")
        normalized = f"选{count}张。{normalized}"
        intent = replace(intent, target_count=count)
    if intent.target_count > available:
        raise ValueError(
            f"筛选后只有{available}张可用照片，不能满足{intent.target_count}张；请调整门槛或数量。"
        )
    return normalized, intent, not explicit


def create_workflow_router(get_database, get_assessment, get_selection, get_settings):
    router = APIRouter(prefix="/workflow", tags=["photo-first-demo"])

    def database():
        db = get_database()
        with db.connect() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS workflow_curations_v1 (id TEXT PRIMARY KEY, album_id TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            c.execute("CREATE TABLE IF NOT EXISTS workflow_selection_drafts_v1 (album_id TEXT PRIMARY KEY, draft_json TEXT NOT NULL)")
        return db

    def load(curation_id):
        db = database()
        with db.connect() as c:
            row = c.execute(
                "SELECT result_json FROM workflow_curations_v1 WHERE id=?",
                (curation_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "筛选结果不存在")
        return json.loads(row["result_json"])

    def checked(curation_id):
        stored = load(curation_id)
        current = curate(
            database(),
            get_assessment(),
            stored["album_id"],
            CurationRequest(**stored["thresholds"]),
            get_selection().provider,
        )
        if current["snapshot_sha256"] != stored["snapshot_sha256"]:
            raise ValueError("照片或评分已变化，请重新评估筛选。")
        return stored

    @router.post("/albums/{album_id}/curations")
    def create(album_id: str, request: CurationRequest):
        try:
            db = database()
            result = curate(db, get_assessment(), album_id, request, get_selection().provider)
            result["id"] = uuid.uuid4().hex
            with db.connect() as c:
                c.execute(
                    "INSERT INTO workflow_curations_v1(id,album_id,result_json) VALUES (?,?,?)",
                    (result["id"], album_id, json.dumps(result)),
                )
            return result
        except KeyError:
            raise HTTPException(404, "相册不存在") from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @router.get("/albums/{album_id}/selection-draft")
    def get_draft(album_id: str):
        with database().connect() as c:
            row = c.execute("SELECT draft_json FROM workflow_selection_drafts_v1 WHERE album_id=?", (album_id,)).fetchone()
        return json.loads(row[0]) if row else None

    @router.post("/albums/{album_id}/selection-draft")
    def save_draft(album_id: str, request: SelectionDraft):
        try:
            current = checked(request.curation_id)
            if current["album_id"] != album_id or not set(request.required_photo_ids) <= {p["photo_id"] for p in current["kept"]}:
                raise ValueError("草稿指定照片必须来自此相册的当前保留范围。")
            draft = request.model_dump() | {"revision": uuid.uuid4().hex}
            with database().connect() as c:
                c.execute("INSERT INTO workflow_selection_drafts_v1 VALUES (?,?) ON CONFLICT(album_id) DO UPDATE SET draft_json=excluded.draft_json", (album_id, json.dumps(draft)))
            return draft
        except (KeyError, ValueError) as e:
            raise HTTPException(409, str(e)) from None

    @router.get("/curations/{curation_id}")
    def get(curation_id: str):
        try:
            return checked(curation_id)
        except (ValueError, KeyError) as e:
            raise HTTPException(409, str(e)) from None

    @router.post("/curations/{curation_id}/select")
    def select(curation_id: str, request: PickRequest):
        try:
            if request.allow_proxy_memory and not request.use_preference_memory:
                raise ValueError("助手代理偏好需要同时启用偏好记忆。")
            current = checked(curation_id)
            resolved_prompt, representative_evidence = resolve_representatives(request, {p["photo_id"] for p in current["kept"]})
            prompt, intent, defaulted = normalize_query(
                resolved_prompt, request.default_count, len(current["kept"])
            )
            provenance = {
                "workflow": "photo-first-v1",
                "original_prompt": request.prompt,
                "manual_representatives": representative_evidence,
                "effective_prompt": prompt,
                "default_count_applied": defaulted,
                "curation_id": curation_id,
                "curation_sha256": current["snapshot_sha256"],
                "text_interpretation": "bounded-count-grammar",
                "ranking": "pretrained-OpenCLIP+MUSIQ",
                "preference_profile": "assistant-proxy-enabled" if request.allow_proxy_memory else "observed-feedback-only" if request.use_preference_memory else "none",
            }
            semantic_query = None
            if request.use_language_model:
                settings = get_settings()
                if (
                    settings.vlm_provider == "openai-compatible"
                    and not request.allow_cloud
                ):
                    raise ValueError(
                        "请确认允许本次文字发送到已配置的云端模型；不会发送相册原图。"
                    )
                parsed = _parser(settings).parse(prompt)
                intent = parsed.to_selection_intent()
                semantic_query = parsed.semantic_query
                provenance.update(
                    text_interpretation="language-model",
                    language_model=parsed.provenance.model_dump(),
                )
            result = get_selection().select(
                SelectionRequest(
                    album_id=current["album_id"],
                    prompt=prompt,
                    subset_photo_ids=[p["photo_id"] for p in current["kept"]],
                    use_learned_quality=True,
                    use_preference_memory=request.use_preference_memory,
                    allow_proxy_memory=request.allow_proxy_memory,
                    required_photo_ids=list(dict.fromkeys(request.required_photo_ids + request.representative_photo_ids)),
                ),
                intent=intent,
                semantic_query=semantic_query,
                intent_provenance=provenance,
                collection_diversity=True,
            )
            checked(curation_id)  # Do not present a stale candidate view as current.
            return result
        except (ValueError, KeyError, StructuredIntentError) as e:
            raise HTTPException(409, str(e)) from None
        except (EmbeddingProviderUnavailableError, StructuredProviderError):
            raise HTTPException(
                503, "选片模型不可用；没有切换到规则排序，请检查模型配置。"
            ) from None

    @router.get("/selections/{selection_id}/images/{photo_id}")
    def selected_image(selection_id: str, photo_id: str):
        try:
            selection = get_selection().get(selection_id)
            selected = next(
                (p for p in selection.selected if p.photo_id == photo_id), None
            )
            if selected is None:
                raise HTTPException(404, "照片不在当前选片结果中")
            if not selection.intent_provenance or not selection.intent_provenance.get(
                "curation_id"
            ):
                raise HTTPException(409, "请先从Demo评估结果选片")
            current = checked(selection.intent_provenance["curation_id"])
            if photo_id not in {p["photo_id"] for p in current["kept"]}:
                raise HTTPException(409, "筛选内容已变化")
            content = _image_bytes(database(), selection.album_id, photo_id)
            checked(current["id"])
            return Response(
                content=content,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(409, str(e)) from None

    return router
