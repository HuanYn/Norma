# Grounded multimodal RAG

Norma's RAG endpoint combines learned multilingual retrieval with a local
vision-language model:

1. multilingual OpenCLIP embeds the user query and retrieves a bounded Top-K;
2. the compatible 67D preference posterior, when present, contributes its
   learned residual to that ranking; otherwise ranking remains exact cosine;
3. each selected original is frozen as an immutable byte snapshot;
4. local Qwen3-VL generates structured claims and citations from those pixels;
5. the server constructs the canonical answer and provenance itself;
6. citation, allow-list, content-digest, path-leak, and provenance checks fail
   closed before an immutable audit row is written.

This is a real retrieval-to-generation path. It is not DPO, SFT, LoRA, or
model fine-tuning: OpenCLIP and Qwen3-VL are frozen at inference time. The
validator proves referential integrity (which evidence was supplied and which
allowed photo IDs were cited); it does **not** prove semantic entailment between
each generated claim and the cited pixels.

## Install and local model

Install the multimodal runtime once:

```powershell
python -m pip install -e ".[dev,selection,multimodal]"
```

Run Norma on its default loopback interface. The website/API has no
authentication, so this endpoint must not be exposed to an untrusted network:

```powershell
python -m ai web
```

The generation model is the official Apache-2.0
[`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
snapshot. Norma never downloads it during an API request. Put the complete
snapshot at the default location:

```text
.norma/data/models/qwen3-vl/Qwen3-VL-2B-Instruct-modelscope
```

Provision that exact directory explicitly from the repository root. The command
downloads only revision `89644892e4d85e24eaac8bacfd4f463576704203`, streams
each asset through the checked-in size/SHA-256 contract, and atomically exposes
the directory only after all 11 files pass:

```powershell
python scripts/install_qwen3vl_model.py
```

If the fixed revision is already present in the Hugging Face cache, reconstruct
and verify the directory without network access:

```powershell
python scripts/install_qwen3vl_model.py --offline
```

An existing valid target is verified and reused. An incomplete, modified, or
extra-file target fails closed and is never overwritten automatically.

or configure an absolute directory before startup:

```powershell
$env:NORMA_VLM_MODEL_PATH = "D:\NormaModels\Qwen3-VL-2B-Instruct-modelscope"
$env:NORMA_VLM_MAX_NEW_TOKENS = "256"
python -m ai web
```

The runtime is CPU-only and uses `local_files_only=True` and
`trust_remote_code=False`. It checks the runtime assets against a
version-controlled manifest containing the pinned revision, sizes, and full
SHA-256 digests. It performs full manifest verification before and after the
first model load, and does not publish the in-process model until both checks
succeed. Later requests verify the loaded snapshot's file metadata without
re-hashing the multi-gigabyte model for every generation. The provider
fingerprint binds the manifest, preprocessing contract, token budget, Python,
and the directly used Transformers/Torch/vision/tokenizer/template/asset-reader
runtime versions, plus the native numeric-threading contract. It also binds a
SHA-256 of the system prompt, strict
claims/citations output schema, JSON serialization, and path-redaction contract;
changing real model input semantics therefore creates a new generation identity.
Missing or modified assets return HTTP 503 instead of falling
back to a Hub or another model. This identity does not promise bitwise equality
across different CPU instruction sets or operating-system math libraries.

## Prepare an album

Opening a folder only creates the local catalog and thumbnails. Click
**语义索引** before calling RAG. Schema v13 binds every embedding to the full
SHA-256 of the source bytes; embeddings created by an older schema are shown as
stale and must be recomputed once.

The background embedding job hashes each original before and after inference,
then commits the vector, provider, source stat, and content digest in one guarded
transaction. A later catalog scan invalidates the vector if bytes changed even
when an attacker or copy tool restored the old size and mtime.

The interactive search path does not hash every original on every request.
Instead, the background embedding job records the source digest and the RAG
boundary re-hashes only the selected Top-K originals before generation. This
prevents a stale vector from ranking one image while the VLM receives different
pixels, without turning each search into a full-album disk scan.

## Call the endpoint

```powershell
$albumId = "YOUR_ALBUM_ID"
$body = @{
  query = "哪些照片里有人戴眼镜？请给出有证据的简短说明"
  top_k = 3
  user_id = "local"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/albums/$albumId/rag" `
  -ContentType "application/json" `
  -Body $body
```

`top_k` is limited to 1–6. The response contains the retrieval result, canonical
answer, claim list, claim-to-photo citations, and retrieval/generation
provenance digests. Successful runs are appended to SQLite `rag_runs`; update,
delete, and replace-style overwrites are rejected. Schema v14 stores this audit
table `WITHOUT ROWID`, so an `INSERT OR REPLACE` cannot bypass immutability by
targeting an old hidden rowid with a new run ID. The audit stores bounded
evidence descriptors and content digests, not another copy of the source image
bytes. Every response also fixes these two fields:

```json
{
  "validation_level": "citation-referential-only",
  "semantic_entailment_verified": false
}
```

Expected errors:

- `404`: album is missing or empty;
- `409`: source, embedding, candidate universe, preference model, or ranking
  inputs changed during the run;
- `413`: selected evidence exceeds a bounded local-VLM byte or pixel budget;
- `422`: no usable evidence or model output failed the strict claim/citation
  contract;
- `503`: pinned local Qwen3-VL assets or required runtime dependencies are not
  available.

## Trust and resource boundaries

- Raw source paths and image-loader paths are excluded from the model-visible
  evidence. Windows, UNC, POSIX, home, cache, temporary, and traversal-style
  paths are redacted on input and rejected on output.
- The model may return only `claims` and `citations`. Extra top-level fields,
  duplicate JSON keys at any nesting depth, unknown photo IDs, uncited claims,
  duplicate claims, or path disclosure fail closed.
- The server—not the model—constructs `answer` and `provenance`, so the model
  cannot forge those fields.
- Candidate provenance includes every embedding digest, source-content binding,
  quality/reject input, and compatible preference-model snapshot used for
  ranking. It is rechecked after generation.
- Local CPU generation is serialized before original-image bytes are retained.
  One request is limited to Top-K 1–6, 128 MiB encoded evidence, 64 megapixels
  per decoded image, 96 megapixels decoded in aggregate, and 1 MiB per cached
  embedding file. Before Qwen preprocessing, Norma deterministically derives
  patch-aligned model inputs under a 3,840-visual-token budget; after processing,
  it recomputes the actual token count from `image_grid_thw` and rejects anything
  above the 4,096-token hard limit with `413`. Resizing affects only the model
  input derivative: the frozen evidence bytes and their SHA-256 identity remain
  unchanged.
- Photos, evidence bytes, embeddings, prompts, and results remain on the local
  machine. Do not expose the unauthenticated local web service to an untrusted
  network.

## Reproducible smoke

The public-image smoke script exercises real local model loading, image input,
strict structured output, citation validation, canonical answer construction,
and provenance checks:

```powershell
python scripts/download_public_smoke_image.py
python figures/benchmark_qwen3vl_grounded_smoke.py `
  --model-path .norma/data/models/qwen3-vl/Qwen3-VL-2B-Instruct-modelscope `
  --image .norma/public-smoke/gothic-architecture-banner.jpg `
  --output figures/qwen3vl_grounded_smoke_20260829.json `
  --max-new-tokens 256 `
  --repeats 2
```

When the command completes, its report records full model/image hashes, cold
and warm latency, peak process RSS when `psutil` is available,
manifest-verification cost, strict-validation status, and deterministic replay.
No performance number is asserted in this document before that artifact has
passed review. A one-image smoke is an engineering check, not a retrieval
benchmark or evidence for semantic-entailment accuracy.

## What this does and does not learn

OpenCLIP and Qwen3-VL are pretrained learned models but remain frozen in Norma.
The trainable component is the small 67D Bayesian contextual preference
posterior built from local pairwise feedback. RAG consumes that posterior only
through the retrieval score; it does not update Qwen3-VL. Therefore this feature
must not be described as DPO, SFT, LoRA, reinforcement learning, or end-to-end
multimodal fine-tuning. Citation allow-list checks establish referential
integrity, not semantic entailment or factual correctness.
