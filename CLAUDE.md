# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**droneImaging** is a FastAPI microservice (port 8002) that provides drone imagery ingestion, semantic search, and map tile serving for the `phenomicsAgentCC` phenotyping agent. The agent calls this service's HTTP API via Function Calling; the frontend renders imagery on a Leaflet map using the returned tile URLs and bounding boxes.

Core ingestion pipeline (async, per-image): upload raw GeoTIFF → parse metadata → convert to COG → generate thumbnail → VLM description → text embedding → Milvus vector index. Status progresses: `uploaded → parsing → converting → describing → embedding → ready` (or `error`).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (from project root)
uvicorn src.main:app --port 8002 --reload

# Health check
curl http://localhost:8002/health
```

There is **no test suite, linter, or formatter** configured yet (no pyproject.toml, pytest, ruff, or mypy). `requirements.txt` contains only runtime dependencies.

## Configuration

All external service connection info lives in `config.yaml` (git-ignored — copy from `config.yaml.example`). The config is loaded once into a global Pydantic singleton via `src/config.py`:

- `load_config()` reads `config.yaml` and caches the `AppConfig`; `get_config()` returns it (auto-loads if needed).
- `BASE_DIR` = project root (where `config.yaml` lives).
- `get_public_base_url()` resolves the externally-reachable base URL for tile/thumbnail links: env `DRONE_PUBLIC_BASE_URL` wins, else `http://{host}:{port}` (replaces `0.0.0.0` with `localhost`). Set the env var when behind Docker/reverse proxy.

**Secrets handling**: `config.yaml`, `.env`, `test_image/`, and `logs/` are git-ignored. Never commit real credentials.

## Architecture

### Layered structure (`src/`)

| Layer | Path | Role |
|-------|------|------|
| Entry | `main.py` | FastAPI app, lifespan wires DB + MinIO + Milvus init, mounts routers |
| Config | `config.py` | Pydantic settings from `config.yaml` |
| DB | `database.py` | SQLAlchemy async engine + session factory, `init_db()` auto-creates tables |
| ORM models | `models/` | `Image` (imagery registry), `FetchSource` (P1 reserved) |
| Pydantic schemas | `schemas/image.py` | Request/response models |
| Routers | `routers/` | `images` (CRUD/upload/thumbnail), `search` (semantic), `tiles` (XYZ) |
| Services | `services/` | Business logic (see below) |
| Utils | `utils/thumbnail.py` | Thumbnail generation |

### Service modules

- **`pipeline.py`** — orchestrates the async ingestion flow. Uses a global `asyncio.Semaphore(max_concurrent_tasks)` to bound concurrent GDAL-heavy work. CPU-bound steps (COG convert, thumbnail) run via `loop.run_in_executor`. Each step commits independently and updates `status`; any failure sets `status=error` + `error_message`. Temp files cleaned in `finally`.
- **`geotiff.py`** — `parse_geotiff()` extracts bbox/CRS/resolution/bands via rasterio into `GeoTiffMetadata`.
- **`cog.py`** — `convert_to_cog()` uses rasterio's COG driver (no external `gdal_translate` CLI) with configurable blocksize + overview levels.
- **`storage.py`** — MinIO client wrapper: `upload_file`, `upload_bytes`, `download_file`, `get_presigned_url`, `file_exists`, `init_storage()` (creates buckets at startup).
- **`vlm.py`** — `describe_image()` sends thumbnail (base64) + prompt to an OpenAI-compatible VLM API (Qwen-VL default), with exponential-backoff retries. Prompt loaded from `prompts/vlm_describe.txt`.
- **`embedding.py`** — `build_embedding_text()` concatenates VLM description + business metadata; `embed_text()` / `embed_batch()` call the embedding API. **Auth note**: embedding uses `X-API-Key` header (not Bearer), and accepts both custom `{"embeddings": [...]}` and OpenAI `{"data": [...]}` response formats.
- **`milvus_client.py`** — manages the `image_vectors` collection (HNSW index, COSINE metric). Schema: `id` (PK), `text_vector`, `image_vector` (reserved, currently zero-filled), plus scalar filter fields `task_id`/`field_name`/`survey_stage`.

### Routers

- **`images.py`** — `POST /api/images/upload` (multipart, validates `.tif/.tiff`), `GET /api/images/{id}`, `GET /api/images` (filtered + paginated), `GET /api/images/{id}/status`, `GET /api/images/{id}/thumbnail`. Upload saves a temp file, then fires `asyncio.create_task(_run_pipeline(...))` so the response returns immediately while processing happens in the background with its own session.
- **`search.py`** — `POST /api/images/search`: query → embedding → Milvus search → backfill metadata from PostgreSQL → assemble results with tile/thumbnail URLs. Supports scalar filters.
- **`tiles.py`** — `GET /api/tiles/{image_id}/{z}/{x}/{y}.png`: reads COG from MinIO (via presigned URL), renders a 256×256 PNG tile. Handles the case where the tile bounds exceed the image bounds by rendering only the intersection into a transparent canvas (so imagery scales correctly with zoom). Returns transparent tile on render failure for better frontend UX.

### Key design points

- **Async throughout**: SQLAlchemy async sessions; VLM/embedding calls via `httpx.AsyncClient`. CPU-bound rasterio work offloaded to thread pool.
- **Status-driven**: the `Image.status` field is the source of truth for pipeline progress and is exposed via the status endpoint.
- **URL generation**: tile/thumbnail URLs are synthesized at response time from `get_public_base_url()`, not stored in DB.
- **P1 reserved**: `FetchSource` model and the fetch flow are stubbed (model + table exist, no router yet). `image_vector` in Milvus is reserved (zero-filled).
- **VLM prompt** (`prompts/vlm_describe.txt`) now requests **structured JSON** (crop type, growth stage, canopy coverage, anomalies, management traces, image quality, shooting angle) rather than free text — note this diverges from the original `spec.md` which described free-text output.

## API base path

All routes are mounted under `/api`: `/api/images/*`, `/api/images/search`, `/api/tiles/*`. Health check is at both `/health` and `/api/health` (the latter for frontend proxy compatibility).
