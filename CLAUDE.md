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

# spec三份文档生成
当用户说自己对于 droneImaging 的新增或改造功能的初步想法的时候，要用「生成spec三份文档的模板生成spec三份文档的模板工作流」，生成的三份文档放在根目录的docs目录下，每次提出初步想法就生成新的文件夹。

## 生成spec三份文档的模板工作流
```
我正在开发一个项目，叫 droneImaging，使用的编程语言是[python]。

每次我会提出一个初步的想法，需要你通过向我提问，帮助我澄清需求、挖掘边缘场景。澄清清楚后共创三份文档保存到项目根目录的docs下，生成对应的文件夹。

# 三份文档的角色与边界

## spec.md
回答：要解决什么问题、做哪些能力、不做哪些、什么算完成。
写：背景、目标用户、能力清单（一句话一条）、非功能要求、设计骨架、Out of Scope
不写：具体函数名 / 参数名 / 默认值 / 错误文本 / 行号 / SDK 类型名（这些是实现细节，spec 改一次就过期，维护爆炸）

## tasks.md
回答：按什么顺序做、每步动什么文件。
- 5~15 个任务，每个能在一次专注会话内完成
- 每个任务标注：影响文件、依赖任务、参考资料定位（精确到函数/行号都可以）
- 最后一定有「接入主流程」+「端到端验证」两个任务

## checklist.md
每一项必须可勾选、可观测，不许写「实现完整」「质量良好」。
- 把 spec 里被砍掉的具体值（错误文本、默认值、阈值）放进来作为验收项
- 写法举例：「`grep -r X` 返回 ≥3 条」「输入 Y 看到输出 Z」
- 至少一条端到端验收

我的初步想法是:[我的想法]
哪些留给后续章节不做:[不做什么]

请每轮只问几个关键点，不要一次性问完。使用 SDK 时务必通过 context7 MCP 等文档工具查最新 API(函数签名、参数顺序、类型名)，不要凭记忆写。

请开始你的提问。