# 无人机影像服务 — 需求规格说明书

## 1. 项目概述

### 1.1 定位

独立部署的 FastAPI 微服务（代号 droneImaging），为 phenomicsAgentCC 表型智能体提供无人机影像的入库、语义检索和地图展示能力。Agent 通过 Function Calling 调用本服务的 HTTP API，前端通过 Agent 返回的结构化数据在地图上渲染影像。

### 1.2 核心链路

```
用户自然语言 → phenomicsAgentCC Agent → Function Calling → droneImaging API
    → 返回影像列表(元数据 + 瓦片URL + bbox)
    → Agent 组装回复 → 前端 Leaflet 地图叠加影像图层
```

### 1.3 范围与优先级

| 优先级 | 功能 | 说明 |
|--------|------|------|
| P0 | 影像入库（上传） | 开发测试用，手动填写业务字段 |
| P0 | GeoTIFF 元数据解析 | rasterio 提取 bbox/crs/分辨率等 |
| P0 | COG 转换 + 动态瓦片 | 前端流畅浏览 700-800MB 大图 |
| P0 | VLM 语义描述 | Qwen-VL API 自由描述（可配置） |
| P0 | 向量检索 | text embedding → Milvus 相似度搜索 |
| P0 | 前端地图展示 | Leaflet + 天地图底图 + 影像叠加 |
| P1 | 外部存储拉取 | 用户触发，从外部 MinIO/NAS 按元数据拉取 |
| P2 | 试验小区划分 | VLM/CV 识别田块边界 + 自然语言交互 |

### 1.4 不包含

- 用户认证（复用 phenomicsAgentCC 的 JWT 体系，本服务内网部署不做独立鉴权）
- 影像拼接/正射校正（假设入库的 GeoTIFF 已完成预处理）
- 试验小区划分（P2，本期仅预留数据模型）

---

## 2. 技术栈

| 层次 | 选型 | 说明 |
|------|------|------|
| 后端框架 | FastAPI + Uvicorn | 异步，端口 8002 |
| GeoTIFF 处理 | rasterio + GDAL | 元数据解析、COG 转换 |
| 瓦片服务 | titiler.core | 内嵌 FastAPI 路由，从 MinIO 读 COG 按需出瓦片 |
| 对象存储 | MinIO | 原始 TIFF + COG + 缩略图 |
| 关系数据库 | PostgreSQL + SQLAlchemy (async) | 业务元数据、影像注册表 |
| 向量数据库 | Milvus (pymilvus) | 语义检索 |
| Embedding | BGE-large-zh / GTE（可配置） | 文本向量化 |
| VLM | Qwen-VL API（可配置切换本地模型） | 影像结构化描述 |
| 前端地图 | Leaflet + 天地图瓦片 | 影像叠加展示 |
| 前端框架 | Vue 3 (集成在 phenomicsAgentCC frontend) | 新增 /imaging 页面 |
| 配置管理 | config.yaml | 所有连接信息外部化 |

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    phenomicsAgentCC Frontend                      │
│  ┌──────────┐  ┌────────────────────────┐  ┌────────────────┐  │
│  │ Sidebar  │  │   MapView (Leaflet)    │  │  ChatPanel     │  │
│  │          │  │   天地图底图 + 影像叠加  │  │  (对话交互)    │  │
│  └──────────┘  └────────────────────────┘  └────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ SSE / REST
┌────────────────────────────▼────────────────────────────────────┐
│                    phenomicsAgentCC Backend                       │
│         Agent (LLM Function Calling) + services.yaml             │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (内网)
┌────────────────────────────▼────────────────────────────────────┐
│                    droneImaging Service (:8002)                   │
│                                                                   │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────┐ ┌───────────┐ │
│  │ Upload/Fetch│ │ GeoTIFF Parse│ │ COG Convert│ │ Tile Serve│ │
│  │   Module    │ │   Module     │ │   Module   │ │  (titiler)│ │
│  └──────┬──────┘ └──────┬───────┘ └─────┬──────┘ └─────┬─────┘ │
│         │               │               │              │        │
│  ┌──────▼──────┐ ┌──────▼───────┐ ┌─────▼──────┐      │        │
│  │ VLM Describe│ │  Embedding   │ │   Search   │      │        │
│  │   Module    │ │   Module     │ │   Module   │      │        │
│  └──────┬──────┘ └──────┬───────┘ └─────┬──────┘      │        │
│         │               │               │              │        │
└─────────┼───────────────┼───────────────┼──────────────┼────────┘
          │               │               │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼────┐
    │  Qwen-VL  │  │ Embedding │  │  Milvus   │  │  MinIO  │
    │   API     │  │  Model    │  │           │  │(COG+Raw)│
    └───────────┘  └───────────┘  └───────────┘  └─────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │   PostgreSQL    │
                                              │ (业务元数据)     │
                                              └─────────────────┘
```

---

## 4. 数据模型

### 4.1 PostgreSQL 表结构

#### images（影像注册表）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | 主键 |
| task_id | VARCHAR(100) | 任务编号 |
| field_group | VARCHAR(200) | 试验田分组 |
| field_name | VARCHAR(200) | 试验田名称 |
| survey_stage | VARCHAR(100) | 调查阶段（如分蘖期、抽穗期） |
| device_model | VARCHAR(200) | 设备型号 |
| data_type | VARCHAR(100) | 数据类型（如可见光、多光谱） |
| surveyor | VARCHAR(100) | 调查员 |
| survey_time | TIMESTAMP | 调查时间 |
| upload_time | TIMESTAMP | 上传/入库时间 |
| original_filename | VARCHAR(500) | 原始文件名 |
| original_path | VARCHAR(1000) | MinIO 中原始文件路径 |
| cog_path | VARCHAR(1000) | MinIO 中 COG 文件路径 |
| thumbnail_path | VARCHAR(1000) | MinIO 中缩略图路径 |
| file_size_bytes | BIGINT | 文件大小（字节） |
| image_width | INTEGER | 像素宽度 |
| image_height | INTEGER | 像素高度 |
| bands | INTEGER | 波段数 |
| crs | VARCHAR(50) | 坐标系（如 EPSG:4326） |
| bbox | JSONB | 地理范围 [min_lon, min_lat, max_lon, max_lat] |
| center_lon | DOUBLE PRECISION | 中心经度 |
| center_lat | DOUBLE PRECISION | 中心纬度 |
| pixel_scale_x | DOUBLE PRECISION | X方向像素分辨率（度/像素） |
| pixel_scale_y | DOUBLE PRECISION | Y方向像素分辨率（度/像素） |
| nodata | DOUBLE PRECISION | NoData 值 |
| geotransform | JSONB | 完整仿射变换参数 |
| vlm_description | TEXT | VLM 生成的自由描述 |
| vlm_model | VARCHAR(100) | 使用的 VLM 模型名 |
| vlm_time | TIMESTAMP | VLM 描述生成时间 |
| embedding_id | VARCHAR(100) | Milvus 中对应的向量 ID |
| status | VARCHAR(50) | 处理状态：uploaded/parsing/converting/ready/error |
| error_message | TEXT | 错误信息（status=error时） |
| source | VARCHAR(50) | 来源：upload/fetch |
| extra_metadata | JSONB | 扩展元数据（预留） |
| created_at | TIMESTAMP | 记录创建时间 |
| updated_at | TIMESTAMP | 记录更新时间 |

索引：task_id, field_name, survey_stage, status, (center_lon, center_lat)

#### fetch_sources（外部存储源配置，P1 预留）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID PK | 主键 |
| name | VARCHAR(200) | 存储源名称 |
| source_type | VARCHAR(50) | 类型：minio/nas/http |
| endpoint | VARCHAR(500) | 连接地址 |
| credentials | JSONB | 认证信息（加密存储） |
| config | JSONB | 额外配置 |
| created_at | TIMESTAMP | 创建时间 |

### 4.2 Milvus Collection

Collection 名：`image_vectors`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(64) PK | 与 PostgreSQL images.id 对应 |
| text_vector | FLOAT_VECTOR(1024) | 文本 embedding（VLM描述+元数据） |
| image_vector | FLOAT_VECTOR(1024) | 图片 embedding（预留，暂不填充） |
| task_id | VARCHAR(100) | 标量过滤字段 |
| field_name | VARCHAR(200) | 标量过滤字段 |
| survey_stage | VARCHAR(100) | 标量过滤字段 |

索引：text_vector 建 IVF_FLAT 或 HNSW 索引，metric_type=COSINE。

### 4.3 MinIO 桶结构

```
Bucket: drone-raw
  └── {task_id}/{image_id}/{original_filename}     # 原始 GeoTIFF

Bucket: drone-cog
  └── {task_id}/{image_id}/{stem}.tif              # COG 格式

Bucket: drone-thumb
  └── {task_id}/{image_id}/{stem}_thumb.jpg        # 缩略图（供 VLM 和前端列表用）
```

---

## 5. API 接口定义

基础路径：`http://<host>:8002/api`

### 5.1 影像上传

```
POST /api/images/upload
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | GeoTIFF 文件 |
| task_id | string | 是 | 任务编号 |
| field_group | string | 否 | 试验田分组 |
| field_name | string | 否 | 试验田名称 |
| survey_stage | string | 否 | 调查阶段 |
| device_model | string | 否 | 设备型号 |
| data_type | string | 否 | 数据类型 |
| surveyor | string | 否 | 调查员 |
| survey_time | string | 否 | 调查时间（ISO 8601） |

**响应 201：**
```json
{
  "id": "uuid",
  "status": "uploaded",
  "message": "影像已上传，后台处理中"
}
```

**后台异步流程：** 解析 GeoTIFF 元数据 → 上传 MinIO → COG 转换 → 生成缩略图 → VLM 描述 → Embedding → 写入 Milvus → status=ready

### 5.2 外部拉取（P1 预留）

```
POST /api/images/fetch
Content-Type: application/json
```

```json
{
  "source_id": "外部存储源ID",
  "task_id": "任务编号",
  "image_ids": ["影像ID列表"]
}
```

**响应 202：**
```json
{
  "job_id": "uuid",
  "status": "fetching",
  "count": 5
}
```

### 5.3 语义检索

```
POST /api/images/search
Content-Type: application/json
```

```json
{
  "query": "海南试验点水稻分蘖期的影像",
  "limit": 10,
  "filters": {
    "task_id": "可选",
    "field_name": "可选",
    "survey_stage": "可选"
  }
}
```

**响应 200：**
```json
{
  "results": [
    {
      "id": "uuid",
      "task_id": "TASK-2026-001",
      "field_name": "海南南繁基地A区",
      "survey_stage": "分蘖期",
      "survey_time": "2026-07-15T10:30:00",
      "bbox": [109.2130, 18.3635, 109.2136, 18.3640],
      "center": [109.2133, 18.3638],
      "thumbnail_url": "http://<host>:8002/api/images/{id}/thumbnail",
      "tile_url": "http://<host>:8002/api/tiles/{id}/{z}/{x}/{y}.png",
      "vlm_description": "影像显示水稻处于分蘖期，冠层覆盖度约60%...",
      "score": 0.87
    }
  ],
  "total": 3
}
```

### 5.4 影像详情

```
GET /api/images/{image_id}
```

**响应 200：**
```json
{
  "id": "uuid",
  "task_id": "TASK-2026-001",
  "field_group": "A组",
  "field_name": "海南南繁基地A区",
  "survey_stage": "分蘖期",
  "device_model": "DJI Mavic 3M",
  "data_type": "可见光",
  "surveyor": "张三",
  "survey_time": "2026-07-15T10:30:00",
  "upload_time": "2026-07-20T14:00:00",
  "original_filename": "flight_001.tif",
  "file_size_bytes": 754974720,
  "image_width": 16220,
  "image_height": 13758,
  "bands": 4,
  "crs": "EPSG:4326",
  "bbox": [109.2130, 18.3635, 109.2136, 18.3640],
  "center": [109.2133, 18.3638],
  "pixel_scale_x": 3.5e-8,
  "pixel_scale_y": 3.34e-8,
  "nodata": 0,
  "vlm_description": "...",
  "vlm_model": "qwen-vl-max",
  "status": "ready",
  "tile_url": "http://<host>:8002/api/tiles/{id}/{z}/{x}/{y}.png",
  "thumbnail_url": "http://<host>:8002/api/images/{id}/thumbnail"
}
```

### 5.5 瓦片服务

```
GET /api/tiles/{image_id}/{z}/{x}/{y}.png
```

由 titiler 内嵌路由处理，从 MinIO 读取 COG，按 zoom/x/y 返回 256×256 PNG 瓦片。支持 WebP 格式（Accept 头协商）。

### 5.6 缩略图

```
GET /api/images/{image_id}/thumbnail
```

返回 JPEG 缩略图（长边 512px），用于前端列表展示和 VLM 输入。

### 5.7 影像列表（结构化查询）

```
GET /api/images?task_id=xxx&field_name=xxx&survey_stage=xxx&page=1&page_size=20
```

**响应 200：**
```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 20
}
```

### 5.8 处理状态查询

```
GET /api/images/{image_id}/status
```

**响应 200：**
```json
{
  "id": "uuid",
  "status": "converting",
  "progress": "COG转换中...",
  "error_message": null
}
```

---

## 6. 核心流程

### 6.1 入库流程（异步 Pipeline）

```
上传/拉取 GeoTIFF
    │
    ▼
[1] 存储原始文件 → MinIO (drone-raw)
    │
    ▼
[2] rasterio 解析元数据 → 写入 PostgreSQL (status=parsing→converting)
    │
    ▼
[3] gdal_translate -of COG → MinIO (drone-cog)
    │
    ▼
[4] 生成缩略图 (rasterio + Pillow) → MinIO (drone-thumb)
    │
    ▼
[5] VLM 描述：缩略图 → Qwen-VL API → vlm_description
    │
    ▼
[6] Embedding：拼接(vlm_description + 业务元数据) → text embedding
    │
    ▼
[7] 写入 Milvus → status=ready
```

步骤 3-7 为异步后台任务（FastAPI BackgroundTasks 或 asyncio），失败时 status=error 并记录 error_message。

### 6.2 语义检索流程

```
Agent 调用 search_images(query="海南试验点水稻分蘖期")
    │
    ▼
[1] 解析 query，提取可能的结构化过滤条件（task_id/field_name/survey_stage）
    │
    ▼
[2] query → text embedding
    │
    ▼
[3] Milvus 向量搜索（top_k=limit, 可选标量过滤）
    │
    ▼
[4] 按 id 查 PostgreSQL 补全元数据
    │
    ▼
[5] 组装结果（含 tile_url, thumbnail_url, bbox）→ 返回
```

### 6.3 VLM 描述生成

输入：缩略图（512px JPEG）+ 提示词

提示词模板（config.yaml 可配置路径）：
```
你是一位农业遥感专家。请对这张无人机影像进行详细描述，包括：
1. 作物种类和生长状态
2. 冠层覆盖度和颜色特征
3. 是否有异常情况（病虫害、倒伏、缺苗等）
4. 田间管理痕迹（灌溉、施肥、施药等）
5. 影像质量和拍摄角度
请用中文回答，200-400字。
```

后续可替换为结构化 JSON 输出（定义字段清单后修改 prompt + 解析逻辑）。

---

## 7. 前端集成设计

### 7.1 路由新增

在 phenomicsAgentCC frontend 的 router.ts 中新增：

```typescript
{ path: '/imaging', component: ImagingView, meta: { requiresAuth: true } }
```

### 7.2 页面布局

```
┌─────────────────────────────────────────────────────────────────┐
│ .imaging-layout (flex, height:100%)                             │
│ ┌──────────┐ ┌──────────────────────────┐ ┌──────────────────┐ │
│ │ Sidebar  │ │      Map Area            │ │   Chat Panel     │ │
│ │ (复用现有 │ │  ┌────────────────────┐  │ │   (右侧窄版)    │ │
│ │  侧边栏) │ │  │  Leaflet Map       │  │ │                  │ │
│ │          │ │  │  天地图底图         │  │ │  对话消息流      │ │
│ │ +影像服务 │ │  │  + 影像叠加图层    │  │ │                  │ │
│ │  导航项  │ │  │  + bbox 边框       │  │ │  输入框          │ │
│ │          │ │  └────────────────────┘  │ │                  │ │
│ │          │ │  影像信息面板(底部/浮层)  │ │  服务开关        │ │
│ └──────────┘ └──────────────────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

- Sidebar：复用现有侧边栏，新增"无人机影像"导航项，点击切换到 /imaging
- Map Area（中间，flex:1）：Leaflet 地图，天地图底图瓦片，影像作为 ImageOverlay 或 TileLayer 叠加
- Chat Panel（右侧，~360px）：精简版聊天面板，Agent 对话交互，enabled_services 默认包含影像相关工具

### 7.3 地图交互

- Agent 返回搜索结果后，前端解析 tool_call 结果中的 bbox 和 tile_url
- 自动 fitBounds 到影像范围
- 影像以 XYZ TileLayer 加载（`/api/tiles/{id}/{z}/{x}/{y}.png`）
- 多张影像时显示 bbox 边框列表，点击切换/叠加
- 点击影像区域弹出信息浮层（VLM 描述、业务字段、缩略图）

### 7.4 前端依赖新增

```json
{
  "leaflet": "^1.9.4",
  "@types/leaflet": "^1.9.8"
}
```

天地图瓦片 URL 模板（需申请 key）：
```
http://t{0-7}.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILECOL={x}&TILEROW={y}&TILEMATRIX={z}&tk={天地图key}
```

---

## 8. 配置文件

`config.yaml`（项目根目录）：

```yaml
server:
  host: 0.0.0.0
  port: 8002
  workers: 2

minio:
  endpoint: "http://localhost:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin"
  secure: false
  buckets:
    raw: "drone-raw"
    cog: "drone-cog"
    thumb: "drone-thumb"

postgresql:
  url: "postgresql+asyncpg://postgres:password@localhost:5432/drone_imaging"

milvus:
  host: "localhost"
  port: 19530
  collection: "image_vectors"
  vector_dim: 1024

embedding:
  model: "BAAI/bge-large-zh-v1.5"
  api_url: "http://localhost:8080/embed"    # embedding 服务地址
  api_key: ""
  batch_size: 8

vlm:
  provider: "qwen-vl"                       # 可切换: qwen-vl / local
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "sk-xxx"
  model: "qwen-vl-max"
  prompt_file: "prompts/vlm_describe.txt"   # VLM 提示词文件
  max_image_size: 1024                      # 发送给 VLM 的最大图片尺寸

tianditu:
  key: "your-tianditu-key"

processing:
  cog_blocksize: 512
  cog_overview_levels: [2, 4, 8, 16]
  thumbnail_size: 512
  max_concurrent_tasks: 3                   # 后台处理并发数
```

---

## 9. Agent 工具注册（services.yaml）

在 phenomicsAgentCC 的 services.yaml 中注册：

```yaml
services:
  search_images:
    name: search_images
    description: 根据自然语言描述检索无人机影像。输入查询描述，返回匹配的影像列表，包含地理位置、拍摄信息和AI描述。
    url: http://localhost:8002/api/images/search
    method: POST
    request_template:
      query: "{query}"
      limit: "{limit}"
    timeout: 30

  get_image_detail:
    name: get_image_detail
    description: 获取单张无人机影像的完整详细信息，包括GeoTIFF元数据、业务信息、AI描述和瓦片地址。
    url: http://localhost:8002/api/images/{image_id}
    method: GET
    timeout: 15

  upload_image:
    name: upload_image
    description: 上传无人机GeoTIFF影像并填写业务信息（任务编号、试验田、调查阶段等）。
    url: http://localhost:8002/api/images/upload
    method: POST
    timeout: 120

  fetch_images:
    name: fetch_images
    description: 从外部存储系统拉取无人机影像到本地。指定存储源和影像ID列表。
    url: http://localhost:8002/api/images/fetch
    method: POST
    request_template:
      source_id: "{source_id}"
      task_id: "{task_id}"
    timeout: 300
```

---

## 10. 项目目录结构

```
droneImaging/
├── docs/
│   ├── spec.md              # 本文档
│   ├── tasks.md             # 任务分解
│   └── checklist.md         # 验收标准
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置加载（config.yaml → Pydantic Settings）
│   ├── database.py          # PostgreSQL 连接 + SQLAlchemy async engine
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image.py         # Image ORM 模型
│   │   └── fetch_source.py  # FetchSource ORM 模型
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── image.py         # Pydantic 请求/响应模型
│   │   └── search.py        # 检索相关模型
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── images.py        # 影像 CRUD + 上传 + 列表
│   │   ├── search.py        # 语义检索
│   │   └── tiles.py         # 瓦片服务（titiler 路由挂载）
│   ├── services/
│   │   ├── __init__.py
│   │   ├── geotiff.py       # GeoTIFF 解析（rasterio）
│   │   ├── cog.py           # COG 转换（gdal_translate）
│   │   ├── storage.py       # MinIO 操作封装
│   │   ├── vlm.py           # VLM 描述生成
│   │   ├── embedding.py     # 文本向量化
│   │   ├── milvus_client.py # Milvus 操作封装
│   │   └── pipeline.py      # 入库异步 Pipeline 编排
│   └── utils/
│       ├── __init__.py
│       └── thumbnail.py     # 缩略图生成
├── prompts/
│   └── vlm_describe.txt     # VLM 提示词
├── config.yaml              # 配置文件（外部化）
├── requirements.txt
└── README.md
```

---

## 11. 非功能需求

- **性能：** 瓦片响应 < 200ms（COG 已缓存时）；语义检索 < 1s
- **并发：** 后台处理 Pipeline 支持 3 并发（可配置），避免 GDAL 转换占满资源
- **容错：** 入库 Pipeline 任一步骤失败不影响其他步骤，status 记录错误，支持重试
- **可配置：** 所有外部服务地址、模型、提示词均通过 config.yaml 管理
- **日志：** 结构化日志（loguru），按天轮转，区分 access / processing / error
