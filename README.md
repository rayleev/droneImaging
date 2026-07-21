# droneImaging — 无人机影像服务

独立部署的 FastAPI 微服务，为 `phenomicsAgentCC` 表型智能体提供无人机影像的**入库、语义检索和地图展示**能力。Agent 通过 Function Calling 调用本服务的 HTTP API，前端通过 Agent 返回的结构化数据在 Leaflet 地图上渲染影像。

## 功能特性

- **影像入库**：上传 GeoTIFF → 自动解析元数据 → COG 转换 → 缩略图生成 → VLM 语义描述 → 向量化 → Milvus 索引（全异步 Pipeline）
- **语义检索**：自然语言查询 → 文本 embedding → Milvus 向量搜索 → 返回带地理位置和瓦片地址的影像列表
- **地图瓦片**：基于 COG 的 XYZ 动态瓦片服务，支持前端流畅浏览 700-800MB 大图
- **外部拉取**（P1 预留）：从外部 MinIO/NAS 按元数据拉取影像

## 技术栈

| 层次 | 选型 |
|------|------|
| 后端框架 | FastAPI + Uvicorn（异步，端口 8002） |
| GeoTIFF 处理 | rasterio + GDAL（元数据解析、COG 转换） |
| 对象存储 | MinIO（原始 TIFF + COG + 缩略图） |
| 关系数据库 | PostgreSQL + SQLAlchemy（async） |
| 向量数据库 | Milvus（pymilvus） |
| Embedding | BGE-M3（可配置） |
| VLM | Qwen-VL（可配置切换本地模型） |
| 配置管理 | config.yaml（Pydantic Settings） |

## 快速启动

### 本地启动（开发）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 复制并填写配置（config.yaml 已被 git 忽略，不含真实密钥）
cp config.yaml.example config.yaml
# 编辑 config.yaml，填入实际的数据库 / MinIO / Milvus / API Key

# 3. 启动服务
uvicorn src.main:app --port 8002 --reload

# 4. 验证
curl http://localhost:8002/health
```

启动时自动完成：PostgreSQL 建表、MinIO 建桶、Milvus 建 collection。

### Docker 启动

```bash
# 构建镜像
docker build -t drone-imaging .

# 运行（挂载配置文件，端口 8002）
docker run -d \
  --name drone-imaging \
  -p 8002:8002 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e DRONE_PUBLIC_BASE_URL=http://localhost:8002 \
  drone-imaging
```

### Docker Compose 启动（含依赖服务）

一键启动完整本地开发栈（PostgreSQL + MinIO + Milvus + 本服务）：

```bash
docker compose up -d
```

服务组件：

| 服务 | 端口 | 说明 |
|------|------|------|
| drone-imaging | 8002 | 本服务 |
| postgres | 5432 | 业务元数据 |
| minio | 9000 / 9001 | 对象存储（控制台 9001） |
| milvus | 19530 | 向量数据库 |
| milvus-minio | 9000 | Milvus 内部存储（与上方 minio 端口错开，或按需调整） |
| etcd | 2379 | Milvus 元数据存储 |

> 生产环境建议将依赖服务替换为独立部署实例，仅保留本服务容器。

## 配置说明

所有外部服务连接信息和处理参数通过 `config.yaml` 管理（详见 `config.yaml.example` 模板）：

| 配置节 | 说明 |
|--------|------|
| `server` | 服务 host/port/workers |
| `minio` | MinIO 连接 + 三个桶名（raw / cog / thumb） |
| `postgresql` | 数据库连接 URL |
| `milvus` | Milvus 连接 + collection 名 + 向量维度 |
| `embedding` | Embedding API 地址、模型、密钥 |
| `vlm` | VLM 提供者、API 地址、模型、提示词路径 |
| `tianditu` | 天地图 key |
| `processing` | COG 块大小、缩略图尺寸、后台并发数 |

**对外 URL 配置**：设置环境变量 `DRONE_PUBLIC_BASE_URL` 指定客户端可访问的地址（Docker / 反向代理部署时必须设置，否则生成的瓦片/缩略图 URL 不可达）。

## API 概览

基础路径：`http://<host>:8002/api`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/images/upload` | 上传 GeoTIFF（multipart），返回 201 + 后台处理 |
| GET | `/api/images/{id}` | 影像完整详情 |
| GET | `/api/images` | 影像列表（支持过滤 + 分页） |
| GET | `/api/images/{id}/status` | 处理状态查询 |
| GET | `/api/images/{id}/thumbnail` | 缩略图（JPEG） |
| POST | `/api/images/search` | 语义检索 |
| GET | `/api/tiles/{id}/{z}/{x}/{y}.png` | XYZ 瓦片（256×256 PNG） |
| GET | `/health` | 健康检查 |

详细接口定义见 `docs/spec.md`。

## 入库流程（异步 Pipeline）

```
上传 GeoTIFF
  → [1] 存储原始文件 → MinIO (drone-raw)
  → [2] rasterio 解析元数据 → PostgreSQL
  → [3] gdal_translate COG → MinIO (drone-cog)
  → [4] 生成缩略图 → MinIO (drone-thumb)
  → [5] VLM 描述（Qwen-VL）
  → [6] Embedding → Milvus
  → status = ready
```

任一步骤失败标记 `status=error` 并记录 `error_message`，不影响其他影像处理。Pipeline 并发数由 `processing.max_concurrent_tasks` 控制。

## 项目结构

```
droneImaging/
├── docs/
│   ├── spec.md              # 需求规格说明书
│   ├── tasks.md             # 任务分解
│   └── checklist.md         # 验收标准
├── src/
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置加载（config.yaml → Pydantic）
│   ├── database.py          # PostgreSQL 连接（SQLAlchemy async）
│   ├── models/              # ORM 模型（Image, FetchSource）
│   ├── schemas/             # Pydantic 请求/响应模型
│   ├── routers/             # 路由（images, search, tiles）
│   ├── services/            # 业务逻辑（pipeline, geotiff, cog, storage, vlm, embedding, milvus_client）
│   └── utils/               # 工具（thumbnail）
├── prompts/
│   └── vlm_describe.txt     # VLM 提示词
├── config.yaml              # 配置文件（git 忽略，含密钥）
├── config.yaml.example      # 配置模板
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

## 文档

- 需求规格：`docs/spec.md`
- 任务分解：`docs/tasks.md`
- 验收标准：`docs/checklist.md`
