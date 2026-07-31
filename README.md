# droneImaging — 无人机影像服务

独立部署的 FastAPI 微服务，为 `phenomicsAgentCC` 表型智能体提供无人机影像的**入库、语义检索、地图展示、试验小区划分与智能补全**能力。Agent 通过 Function Calling 调用本服务的 HTTP API，前端通过 Agent 返回的结构化数据在 Leaflet 地图上渲染影像与小区。

## 功能特性

- **影像入库**：上传 GeoTIFF → 自动解析元数据 → COG 转换 → 缩略图生成 → VLM 语义描述 → 向量化 → Milvus 索引（全异步 Pipeline）
- **影像管理**：列表查询（搜索 + 多字段过滤 + 分页）、详情、新增、编辑、软删除、状态查询、缩略图
- **语义检索**：自然语言查询 → 文本 embedding → Milvus 向量搜索 → 返回带地理位置和瓦片地址的影像列表
- **地图瓦片**：基于 COG 的 XYZ 动态瓦片服务（256×256 PNG），支持前端流畅浏览 700-800MB 大图，越界瓦片自动透明
- **试验小区划分**：按行列数或小区尺寸（米）做网格划分，支持旋转、A1/线性编号方案
- **试验小区智能补全**：用户绘制一个示例区域 + 自然语言描述 → 自动补全所有小区，内置 5 种可切换策略
- **外部拉取**（P1 预留）：从外部 MinIO/NAS 按元数据拉取影像

## 技术栈

| 层次 | 选型 |
|------|------|
| 后端框架 | FastAPI + Uvicorn（异步，端口 8002） |
| GeoTIFF 处理 | rasterio + GDAL（元数据解析、COG 转换、瓦片渲染） |
| 对象存储 | MinIO（原始 TIFF + COG + 缩略图） |
| 关系数据库 | PostgreSQL + SQLAlchemy（async） |
| 向量数据库 | Milvus（pymilvus） |
| Embedding | BGE-M3（可配置） |
| VLM | Qwen-VL / GLM-4V（可配置切换） |
| LLM | DSv4-flash 等（用于小区补全的描述解析） |
| SAM | 远程分割服务（可选，端口 8003） |
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
|--------|--------|
| `server` | 服务 host/port/workers、上传大小上限 |
| `minio` | MinIO 连接 + 三个桶名（raw / cog / thumb） |
| `postgresql` | 数据库连接 URL |
| `milvus` | Milvus 连接 + collection 名 + 向量维度 + 启动时是否重建 |
| `embedding` | Embedding API 地址、模型、密钥、批量大小 |
| `vlm` | VLM 提供者、API 地址、模型、提示词路径、图片尺寸、超时、重试 |
| `llm` | LLM 提供者、API 地址、模型、超时、重试（用于小区补全） |
| `tianditu` | 天地图 key |
| `processing` | COG 块大小、overview 层级、缩略图尺寸、后台并发数 |
| `completion` | 试验小区智能补全：策略选择、SAM 服务地址、可分别覆盖的 llm/vlm 子配置 |

**completion 配置 fallback 逻辑**：`completion.llm` / `completion.vlm` 子节为 `null` 时自动复用顶层 `llm` / `vlm`，避免重复配置。所有连接字段均可通过环境变量覆盖（如 `VLM_API_KEY`、`LLM_API_URL`、`MINIO_ENDPOINT` 等）。

**对外 URL 配置**：设置环境变量 `DRONE_PUBLIC_BASE_URL` 指定客户端可访问的地址（Docker / 反向代理部署时必须设置，否则生成的瓦片/缩略图 URL 不可达）。

## API 概览

基础路径：`http://<host>:8002/api`

### 影像管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/images/upload` | 上传 GeoTIFF（multipart），返回 201 + 后台处理 |
| GET | `/api/images` | 影像列表（搜索 + 多字段过滤 + 分页） |
| POST | `/api/images` | 新增影像记录（不触发 Pipeline） |
| GET | `/api/images/{id}` | 影像完整详情 |
| PUT | `/api/images/{id}` | 编辑影像记录 |
| DELETE | `/api/images/{id}` | 软删除影像 |
| GET | `/api/images/{id}/status` | 处理状态查询 |
| GET | `/api/images/{id}/thumbnail` | 缩略图（JPEG） |
| POST | `/api/images/search` | 语义检索（自然语言 → 向量 → Milvus） |

### 瓦片服务

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/tiles/{id}/{z}/{x}/{y}.png` | XYZ 瓦片（256×256 PNG，越界透明） |

### 试验小区

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/plots/divide` | 规则网格划分（按行列数或小区尺寸，支持旋转/编号方案） |
| POST | `/api/plots/complete` | 智能补全（示例区域 + 自然语言 → 自动布满） |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/health` | 健康检查（前端经 `/api/drone` 代理访问） |

## 入库流程（异步 Pipeline）

```
上传 GeoTIFF
  → [1] 存储原始文件 → MinIO (drone-raw)
  → [2] rasterio 解析元数据 → PostgreSQL
  → [3] COG 转换 → MinIO (drone-cog)
  → [4] 生成缩略图 → MinIO (drone-thumb)
  → [5] VLM 描述（结构化 JSON：作物/生长期/冠层/异常/管理痕迹/影像质量/拍摄角度）
  → [6] Embedding → Milvus
  → status = ready
```

任一步骤失败标记 `status=error` 并记录 `error_message`，不影响其他影像处理。Pipeline 并发数由 `processing.max_concurrent_tasks` 控制，CPU 密集步骤（COG/缩略图）丢到线程池执行。

## 试验小区智能补全策略

通过 `config.yaml` 的 `completion.strategy` 切换，5 种策略共享同一抽象基类，按需选用：

| 策略值 | 思路 | 适用场景 |
|--------|------|----------|
| `vlm_direct` | VLM 分块识别关键点（田角 / 分隔线）→ 代码计算网格 | 默认推荐，大田块关键点识别 |
| `vlm` | VLM 端到端输出 field_bbox + 行列数 | 形状规整、VLM 直接给出布局 |
| `sam_llm` | SAM 远程分割检测边界 + LLM 解析描述 | 有 SAM 服务、需要真实边界裁剪 |
| `sam_vlm` | SAM 实例分割 + VLM 描述混合 | 复杂田块、需多模态融合 |
| `sam_template` | SAM 实例分割 + 用户画框模板匹配 | 用户画框可作为模板复用 |

> 未配置 SAM 远程服务时，`sam_*` 系列策略会 fallback 到 OpenCV 边界检测或网格生成。

## 项目结构

```
droneImaging/
├── src/
│   ├── main.py                      # FastAPI 应用入口、生命周期
│   ├── config.py                    # 配置加载（config.yaml → Pydantic）
│   ├── database.py                  # PostgreSQL 连接（SQLAlchemy async）
│   ├── models/                      # ORM 模型（Image, FetchSource）
│   ├── schemas/image.py             # Pydantic 请求/响应模型
│   ├── routers/
│   │   ├── images.py                # 影像 CRUD + 上传 + 状态 + 缩略图
│   │   ├── search.py                # 语义检索
│   │   ├── tiles.py                 # XYZ 瓦片
│   │   ├── plots.py                 # 试验小区规则划分
│   │   └── complete.py              # 试验小区智能补全
│   ├── services/
│   │   ├── pipeline.py              # 入库 Pipeline 编排
│   │   ├── geotiff.py               # GeoTIFF 元数据解析
│   │   ├── cog.py                   # COG 转换
│   │   ├── storage.py               # MinIO 客户端
│   │   ├── vlm.py                   # VLM 描述生成
│   │   ├── embedding.py             # 文本向量化
│   │   ├── milvus_client.py         # Milvus collection 管理
│   │   ├── plot_divider.py          # 规则网格划分
│   │   └── plot_completion/         # 智能补全策略
│   │       ├── base.py              # 策略抽象基类
│   │       ├── config.py            # 补全配置加载
│   │       ├── boundary_detector.py # 边界检测（SAM 远程 / OpenCV）
│   │       ├── llm_parser.py        # LLM 解析自然语言描述
│   │       ├── vlm_direct_strategy.py
│   │       ├── vlm_strategy.py
│   │       ├── sam_llm_strategy.py
│   │       ├── sam_vlm_strategy.py
│   │       └── sam_template_strategy.py
│   └── utils/thumbnail.py           # 缩略图生成
├── prompts/                         # 各策略提示词
│   ├── vlm_describe.txt             # 入库 VLM 描述（结构化 JSON）
│   ├── vlm_keypoints.txt            # vlm_direct 策略关键点识别
│   ├── vlm_plot_completion.txt      # vlm 策略端到端划分
│   ├── vlm_direct_plots.txt         # vlm_direct 策略辅助
│   ├── vlm_understand_drawing.txt   # 理解用户绘图
│   └── llm_plot_completion.txt      # LLM 解析描述
├── config.yaml                      # 配置文件（git 忽略，含密钥）
├── config.yaml.example              # 配置模板
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── woodpecker.yml                   # CI 配置
├── AGENTS.md / CLAUDE.md            # AI 编码助手指引
└── README.md
```

## 关键设计点

- **异步全链路**：SQLAlchemy async session；VLM/embedding 调用 via httpx；CPU 密集 rasterio 操作丢到线程池
- **状态机驱动**：`Image.status` 是 Pipeline 进度的唯一真相源，对外通过 status 端点暴露
- **URL 即时合成**：tile/thumbnail URL 在响应时由 `get_public_base_url()` 合成，不入库
- **越界瓦片透明**：瓦片地理范围与影像求交集，只渲染交集到 256×256 画布对应位置，其余透明，影像随缩放正确变化
- **策略可插拔**：智能补全 5 种策略共享抽象基类，配置切换，便于扩展新策略
- **VLM 结构化输出**：入库 VLM 描述输出纯 JSON（作物/生长期/冠层覆盖度/异常/管理痕迹/影像质量/拍摄角度），summary 用于 embedding，其余存 `extra_metadata` JSONB
- **P1 预留**：`FetchSource` 模型与 fetch 流程为预留；Milvus 中 `image_vector` 字段预留（当前 zero-filled）
