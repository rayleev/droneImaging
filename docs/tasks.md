# 无人机影像服务 — 任务分解

## Phase 1：项目骨架与基础设施（P0）

### T1.1 项目初始化
- 创建项目目录结构（按 spec.md 第 10 节）
- 编写 requirements.txt（fastapi, uvicorn, rasterio, minio, sqlalchemy[asyncio], asyncpg, pymilvus, httpx, pillow, loguru, pydantic-settings, python-multipart）
- 编写 config.yaml 模板（所有连接信息占位）
- 实现 src/config.py（Pydantic Settings 加载 config.yaml）
- 实现 src/main.py（FastAPI 应用入口，健康检查 /health）
- 验证：`uvicorn src.main:app --port 8002` 启动成功，/health 返回 200

### T1.2 数据库层
- 实现 src/database.py（async engine + session factory）
- 实现 src/models/image.py（Image ORM 模型，含所有字段和注释）
- 实现 src/models/fetch_source.py（FetchSource ORM 模型，P1 预留）
- 启动时自动建表（create_all）
- 验证：启动后 PostgreSQL 中出现 images 和 fetch_sources 表

### T1.3 MinIO 存储层
- 实现 src/services/storage.py（MinIO 客户端封装）
  - upload_file(bucket, object_name, file_path/file_obj)
  - download_file(bucket, object_name, local_path)
  - get_presigned_url(bucket, object_name)
  - ensure_buckets()（启动时检查/创建桶）
- 验证：上传测试文件到 drone-raw 桶成功

---

## Phase 2：GeoTIFF 处理与 COG 转换（P0）

### T2.1 GeoTIFF 元数据解析
- 实现 src/services/geotiff.py
  - parse_geotiff(file_path) → GeoTiffMetadata（Pydantic 模型）
  - 提取：width, height, bands, crs, bbox, center, pixel_scale, nodata, geotransform
  - 使用 rasterio 读取
- 验证：用 result_exif.json 对应的 TIFF 文件测试，输出 bbox 正确

### T2.2 COG 转换
- 实现 src/services/cog.py
  - convert_to_cog(input_path, output_path, blocksize=512, overview_levels=[2,4,8,16])
  - 调用 gdal_translate -of COG（subprocess 或 osgeo.gdal）
  - 支持进度回调
- 验证：转换后文件可被 rasterio 正常读取，overview 层级正确

### T2.3 缩略图生成
- 实现 src/utils/thumbnail.py
  - generate_thumbnail(input_path, output_path, max_size=512)
  - rasterio 读取 → 降采样 → Pillow 保存 JPEG
- 验证：生成 512px 缩略图，文件 < 200KB

---

## Phase 3：影像入库 Pipeline（P0）

### T3.1 上传接口
- 实现 src/routers/images.py
  - POST /api/images/upload（multipart/form-data）
  - 接收文件 + 业务字段 → 写入 PostgreSQL（status=uploaded）→ 触发后台 Pipeline
  - GET /api/images/{id}（详情）
  - GET /api/images（列表，支持 task_id/field_name/survey_stage 过滤 + 分页）
  - GET /api/images/{id}/status（处理状态）
  - GET /api/images/{id}/thumbnail（缩略图）
- 实现 src/schemas/image.py（请求/响应 Pydantic 模型）

### T3.2 异步 Pipeline 编排
- 实现 src/services/pipeline.py
  - process_image(image_id): 编排完整入库流程
    1. 上传原始文件到 MinIO (drone-raw)
    2. 解析 GeoTIFF 元数据 → 更新 DB
    3. COG 转换 → 上传 MinIO (drone-cog)
    4. 生成缩略图 → 上传 MinIO (drone-thumb)
    5. VLM 描述（Phase 4）
    6. Embedding + Milvus 写入（Phase 4）
    7. status=ready
  - 错误处理：任一步骤失败 → status=error + error_message
  - 并发控制：asyncio.Semaphore(max_concurrent_tasks)
- 验证：上传一张测试 TIFF，观察 status 从 uploaded → parsing → converting → ready

---

## Phase 4：VLM 描述与向量检索（P0）

### T4.1 VLM 描述生成
- 实现 src/services/vlm.py
  - describe_image(thumbnail_path) → str
  - 调用 Qwen-VL API（OpenAI 兼容格式，httpx）
  - 提示词从 prompts/vlm_describe.txt 读取
  - 支持配置切换（provider: qwen-vl / local）
  - 超时和重试（3次，指数退避）
- 编写 prompts/vlm_describe.txt
- 验证：传入缩略图，返回 200-400 字中文描述

### T4.2 Embedding 服务
- 实现 src/services/embedding.py
  - embed_text(text) → List[float]
  - embed_batch(texts) → List[List[float]]
  - 调用配置的 embedding API
  - 拼接逻辑：vlm_description + f"任务:{task_id} 试验田:{field_name} 阶段:{survey_stage} 设备:{device_model}"
- 验证：返回 1024 维向量

### T4.3 Milvus 客户端
- 实现 src/services/milvus_client.py
  - ensure_collection()（启动时检查/创建 collection + 索引）
  - insert_vector(id, text_vector, metadata)
  - search(query_vector, top_k, filters) → List[(id, score)]
  - delete_by_id(id)
- 验证：插入测试向量，搜索返回正确结果

### T4.4 语义检索接口
- 实现 src/routers/search.py
  - POST /api/images/search
  - query → embedding → Milvus 搜索 → 补全 PostgreSQL 元数据 → 组装响应
  - 支持 filters（task_id, field_name, survey_stage）标量过滤
- 验证：搜索"海南试验点水稻分蘖期"返回相关影像

---

## Phase 5：瓦片服务（P0）

### T5.1 titiler 集成
- 实现 src/routers/tiles.py
  - 挂载 titiler 路由：GET /api/tiles/{image_id}/{z}/{x}/{y}.png
  - 自定义 COGReader：从 MinIO 读取 COG（通过 presigned URL 或 fsspec）
  - 支持 PNG/WebP 输出
  - 支持透明度（nodata 区域透明）
- 验证：浏览器访问瓦片 URL，返回正确的 256×256 图片

---

## Phase 6：前端集成（P0）

### T6.1 路由与布局
- phenomicsAgentCC frontend 新增 /imaging 路由
- 创建 ImagingView.vue（三栏布局：sidebar + map + chat）
- Sidebar 新增"无人机影像"导航项
- 安装 leaflet 依赖

### T6.2 地图组件
- 创建 MapPanel.vue
  - Leaflet 初始化 + 天地图底图瓦片
  - 影像 TileLayer 加载（/api/tiles/{id}/{z}/{x}/{y}.png）
  - fitBounds 到影像 bbox
  - 多影像 bbox 边框列表
  - 点击影像弹出信息浮层

### T6.3 聊天面板集成
- 创建 ChatPanel.vue（右侧精简版聊天）
  - 复用 useChat composable
  - enabled_services 默认包含 search_images, get_image_detail
  - 解析 tool_call 结果，提取影像数据 → 通知 MapPanel 加载图层
  - 影像结果卡片（缩略图 + 名称 + 阶段 + 得分）

### T6.4 联调
- Vite proxy 配置：/api/drone → http://localhost:8002
- 端到端测试：对话"帮我看一下海南试验点的影像" → 地图显示影像

---

## Phase 7：外部拉取接口（P1）

### T7.1 存储源管理
- fetch_sources 表 CRUD 接口
- POST /api/sources（注册外部存储源）
- GET /api/sources（列表）

### T7.2 拉取流程
- POST /api/images/fetch
- 按元数据从外部 MinIO/NAS 拉取文件
- 复用入库 Pipeline
- 进度追踪（job_id + status 查询）

---

## Phase 8：试验小区划分（P2，暂不实现）

### T8.1 田块边界识别
- VLM/CV 模型识别田块边界、道路、沟渠
- 语义分割 → GeoJSON 多边形

### T8.2 自然语言交互
- 解析用户划分指令（行列数、处理组）
- 模糊指令反问确认
- 生成带编号小区列表（边界坐标）

### T8.3 前端展示
- 原图叠加矢量边界
- 点击小区查看详情
- 边界编辑（拖拽顶点）
