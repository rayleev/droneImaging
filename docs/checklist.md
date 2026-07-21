# 无人机影像服务 — 验收标准

## Phase 1：项目骨架

- [ ] `uvicorn src.main:app --port 8002` 启动无报错
- [ ] GET /health 返回 `{"status": "ok"}`
- [ ] config.yaml 所有配置项可正确加载（修改配置后重启生效）
- [ ] PostgreSQL 中自动创建 images 表和 fetch_sources 表，字段含中文注释
- [ ] MinIO 中自动创建 drone-raw、drone-cog、drone-thumb 三个桶

## Phase 2：GeoTIFF 处理

- [ ] 用测试 GeoTIFF（16220×13758, EPSG:4326）验证元数据解析：
  - bbox 输出为 [109.2130, 18.3635, 109.2136, 18.3640]（±0.001 精度）
  - crs 输出为 "EPSG:4326"
  - image_width=16220, image_height=13758, bands=4
- [ ] COG 转换后文件可被 rasterio 正常打开
- [ ] COG 包含 overview 层级（2, 4, 8, 16）
- [ ] COG 文件大小合理（不超过原始文件的 1.2 倍）
- [ ] 缩略图生成为 JPEG，长边 512px，文件 < 200KB

## Phase 3：入库 Pipeline

- [ ] POST /api/images/upload 上传 GeoTIFF + 业务字段，返回 201 + image_id
- [ ] 上传后 status 变化：uploaded → parsing → converting → ready
- [ ] 原始文件出现在 MinIO drone-raw 桶正确路径
- [ ] COG 文件出现在 MinIO drone-cog 桶
- [ ] 缩略图出现在 MinIO drone-thumb 桶
- [ ] PostgreSQL images 表记录完整（所有业务字段 + 元数据字段）
- [ ] 上传非 GeoTIFF 文件时返回 400 错误
- [ ] Pipeline 中间步骤失败时 status=error，error_message 有具体信息
- [ ] GET /api/images/{id} 返回完整详情
- [ ] GET /api/images?task_id=xxx 正确过滤
- [ ] GET /api/images/{id}/thumbnail 返回 JPEG 图片

## Phase 4：VLM 与检索

- [ ] 入库完成后 vlm_description 字段非空，200-400 字中文
- [ ] VLM API 超时时自动重试（最多 3 次）
- [ ] Embedding 向量维度 = 1024
- [ ] Milvus collection 创建成功，含 text_vector + image_vector(预留) 字段
- [ ] POST /api/images/search 传入自然语言 query，返回相关影像列表
- [ ] 搜索结果包含：id, bbox, tile_url, thumbnail_url, vlm_description, score
- [ ] filters 参数（task_id, field_name, survey_stage）正确过滤
- [ ] 无匹配结果时返回空列表（非报错）

## Phase 5：瓦片服务

- [ ] GET /api/tiles/{image_id}/{z}/{x}/{y}.png 返回 256×256 PNG
- [ ] 不同 zoom 级别（14-20）瓦片正确渲染
- [ ] nodata 区域为透明（PNG alpha 通道）
- [ ] 不存在的 image_id 返回 404
- [ ] 瓦片响应时间 < 500ms（首次）/ < 200ms（缓存后）

## Phase 6：前端集成

- [ ] phenomicsAgentCC 侧边栏出现"无人机影像"导航项
- [ ] 点击进入 /imaging 页面，三栏布局正确渲染
- [ ] 天地图底图正常加载（需有效 key）
- [ ] 对话"帮我看一下海南试验点的影像"后：
  - 右侧聊天面板显示 Agent 回复
  - 中间地图自动定位到影像 bbox
  - 影像瓦片正确叠加在地图上
- [ ] 多张影像时显示 bbox 边框，可点击切换
- [ ] 点击影像弹出信息浮层（VLM 描述 + 业务字段）
- [ ] 地图缩放/平移流畅，无明显卡顿

## Phase 7：外部拉取（P1）

- [ ] POST /api/sources 注册外部 MinIO 存储源成功
- [ ] POST /api/images/fetch 触发拉取，返回 job_id
- [ ] 拉取完成后影像走正常入库 Pipeline，status=ready
- [ ] 拉取失败时 status=error，有具体错误信息

## 通用验收

- [ ] 所有 API 错误返回统一格式 `{"detail": "错误描述"}`
- [ ] 日志输出到 logs/ 目录，按天轮转
- [ ] config.yaml 修改后重启即生效，无需改代码
- [ ] VLM 模型可通过配置切换（qwen-vl → local）
- [ ] 代码核心逻辑有注释，数据库表和字段有注释
