#!/bin/bash
# droneImaging — 部署脚本
# 由 Gitea webhook 触发（push 到 main 分支）

set -e

REPO_DIR=~/droneImaging
LOG_FILE=~/droneImaging/deploy.log
IMAGE_NAME=drone-imaging:latest
CONTAINER_NAME=drone-imaging

echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG_FILE"
echo "[deploy] 开始部署" >> "$LOG_FILE"

# 1. 拉取最新代码
cd "$REPO_DIR"
echo "[deploy] git pull" >> "$LOG_FILE"
git pull origin main >> "$LOG_FILE" 2>&1

# 2. 构建镜像（使用阿里源）
echo "[deploy] docker build" >> "$LOG_FILE"
docker build -t "$IMAGE_NAME" . >> "$LOG_FILE" 2>&1

# 3. 停止旧容器，启动新容器
echo "[deploy] docker restart" >> "$LOG_FILE"
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p 8005:8002 \
  -e DRONE_PUBLIC_BASE_URL=http://10.33.105.145:8005 \
  -v ~/droneImaging/config.yaml:/app/config.yaml:ro \
  "$IMAGE_NAME" >> "$LOG_FILE" 2>&1

# 4. 验证
sleep 5
if curl -sf http://localhost:8005/health > /dev/null 2>&1; then
  echo "[deploy] ✅ 部署成功，服务健康" >> "$LOG_FILE"
else
  echo "[deploy] ❌ 部署完成但健康检查失败" >> "$LOG_FILE"
fi

echo "[deploy] 完成" >> "$LOG_FILE"
