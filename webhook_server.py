#!/usr/bin/env python3
"""
droneImaging — Webhook 接收器
监听 Gitea push 事件，触发部署脚本

启动: python3 webhook_server.py
默认端口: 9001
"""

import hmac
import hashlib
import subprocess
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── 配置 ──────────────────────────────────────────────
LISTEN_PORT = 9001
WEBHOOK_SECRET = "phenomics-webhook-secret-2026"  # 和 Gitea webhook 配置一致
DEPLOY_SCRIPT = "/home/root01/droneImaging/deploy.sh"
LOG_FILE = "/home/root01/droneImaging/webhook.log"

# ── 日志 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("webhook")


class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # 验证签名（如果配置了 secret）
        signature = self.headers.get("X-Gitea-Signature", "")
        if WEBHOOK_SECRET and signature:
            expected = hmac.new(
                WEBHOOK_SECRET.encode(), body, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected, signature):
                logger.warning("签名验证失败")
                self.send_response(403)
                self.end_headers()
                return

        # 解析 payload
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("JSON 解析失败")
            self.send_response(400)
            self.end_headers()
            return

        # 只处理 push 事件到 main 分支
        ref = payload.get("ref", "")
        if ref == "refs/heads/main":
            logger.info("收到 main 分支 push，触发部署")
            # 异步执行部署脚本（不阻塞响应）
            subprocess.Popen(
                ["bash", DEPLOY_SCRIPT],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"deploy triggered")
        else:
            logger.info(f"忽略非 main 分支推送: {ref}")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ignored")

    def log_message(self, format, *args):
        # 抑制默认 HTTP 日志
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", LISTEN_PORT), WebhookHandler)
    logger.info(f"Webhook 接收器启动，监听端口 {LISTEN_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Webhook 接收器已停止")
