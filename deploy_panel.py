#!/usr/bin/env python3
"""
部署面板 — 网页触发多项目部署
启动: python3 deploy_panel.py
默认端口: 9001
"""

import subprocess
import json
import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── 配置 ──────────────────────────────────────────────
LISTEN_PORT = 9001
LOG_DIR = os.path.expanduser("~/deploy-logs")

# 项目列表：名称 → 部署脚本路径
PROJECTS = {
    "droneImaging": {
        "script": os.path.expanduser("~/droneImaging/deploy.sh"),
        "log": os.path.expanduser("~/droneImaging/deploy.log"),
        "health": "http://localhost:8005/health",
    },
    # 以后加新项目在这里加：
    # "another-project": {
    #     "script": "~/another-project/deploy.sh",
    #     "log": "~/another-project/deploy.log",
    #     "health": "http://localhost:8006/health",
    # },
}

os.makedirs(LOG_DIR, exist_ok=True)

# ── 日志 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("deploy-panel")

# ── 部署状态 ───────────────────────────────────────────
deploy_status = {name: "idle" for name in PROJECTS}


def run_deploy(project_name):
    """异步执行部署脚本"""
    if project_name not in PROJECTS:
        return False, "项目不存在"

    if deploy_status[project_name] == "running":
        return False, "部署已在进行中"

    script = PROJECTS[project_name]["script"]
    if not os.path.exists(script):
        return False, f"部署脚本不存在: {script}"

    deploy_status[project_name] = "running"
    logger.info(f"开始部署: {project_name}")

    try:
        result = subprocess.run(
            ["bash", script],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            deploy_status[project_name] = "success"
            logger.info(f"部署成功: {project_name}")
        else:
            deploy_status[project_name] = "failed"
            logger.error(f"部署失败: {project_name}")
    except subprocess.TimeoutExpired:
        deploy_status[project_name] = "timeout"
        logger.error(f"部署超时: {project_name}")
    except Exception as e:
        deploy_status[project_name] = "error"
        logger.error(f"部署异常: {project_name} - {e}")

    return True, "部署完成"


def get_log(project_name, lines=50):
    """读取部署日志"""
    if project_name not in PROJECTS:
        return "项目不存在"
    log_file = PROJECTS[project_name].get("log")
    if not log_file or not os.path.exists(log_file):
        return "暂无日志"
    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"读取日志失败: {e}"


# ── HTTP 处理器 ────────────────────────────────────────
class PanelHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)

        # 首页：项目列表 + 部署按钮
        if parsed.path == "/":
            self.send_html(self.page_index())

        # 查看日志
        elif parsed.path == "/log":
            qs = parse_qs(parsed.query)
            name = qs.get("name", [""])[0]
            self.send_html(self.page_log(name))

        # API：获取状态
        elif parsed.path == "/api/status":
            self.send_json(deploy_status)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)

        # API：触发部署
        if parsed.path == "/api/deploy":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                name = data.get("name", "")
            except json.JSONDecodeError:
                name = ""

            ok, msg = run_deploy(name)
            self.send_json({"ok": ok, "message": msg, "status": deploy_status.get(name)})

        else:
            self.send_response(404)
            self.end_headers()

    # ── 页面模板 ──────────────────────────────────────
    def page_index(self):
        cards = ""
        for name, info in PROJECTS.items():
            status = deploy_status.get(name, "idle")
            health = info.get("health", "")
            cards += f"""
            <div class="card">
                <h3>{name}</h3>
                <p>状态: <span class="status {status}">{status}</span></p>
                <button onclick="deploy('{name}')">部署</button>
                <a href="/log?name={name}"><button>查看日志</button></a>
            </div>
            """
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>部署面板</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h3 {{ margin-top: 0; }}
        button {{ padding: 10px 20px; margin-right: 10px; cursor: pointer; border: none; border-radius: 4px; background: #1890ff; color: white; font-size: 14px; }}
        button:hover {{ background: #40a9ff; }}
        .status {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .idle {{ background: #eee; color: #666; }}
        .running {{ background: #fff7e6; color: #fa8c16; }}
        .success {{ background: #f6ffed; color: #52c41a; }}
        .failed, .timeout, .error {{ background: #fff1f0; color: #f5222d; }}
    </style>
</head>
<body>
    <h1>🚀 部署面板</h1>
    {cards}
    <script>
        async function deploy(name) {{
            if (!confirm('确认部署 ' + name + ' ?')) return;
            const resp = await fetch('/api/deploy', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{name: name}})
            }});
            const data = await resp.json();
            alert(data.message);
            setTimeout(() => location.reload(), 1000);
        }}
        // 每 10 秒刷新状态
        setInterval(() => location.reload(), 10000);
    </script>
</body>
</html>"""

    def page_log(self, name):
        log_content = get_log(name)
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{name} - 部署日志</title>
    <style>
        body {{ font-family: monospace; max-width: 1000px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }}
        h1 {{ color: #fff; }}
        pre {{ background: #2d2d2d; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; max-height: 70vh; overflow-y: auto; }}
        a {{ color: #1890ff; }}
    </style>
</head>
<body>
    <h1>📋 {name} - 部署日志</h1>
    <p><a href="/">← 返回面板</a></p>
    <pre>{log_content}</pre>
</body>
</html>"""

    def send_html(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", LISTEN_PORT), PanelHandler)
    logger.info(f"部署面板启动: http://0.0.0.0:{LISTEN_PORT}")
    print(f"部署面板已启动: http://localhost:{LISTEN_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("部署面板已停止")
