# -*- coding: utf-8 -*-
"""Mistral API 网关 — 多 key 轮询 + 429 冷却 + 流式 SSE + OpenAI 兼容。

仅网关模式：配置由挂载的 .env 提供，Key 池由挂载的 keys.txt 提供。

部署:
    docker build -t mistral2api .
    docker run -d -p 8082:8082 -v ./.env:/app/.env:ro -v ./keys.txt:/app/keys.txt:ro mistral2api
本地:
    python server.py  # 读取 .env / keys.txt（如存在）
"""
import json
import os
import time
import threading
from datetime import datetime
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests


def load_dotenv(path: str):
    """.env 解析：KEY=VALUE，支持 # 注释和引号，不覆盖已有环境变量。"""
    try:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'").strip('"')
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


class KeyPool:
    """线程安全的 API key 轮询池 + 429 冷却。"""

    def __init__(self, cooldown_seconds: int = 60):
        self._keys = []          # [{"key": "...", "cooldown_until": 0, "requests": 0, "errors": 0}]
        self._lock = threading.Lock()
        self._idx = 0
        self.cooldown = cooldown_seconds

    def add(self, key: str):
        with self._lock:
            if not any(k["key"] == key for k in self._keys):
                self._keys.append({"key": key, "cooldown_until": 0,
                                   "requests": 0, "errors": 0, "added_at": datetime.now().isoformat()})

    def remove(self, key: str):
        with self._lock:
            self._keys = [k for k in self._keys if k["key"] != key]

    def next(self) -> Optional[dict]:
        """获取下一个可用 key（跳过冷却中的）。"""
        with self._lock:
            if not self._keys:
                return None
            now = time.time()
            for _ in range(len(self._keys)):
                entry = self._keys[self._idx % len(self._keys)]
                self._idx += 1
                if entry["cooldown_until"] < now:
                    entry["requests"] += 1
                    return entry
            # 全部冷却中，返回最早冷却结束的
            earliest = min(self._keys, key=lambda k: k["cooldown_until"])
            earliest["requests"] += 1
            return earliest

    def mark_429(self, key: str):
        """标记 key 遇到 429，冷却。"""
        with self._lock:
            for k in self._keys:
                if k["key"] == key:
                    k["cooldown_until"] = time.time() + self.cooldown
                    k["errors"] += 1
                    break

    def mark_ok(self, key: str):
        """标记 key 请求成功。"""
        with self._lock:
            for k in self._keys:
                if k["key"] == key:
                    # 成功后重置冷却（可能之前是临时问题）
                    k["cooldown_until"] = 0
                    break

    def status(self) -> list:
        """返回所有 key 的状态。"""
        with self._lock:
            now = time.time()
            return [{**k, "cooling_down": k["cooldown_until"] > now,
                     "cooldown_remaining": max(0, int(k["cooldown_until"] - now))}
                    for k in self._keys]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._keys)

    @property
    def available(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for k in self._keys if k["cooldown_until"] < now)

    def load_from_file(self, path: str):
        """从 accounts_*.txt 加载 key（格式：email|key 每行一个）。"""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "|" in line:
                    _, key = line.split("|", 1)
                    self.add(key.strip())
                elif line and len(line) > 20:
                    self.add(line.strip())


MISTRAL_API = "https://api.mistral.ai/v1"
pool = KeyPool()
gateway_keys = []  # 网关自身的 API key（客户端用这个访问）


class GatewayHandler(BaseHTTPRequestHandler):
    """OpenAI 兼容 API 网关。"""

    def _send_json(self, code: int, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        """检查客户端的 API key。"""
        if not gateway_keys:
            return True  # 不设鉴权
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:] in gateway_keys
        return False

    def _proxy_chat(self, body: dict, stream: bool):
        """代理 chat completions 请求到 Mistral API。"""
        if not self._check_auth():
            self._send_json(401, {"error": {"message": "Invalid API key", "type": "auth_error"}})
            return

        entry = pool.next()
        if not entry:
            self._send_json(503, {"error": {"message": "No API keys available", "type": "server_error"}})
            return

        key = entry["key"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        try:
            if stream:
                resp = requests.post(f"{MISTRAL_API}/chat/completions",
                                    json={**body, "stream": True},
                                    headers=headers, timeout=120, stream=True)
            else:
                resp = requests.post(f"{MISTRAL_API}/chat/completions",
                                    json=body, headers=headers, timeout=120)

            if resp.status_code == 429:
                pool.mark_429(key)
                # 重试下一个 key
                self._proxy_chat(body, stream)
                return

            if resp.status_code != 200:
                pool.mark_429(key)
                self._send_json(resp.status_code, resp.json())
                return

            pool.mark_ok(key)

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
            else:
                self._send_json(200, resp.json())

        except requests.exceptions.ConnectionError:
            self._send_json(502, {"error": {"message": "Upstream connection error", "type": "server_error"}})
        except Exception as e:
            self._send_json(500, {"error": {"message": str(e), "type": "server_error"}})

    def do_GET(self):
        if self.path == "/v1/models" or self.path == "/models":
            if not self._check_auth():
                self._send_json(401, {"error": {"message": "Invalid API key"}})
                return
            # 返回 Mistral 可用模型
            entry = pool.next()
            if not entry:
                self._send_json(503, {"error": {"message": "No keys"}})
                return
            try:
                resp = requests.get(f"{MISTRAL_API}/models",
                                  headers={"Authorization": f"Bearer {entry['key']}"},
                                  timeout=30)
                self._send_json(200, resp.json())
            except Exception as e:
                self._send_json(500, {"error": {"message": str(e)}})

        elif self.path == "/health":
            self._send_json(200, {"status": "ok", "keys": pool.count,
                                  "available": pool.available,
                                  "timestamp": datetime.now().isoformat()})

        elif self.path == "/" or self.path == "":
            self._send_json(200, {"service": "mistral2api gateway",
                                  "version": "0.3.0",
                                  "endpoints": ["/v1/chat/completions", "/v1/models", "/health"]})

        elif self.path == "/admin/keys":
            self._send_json(200, {"keys": pool.status()})

        else:
            self._send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self):
        if self.path == "/v1/chat/completions" or self.path == "/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            stream = body.get("stream", False)
            self._proxy_chat(body, stream)
        else:
            self._send_json(404, {"error": {"message": "Not found"}})

    def do_DELETE(self):
        if self.path.startswith("/admin/keys/"):
            key = self.path.split("/admin/keys/")[-1]
            pool.remove(key)
            self._send_json(200, {"removed": key})

    def log_message(self, format, *args):
        # 简化日志
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def main():
    # 仅从挂载文件读取：/app/.env + /app/keys.txt（本地则 .env / keys.txt）
    load_dotenv("/app/.env")
    load_dotenv(".env")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8082"))
    cooldown = int(os.getenv("COOLDOWN", "60"))
    proxy = os.getenv("PROXY", "")

    global gateway_keys, pool
    pool = KeyPool(cooldown_seconds=cooldown)

    raw_keys = os.getenv("GATEWAY_KEYS", "")
    if raw_keys:
        gateway_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    for p in ("/app/keys.txt", "keys.txt"):
        if os.path.exists(p):
            try:
                pool.load_from_file(p)
                print(f"loaded {pool.count} keys from {p}")
            except Exception as e:
                print(f"load keys failed {p}: {e}")
            break

    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    server = HTTPServer((host, port), GatewayHandler)
    print(f"gateway listening on http://{host}:{port}")
    print(f"keys: {pool.count} available: {pool.available} auth: {'on' if gateway_keys else 'off'}")
    server.serve_forever()


if __name__ == "__main__":
    main()
