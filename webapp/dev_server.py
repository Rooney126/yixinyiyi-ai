"""本地开发服务器（仅开发验证用，部署不含此文件）：静态服务本目录，把 /api/* 与 /stats 反向代理到
线上（透传 query 与 key）。用法：python dev_server.py → http://127.0.0.1:8000（DEV_UPSTREAM/DEV_PORT 可覆盖）"""
import os
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("DEV_UPSTREAM", "https://your-server.example").rstrip("/")
ROOT, PORT = os.path.dirname(os.path.abspath(__file__)), int(os.environ.get("DEV_PORT") or 8000)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith(("/api/", "/stats")):   # /stats 一起反代：口令探测用它
            self._proxy()   # self.path 自带 query string，key 随之透传
        else:
            super().do_GET()

    def _proxy(self):
        # 带上客户端 UA 转发：线上有 Cloudflare，python-urllib 默认 UA 会被 1010 拒掉
        req = urllib.request.Request(UPSTREAM + self.path, headers={
            "User-Agent": self.headers.get("User-Agent") or "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
                ctype, code = r.headers.get("Content-Type", "application/json"), r.status
        except urllib.error.HTTPError as e:
            body, ctype, code = e.read(), e.headers.get("Content-Type", "text/plain"), e.code
        except Exception as e:
            body = ('{"error": "dev_server proxy failed: %s"}' % e).encode("utf-8")
            ctype, code = "application/json; charset=utf-8", 502
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("http://127.0.0.1:%d  (/api/* -> %s)" % (PORT, UPSTREAM))
    ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=ROOT)).serve_forever()
