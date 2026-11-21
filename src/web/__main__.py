"""启动本地网页：python -m src.web"""
from __future__ import annotations

import socket
import threading
import webbrowser

import uvicorn

from .app import app

HOST = "127.0.0.1"  # 只绑本机，内网里别人访问不到
BASE_PORT = 8765


def _free_port(start: int = BASE_PORT, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"端口 {start}~{start + tries} 都被占用了")


def main() -> None:
    port = _free_port()
    url = f"http://{HOST}:{port}"
    print()
    print("  审计问卷汇总 · 已启动")
    print(f"  地址：{url}")
    print("  关掉这个窗口就停止服务；数据只在内存里，不落盘")
    print()
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
