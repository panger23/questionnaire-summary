"""启动本地网页：python -m src.web

打包成 exe 后由 app_entry.py 调用同一个 main()。
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .app import app

HOST = "127.0.0.1"  # 只绑本机，内网里别人访问不到
BASE_PORT = 8765

# 内网统一用 Chrome，优先直接拉起它；找不到再退回系统默认浏览器
CHROME_CANDIDATES = (
    "chrome",
    "google-chrome",
    "google-chrome-stable",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def _free_port(start: int = BASE_PORT, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"端口 {start}~{start + tries} 都被占用了")


def _open_browser(url: str) -> None:
    if os.environ.get("QSUM_NO_BROWSER") == "1":
        return
    for candidate in CHROME_CANDIDATES:
        try:
            subprocess.Popen([candidate, url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, OSError):
            continue
    webbrowser.open(url)


def _say(*lines: str) -> None:
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:  # 老 Windows 控制台编码不对时不至于崩
            print(line.encode("gbk", "replace").decode("gbk", "replace"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args, _ = parser.parse_known_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    if args.no_browser:
        os.environ["QSUM_NO_BROWSER"] = "1"

    port = args.port or _free_port()
    url = f"http://{HOST}:{port}"
    _say(
        "",
        "  汇卷 · 扫描件智能汇总",
        "  ─────────────────────",
        f"  已启动：{url}",
        "  浏览器没自动打开的话，把上面这行复制到 Chrome 里",
        "  关掉这个窗口就停止服务；文件只在内存里处理，不落盘",
        "",
    )
    threading.Timer(1.0, lambda: _open_browser(url)).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
