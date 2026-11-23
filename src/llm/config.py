"""API 配置的读写。

API Key 只存在本机配置文件里，绝不进代码、绝不进仓库（.gitignore 已挡住 配置.json）。
Step 3 的网页设置页会调用同一个 save_config()，保证两边行为一致。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"

CONFIG_NAME = "配置.json"


def config_path() -> Path:
    """配置文件位置。

    - 环境变量 QSUM_CONFIG 优先
    - 打包成 exe 后：配置文件放在 exe 同目录（用户看得见、可备份）
    - 源码运行：项目根目录的 配置.json
    """
    env = os.environ.get("QSUM_CONFIG")
    if env:
        return Path(env).expanduser()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / CONFIG_NAME
    return Path(__file__).resolve().parents[2] / CONFIG_NAME


def load_config() -> dict:
    path = config_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data.setdefault("api_base", DEFAULT_API_BASE)
    data.setdefault("model", DEFAULT_MODEL)
    data.setdefault("api_key", os.environ.get("DASHSCOPE_API_KEY", ""))
    return data


def save_config(cfg: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        "api_base": cfg.get("api_base") or DEFAULT_API_BASE,
        "model": cfg.get("model") or DEFAULT_MODEL,
        "api_key": cfg.get("api_key", ""),
    }
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), "utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def masked_key(key: str) -> str:
    """给界面显示的掩码形式：sk-****abcd。"""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:3]}****{key[-4:]}"
