"""PyInstaller 打包入口（源码运行时也可以用：python app_entry.py）。"""
from __future__ import annotations

from src.web.__main__ import main

if __name__ == "__main__":
    main()
