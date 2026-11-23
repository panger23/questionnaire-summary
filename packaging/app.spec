# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（Windows / macOS 通用）。

用法：
    pyinstaller --clean --noconfirm build/app.spec

产物：dist/HuiJuan/  （onedir 形式：启动快、杀软误报少）
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

APP_NAME = "HuiJuan"

# spec 文件和入口脚本不在同一层：以项目根目录为基准解析所有相对路径
ROOT = Path(SPECPATH).resolve().parent   # noqa: F821 - SPECPATH 由 PyInstaller 注入
os.chdir(ROOT)

hiddenimports = [
    # uvicorn 靠字符串动态加载这些模块，PyInstaller 静态分析看不到
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # 上传表单解析
    "multipart",
    "multipart.multipart",
    # OCR 与推理引擎
    "rapidocr_onnxruntime",
    "onnxruntime.capi._pybind_state",
]

datas = [(str(ROOT / "src/web/static"), "src/web/static")]
datas += collect_data_files("rapidocr_onnxruntime")  # 三个 .onnx 模型 + 字典

binaries = collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("cv2")

excludes = [
    "tkinter", "matplotlib", "IPython", "pytest", "notebook",
    "PyQt5", "PySide2", "PySide6", "wx",
    "pymupdf4llm",          # PDF 转 Markdown，用不到
]

a = Analysis(
    [str(ROOT / "app_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=True,  # 控制台里显示访问地址；关掉窗口即停止服务
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
