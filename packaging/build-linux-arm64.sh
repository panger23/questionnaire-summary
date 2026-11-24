#!/usr/bin/env bash
# 在 arm64 Linux（容器里）构建「信创 / 银河麒麟」免安装包。
#
# 用法（在仓库根目录的手机/服务器上，容器内执行）：
#   docker run --rm -v "$PWD":/src -w /src quay.io/pypa/manylinux_2_28_aarch64:latest \
#       bash packaging/build-linux-arm64.sh
#
# 为什么用 manylinux_2_28：它把 glibc 下限锁在 2.28，
# 而银河麒麟桌面 V10 SP1（Debian 11 底座）是 glibc 2.31，所以能跑；
# 反过来（在 glibc 2.3x 的新系统上构建）就会在麒麟上报 GLIBC 版本过低。
#
# 为什么不直接用镜像里的 /opt/python：manylinux 的 CPython 是静态编译的（没有
# libpython.so），PyInstaller 明确要求共享库，所以这里额外下载一份
# python-build-standalone（带 libpython3.11.so，且同样是 glibc 2.17 打底）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- Python 3.11（python-build-standalone，带共享库） ----
PBS_TAG="${PBS_TAG:-20261123}"
PBS_VER="${PBS_VER:-3.11.18}"
PBS_DIR="${PBS_DIR:-/opt/pbs}"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/cpython-${PBS_VER}+${PBS_TAG}-aarch64-unknown-linux-gnu-install_only.tar.gz"

PY="${PYTHON_BIN:-}"
if [ -z "$PY" ]; then
  if [ ! -x "$PBS_DIR/python/bin/python3" ]; then
    echo "== 下载 python-build-standalone ${PBS_VER}（${PBS_TAG}）"
    mkdir -p "$PBS_DIR"
    # 用镜像自带的 python 下载，不依赖 curl/wget
    BOOT_PY="$(command -v python3.11 || command -v python3)"
    "$BOOT_PY" - "$PBS_URL" /tmp/pbs.tar.gz <<'PY'
import sys, urllib.request
url, dest = sys.argv[1], sys.argv[2]
print("GET", url)
urllib.request.urlretrieve(url, dest)
PY
    tar -xzf /tmp/pbs.tar.gz -C "$PBS_DIR"
    rm -f /tmp/pbs.tar.gz
  fi
  PY="$PBS_DIR/python/bin/python3"
fi
"$PY" -V
echo "== 构建机：$(uname -m)，Python：$("$PY" -V 2>&1)，glibc：$(ldd --version 2>/dev/null | head -1)"

echo "== 安装依赖"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt
"$PY" -m pip install -q pyinstaller

# 网页端不做任何 GUI 操作：换成 headless 版 OpenCV，
# 免得成品在没装 libGL / X11 的内网机器上 import cv2 就崩。
if "$PY" -m pip show opencv-python >/dev/null 2>&1; then
  "$PY" -m pip uninstall -q -y opencv-python
fi
"$PY" -m pip install -q opencv-python-headless

echo "== 依赖自检（PyInstaller 的静态分析看不到部分动态 import，这里先跑一遍真导入）"
"$PY" - <<'PY'
import cv2, numpy, pymupdf, fastapi, openpyxl, xlsxwriter, onnxruntime
from rapidocr_onnxruntime import RapidOCR  # noqa: F401
print("依赖自检 OK：cv2", cv2.__version__,
      "| numpy", numpy.__version__,
      "| onnxruntime", onnxruntime.__version__)
PY

echo "== PyInstaller 打包"
"$PY" -m PyInstaller --clean --noconfirm packaging/app.spec

echo "== 成品依赖的 glibc 版本（必须 ≤ 2.31，麒麟 V10 SP1 是 2.31）"
if command -v objdump >/dev/null 2>&1; then
  find dist/LuShen -type f \( -name '*.so' -o -name '*.so.*' -o -perm -u+x \) -print0 2>/dev/null \
    | xargs -0 -r objdump -T 2>/dev/null \
    | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -u -V | tail -n 3 || true
fi

echo "== 冒烟测试：起服务、请求首页"
./dist/LuShen/LuShen --no-browser --port 8799 >/tmp/lushen-smoke.log 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

"$PY" - <<'PY'
import time
import urllib.request

for _ in range(90):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8799/", timeout=3) as resp:
            if resp.status == 200:
                print("冒烟测试通过：首页返回 200")
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 服务还没起来
        time.sleep(1)
raise SystemExit("冒烟测试失败：打包后的程序没能正常起来")
PY

kill "$pid" 2>/dev/null || true
trap - EXIT
tail -n 5 /tmp/lushen-smoke.log || true

echo "== 组装发布目录"
rm -rf dist/发布
mkdir -p dist/发布/LuShen
cp -a dist/LuShen/. dist/发布/LuShen/
cp packaging/使用说明-linux.txt dist/发布/LuShen/README.txt
cp packaging/start.sh dist/发布/LuShen/start.sh
chmod +x dist/发布/LuShen/LuShen dist/发布/LuShen/start.sh

echo "== 打 zip（用 Python 写，保留可执行位，解压即用）"
"$PY" - <<'PY'
import os
import shutil
import zipfile

out = "dist/LuShen-linux-aarch64.zip"
base = "dist/发布"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for dirpath, _dirnames, filenames in os.walk(base):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            arc = os.path.relpath(full, base).replace(os.sep, "/")
            mode = 0o755 if os.stat(full).st_mode & 0o111 else 0o644
            info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(full, "rb") as src, zf.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)

size = os.path.getsize(out) / 1048576
total = sum(os.path.getsize(os.path.join(d, f))
            for d, _n, fs in os.walk(base) for f in fs) / 1048576
print(f"成品：{out}  压缩包 {size:.1f} MB，解压后 {total:.1f} MB")
PY

echo "== 完成"
