#!/usr/bin/env bash
# 鲁小审 · 汇卷（扫描件智能汇总）—— Linux 启动脚本
#
# 第一次用：在这个文件夹里右键「在终端中打开」，然后执行
#     bash start.sh
# 之后也可以直接双击（前提是这个文件有可执行权限：
#     chmod +x start.sh LuShen
# ）
#
# 关掉终端窗口（或按 Ctrl+C）就停止服务。

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# 有的麒麟版本不让普通用户从 U 盘直接执行，这里给个明确提示而不是"闪退"
if [ ! -x "./LuShen" ]; then
    chmod +x ./LuShen 2>/dev/null || true
fi
if [ ! -x "./LuShen" ]; then
    echo "!! 没有执行权限，请先运行：chmod +x LuShen start.sh"
    exit 1
fi

exec ./LuShen "$@"
