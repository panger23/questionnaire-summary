#!/bin/bash
# 双击启动本地网页。第一次运行会自动装依赖（约 2 分钟），之后秒开。
cd "$(dirname "$0")" || exit 1

if [ ! -x ".venv/bin/python" ]; then
  echo "首次运行：正在创建环境并安装依赖，请稍等（约 2 分钟）…"
  python3 -m venv .venv || { echo "创建虚拟环境失败，请确认已安装 Python 3.11+"; read -r -p "按回车关闭"; exit 1; }
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt || { echo "依赖安装失败"; read -r -p "按回车关闭"; exit 1; }
fi

exec .venv/bin/python -m src.web
