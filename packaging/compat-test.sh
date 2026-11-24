#!/usr/bin/env bash
# 在一个"干净的目标系统"里试跑成品：不装任何东西、不联网，
# 只用 bash 的 /dev/tcp 探一下服务首页是否返回 200。
#
# 用法（在干净容器里）：
#   docker run --rm -v "$PWD/dist:/dist:ro" -v "$PWD/packaging:/packaging:ro" \
#       debian:bullseye-slim bash /packaging/compat-test.sh
set -u

echo "== 试跑环境：$(uname -m) / Debian $(cat /etc/debian_version 2>/dev/null) / glibc $(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')"
echo "== 缺失的动态库（应为空）："
missing=$(ldd /dist/LuShen/LuShen 2>&1 | grep -i 'not found' || true)
if [ -n "$missing" ]; then echo "$missing"; else echo "  （无）"; fi

cd /dist/LuShen || exit 1
./LuShen --no-browser --port 8800 >/tmp/app.log 2>&1 &
pid=$!

ok=0
for _i in $(seq 1 90); do
  if exec 3<>/dev/tcp/127.0.0.1/8800 2>/dev/null; then
    printf 'GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n' >&3
    line=$(head -n 1 <&3 || true)
    exec 3>&- 3<&- 2>/dev/null || true
    case "$line" in
      *" 200 "*) ok=1; break ;;
    esac
  fi
  kill -0 "$pid" 2>/dev/null || break   # 进程已经挂了就别等
  sleep 1
done

kill "$pid" 2>/dev/null || true

if [ "$ok" = 1 ]; then
  echo "== 干净环境启动成功：首页返回 200"
  exit 0
fi

echo "== 干净环境启动失败，程序输出（末 80 行）："
tail -n 80 /tmp/app.log 2>/dev/null || echo "（没有输出，可能进程根本没启动）"
exit 1
