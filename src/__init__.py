"""问卷扫描件汇总引擎。"""
__version__ = "1.1.2"


def build_stamp() -> str:
    """代码版本 + 源码最后修改时间。

    用来判断"当前跑的是不是最新代码"：服务是启动时加载代码的，
    改了代码不重启就还是旧逻辑；界面/窗口里显示的时间能一眼看出来。
    """
    from datetime import datetime
    from pathlib import Path

    newest = 0.0
    for path in Path(__file__).parent.rglob("*.py"):
        newest = max(newest, path.stat().st_mtime)
    when = datetime.fromtimestamp(newest).strftime("%m-%d %H:%M")
    return f"v{__version__}（代码更新于 {when}）"
