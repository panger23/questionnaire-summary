"""问卷扫描件汇总引擎。"""
__version__ = "1.1.3"


from datetime import datetime as _dt

# 进程启动时（模块被 import 的那一刻）就固定下来，之后 pull 到新代码也不会变——
# 这样"页脚显示的时间早于你 pull 的时间"就说明服务器还跑着旧进程。
_PROCESS_STARTED = _dt.now()


def build_stamp() -> str:
    """给界面/启动窗口显示的一句话：当前进程是哪个版本、什么时候启动的。"""
    return f"v{__version__} · 服务启动于 {_PROCESS_STARTED:%Y-%m-%d %H:%M}"
