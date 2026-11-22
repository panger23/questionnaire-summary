"""任务管理：一次上传 = 一个任务，后台线程处理，结果留在内存。

数据不出本机：PDF 只存在内存里，处理时写一个临时文件，处理完立刻删除。
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..engine.pipeline import PageResult, process_pdf


@dataclass
class Job:
    id: str
    filename: str
    pdf: bytes
    use_llm: bool
    llm_scope: str = "all"
    status: str = "queued"  # queued / running / done / error
    stage: str = "排队中"
    pages_done: int = 0
    total_pages: int = 0
    llm_done: int = 0
    llm_total: int = 0
    results: list[PageResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    error: str = ""
    created: float = field(default_factory=time.time)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, pdf: bytes, use_llm: bool,
               llm_scope: str = "all") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, pdf=pdf,
                  use_llm=use_llm, llm_scope=llm_scope)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


STORE = JobStore()


def start_job(job: Job, client=None) -> None:
    threading.Thread(target=_run, args=(job, client), daemon=True).start()


def _run(job: Job, client) -> None:
    job.status = "running"
    job.stage = "正在识别"
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="qsum_")
        with os.fdopen(fd, "wb") as handle:
            handle.write(job.pdf)

        def on_page(done: int, total: int) -> None:
            job.pages_done = done
            job.total_pages = total
            job.stage = "正在识别"

        def on_llm(done: int, total: int) -> None:
            job.stage = "LLM 复核中"
            job.llm_done = done
            job.llm_total = total

        job.results = process_pdf(
            tmp_path,
            progress=on_page,
            llm_client=client,
            llm_progress=on_llm if client is not None else None,
            llm_scope=job.llm_scope,
        )
        if client is not None:
            job.meta = {
                "mode": ("纯代码 + LLM 复核（仅可疑页，%s）" if job.llm_scope == "flagged"
                         else "纯代码 + LLM 复核（全量，%s）") % client.model,
                "model": client.model,
                "llm_calls": client.usage.calls,
                "llm_failures": client.usage.failures,
                "prompt_tokens": client.usage.prompt_tokens,
                "completion_tokens": client.usage.completion_tokens,
                "cost": client.estimated_cost(),
            }
        else:
            job.meta = {"mode": "纯代码模式（离线，未启用 LLM 复核）"}
        if client is not None and getattr(client, "channel_error", ""):
            job.meta["llm_error"] = client.channel_error
            job.meta["mode"] = "纯代码模式（LLM 复核失败，已自动退回）"
        job.stage = "完成"
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - 任何异常都要反馈到界面上
        job.error = f"{type(exc).__name__}: {exc}"
        job.status = "error"
        job.stage = "失败"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
