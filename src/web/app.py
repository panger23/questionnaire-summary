"""本地网页服务。

只监听 127.0.0.1 —— 别人在内网里也访问不到，Key 和数据都只在你本机。
"""
from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

from ..engine.pipeline import average, group_by_branch, sum_scores
from ..export.xlsx import build_workbook
from ..llm.client import VisionClient
from ..llm.config import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    load_config,
    masked_key,
    save_config,
)
from .jobs import STORE, start_job

def _static_dir() -> Path:
    """源码运行时在 src/web/static；PyInstaller 打包后在 _MEIPASS/src/web/static。"""
    here = Path(__file__).parent / "static"
    if (here / "index.html").exists():
        return here
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / "src" / "web" / "static"


STATIC_DIR = _static_dir()

app = FastAPI(title="鲁审 · 汇卷")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB


# ---------------------------------------------------------------- 数据序列化
def _summary(results) -> list[dict]:
    out = []
    for name, items in group_by_branch(results):
        out.append({
            "branch": name,
            "assistant": items[0].header.assistant,
            "count": len(items),
            "avg": average(items),
            "scores": [r.total for r in items],
        })
    return out


def _row(r) -> dict:
    return {
        "page": r.page,
        "branch": r.header.project,
        "assistant": r.header.assistant,
        "lead": r.header.lead,
        "scores": r.scores,
        "total": r.total,
        "llm_scores": r.llm_scores,
        "llm_marks": r.llm_marks,
        "llm_total": r.llm_total,
        "agree": r.agree,
        "flags": [{"short": f.short, "detail": f.detail} for f in r.flags],
        "adopted": r.adopted,
        "grid_ok": r.grid_ok,
    }


def _rows(results) -> list[dict]:
    return [_row(r) for r in results]


def _add_flag(r, text: str) -> None:
    if text not in r.flags:
        r.flags.append(text)


def _find(job, page: int):
    for r in job.results:
        if r.page == page:
            return r
    return None


def _snapshot(job) -> dict:
    flagged = [r for r in job.results if r.flags or r.total is None]
    stats = {
        "pages": len(job.results),
        "blank_pages": job.blank_pages,
        "total_pages": job.total_pages or len(job.results),
        "flagged": len(flagged),
        "flagged_pages": [r.page for r in flagged],
        "mode": job.meta.get("mode", ""),
        "cost": job.meta.get("cost"),
        "llm_error": job.meta.get("llm_error", ""),
        "llm_warning": job.meta.get("llm_warning", ""),
        "agree": sum(1 for r in job.results if r.agree is True),
        "disagree": sum(1 for r in job.results if r.agree is False),
    }
    return {"summary": _summary(job.results), "stats": stats}


# ---------------------------------------------------------------- 页面
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, f"页面文件缺失：{page}")
    return HTMLResponse(page.read_text("utf-8"))


# ---------------------------------------------------------------- 上传与任务
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), mode: str = Form("code")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "文件是空的")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "文件超过 200MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "这不是 PDF 文件")

    if mode not in ("code", "flagged", "all"):
        raise HTTPException(400, "mode 参数不对，应为 code / flagged / all")

    client = None
    llm_scope = "all"
    if mode != "code":
        cfg = load_config()
        if not cfg.get("api_key"):
            raise HTTPException(400, "启用了 LLM，但还没有填写 API Key，请先去「设置」里填")
        client = VisionClient(cfg["api_base"], cfg["api_key"], cfg["model"])
        llm_scope = "flagged" if mode == "flagged" else "all"

    job = STORE.create(file.filename or "upload.pdf", data,
                       use_llm=client is not None, llm_scope=llm_scope)
    start_job(job, client)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return {
        "status": job.status,
        "stage": job.stage,
        "pages_done": job.pages_done,
        "total_pages": job.total_pages,
        "llm_done": job.llm_done,
        "llm_total": job.llm_total,
        "error": job.error,
    }


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status != "done":
        raise HTTPException(409, "任务还没完成")
    return {"job_id": job.id, "filename": job.filename, "mode": job.meta.get("mode", ""),
            "rows": _rows(job.results), **_snapshot(job)}


# ---------------------------------------------------------------- 编辑
class EditIn(BaseModel):
    page: int
    row: int
    score: int | None = None


class PageIn(BaseModel):
    page: int


@app.post("/api/jobs/{job_id}/edit")
def edit_score(job_id: str, body: EditIn):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    r = _find(job, body.page)
    if r is None or not r.readings:
        raise HTTPException(404, "找不到这一页")
    if not 0 <= body.row < len(r.readings):
        raise HTTPException(400, "行号超出范围")
    if body.score is not None and not 1 <= body.score <= 10:
        raise HTTPException(400, "分数必须在 1~10 之间")
    r.readings[body.row].score = body.score
    r.total = sum_scores(r.readings)
    _add_flag(r, "已人工修改分数")
    return {"row": _row(r), **_snapshot(job)}


@app.post("/api/jobs/{job_id}/adopt")
def adopt_llm(job_id: str, body: PageIn):
    """采用 LLM 通道给出的 5 个分数。"""
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    r = _find(job, body.page)
    if r is None or not r.llm_scores:
        raise HTTPException(404, "这一页没有 LLM 结果")
    for i, value in enumerate(r.llm_scores):
        if value is not None and i < len(r.readings):
            r.readings[i].score = value
    r.total = sum_scores(r.readings)
    _add_flag(r, "已人工采用 LLM 通道的分数")
    return {"row": _row(r), **_snapshot(job)}


# ---------------------------------------------------------------- 原图与下载
@app.get("/api/jobs/{job_id}/page/{page_no}")
def page_image(job_id: str, page_no: int, width: int = 1300):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    r = _find(job, page_no)
    if r is None:
        raise HTTPException(404, "找不到这一页")

    doc = pymupdf.open(stream=job.pdf, filetype="pdf")
    try:
        page = doc[page_no - 1]
        pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    finally:
        doc.close()

    if len(r.grid_h) >= 11 and len(r.grid_v) >= 15:
        y0, y1 = max(0, r.grid_h[2] - 10), r.grid_h[10] + 10
        x0, x1 = max(0, r.grid_v[0] - 10), r.grid_v[13] + 10
        gray = gray[y0:y1, x0:x1]

    image = Image.fromarray(gray)
    if image.width > width:
        image = image.resize((width, int(image.height * width / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return Response(buffer.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "max-age=600"})


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status != "done":
        raise HTTPException(409, "任务还没完成")
    stem = Path(job.filename).stem
    name = f"{stem}_汇总_{time.strftime('%Y%m%d_%H%M')}.xlsx"
    out = Path(tempfile.gettempdir()) / name
    build_workbook(job.results, job.filename, out, meta=job.meta)
    return FileResponse(str(out), filename=name,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------------------------------------------------------- 设置
class ConfigIn(BaseModel):
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None


@app.get("/api/config")
def get_config():
    cfg = load_config()
    return {
        "api_base": cfg["api_base"],
        "model": cfg["model"],
        "has_key": bool(cfg["api_key"]),
        "key_masked": masked_key(cfg["api_key"]),
        "default_api_base": DEFAULT_API_BASE,
        "default_model": DEFAULT_MODEL,
    }


@app.post("/api/config")
def set_config(body: ConfigIn):
    cfg = load_config()
    if body.api_base is not None and body.api_base.strip():
        cfg["api_base"] = body.api_base.strip()
    if body.model is not None and body.model.strip():
        cfg["model"] = body.model.strip()
    if body.api_key:  # 空字符串表示"不修改"
        cfg["api_key"] = body.api_key.strip()
    save_config(cfg)
    return get_config()


@app.post("/api/config/clear")
def clear_key():
    cfg = load_config()
    cfg["api_key"] = ""
    save_config(cfg)
    return get_config()


@app.post("/api/config/test")
def test_config():
    cfg = load_config()
    if not cfg.get("api_key"):
        return {"ok": False, "message": "还没有填写 API Key"}
    try:
        client = OpenAI(base_url=cfg["api_base"], api_key=cfg["api_key"],
                        timeout=25, max_retries=0)
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": "只回复两个字：收到"}],
            temperature=0,
        )
        reply = (resp.choices[0].message.content or "").strip()[:20]
        return {"ok": True, "message": f"连接成功，{cfg['model']} 回复：{reply}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"{type(exc).__name__}: {str(exc)[:200]}"}
