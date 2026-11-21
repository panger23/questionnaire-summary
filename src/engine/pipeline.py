"""串起整条流水线：PDF → 每页的分数与表头字段（可选 LLM 交叉校验）。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grid import Grid, find_grid
from .header import Header, header_crop, parse_header
from .marks import INDICATOR_NAMES, RowReading, read_indicator_rows, sum_scores
from .render import ink_mask, render_pages

Progress = Callable[[int, int], None]


@dataclass
class PageResult:
    page: int
    header: Header
    readings: list[RowReading]
    total: int | None
    grid_ok: bool
    flags: list[str] = field(default_factory=list)
    # LLM 通道（未启用时为 None）
    llm_scores: list[int | None] | None = None
    llm_total: int | None = None
    agree: bool | None = None

    @property
    def scores(self) -> list[int | None]:
        return [r.score for r in self.readings]


def crop_for_llm(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """裁给模型看的区域：表头两行 + 5 个指标行，去掉备注列、小计列和手写总分行。"""
    y0, y1 = grid.h[2], grid.h[9]
    x0, x1 = grid.v[0], grid.v[12]
    pad = 6
    sub = gray[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
    return np.repeat(sub[:, :, None], 3, axis=2)


def _apply_llm(r: PageResult, scores: list[int | None] | None) -> None:
    r.llm_scores = scores
    if scores is None:
        r.flags.append("LLM 调用失败或返回格式异常")
        return
    if all(s is None for s in scores):
        r.flags.append("LLM 没有给出任何分数")
        return
    if all(s is not None for s in scores):
        r.llm_total = int(sum(s for s in scores if s is not None))
    diffs = [
        f"{INDICATOR_NAMES[i]}：代码 {a if a is not None else '?'} / "
        f"LLM {b if b is not None else '?'}"
        for i, (a, b) in enumerate(zip(r.scores, scores))
        if a != b
    ]
    if diffs:
        r.agree = False
        r.flags.append("双通道不一致 —— " + "；".join(diffs))
    else:
        r.agree = True


def process_pdf(pdf_path: str, dpi: int = 300, progress: Progress | None = None,
                llm_client=None, workers: int = 6) -> list[PageResult]:
    """处理整个 PDF。llm_client 不为 None 时，额外跑一遍 LLM 通道做交叉校验。"""
    results: list[PageResult] = []
    jobs: list[tuple[int, np.ndarray]] = []
    header_cache: dict[tuple[str, str, str], Header] = {}

    for page_no, gray in render_pages(pdf_path, dpi=dpi):
        mask = ink_mask(gray)
        grid = find_grid(mask)

        if not grid.ok:
            results.append(
                PageResult(page=page_no, header=Header("", "", ""), readings=[],
                           total=None, grid_ok=False, flags=[grid.note])
            )
            if progress:
                progress(page_no, page_no)
            continue

        readings = read_indicator_rows(mask, grid)
        flags = [f"{INDICATOR_NAMES[r.row - 4]}：{msg}" for r in readings for msg in r.flags]

        header = parse_header(header_crop(gray, grid))
        cached = header_cache.get(header.key())
        if cached is not None:
            header = cached
        else:
            header_cache[header.key()] = header
            if not header.project:
                flags.append("表头项目名称未识别出来")
            if not header.assistant:
                flags.append("表头组长助手未识别出来")

        results.append(
            PageResult(page=page_no, header=header, readings=readings,
                       total=sum_scores(readings), grid_ok=True, flags=flags)
        )
        if llm_client is not None:
            jobs.append((len(results) - 1, crop_for_llm(gray, grid)))
        if progress:
            progress(page_no, page_no)

    if llm_client is not None and jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for idx, scores in pool.map(lambda job: (job[0], llm_client.read_scores(job[1])), jobs):
                _apply_llm(results[idx], scores)

    return results


def group_by_branch(results: list[PageResult]) -> list[tuple[str, list[PageResult]]]:
    """按分行分组，顺序 = 在 PDF 中首次出现的顺序。"""
    order: list[str] = []
    buckets: dict[str, list[PageResult]] = {}
    for r in results:
        name = r.header.project or "未识别分行"
        if name not in buckets:
            buckets[name] = []
            order.append(name)
        buckets[name].append(r)
    return [(name, buckets[name]) for name in order]


def average(results: list[PageResult]) -> float | None:
    vals = [r.total for r in results if r.total is not None]
    return sum(vals) / len(vals) if vals else None
