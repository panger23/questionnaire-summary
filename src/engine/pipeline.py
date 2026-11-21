"""串起整条流水线：PDF → 每页的分数与表头字段（可选 LLM 交叉校验）。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grid import Grid, find_grid
from .header import Header, header_crop, parse_header
from .marks import (
    INDICATOR_NAMES,
    INDICATOR_ROWS,
    SCORE_COL_END,
    SCORE_COL_START,
    RowReading,
    read_indicator_rows,
    sum_scores,
)
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


def crop_rows_for_llm(gray: np.ndarray, grid: Grid) -> list[np.ndarray]:
    """按行裁出给模型看的图：每行只含 10 个分值格，一行一张图。

    实测整块发会翻车（模型在最左列会输出等差数列），一行一张图则全对。
    """
    rows: list[np.ndarray] = []
    x0, x1 = grid.v[SCORE_COL_START] - 6, grid.v[SCORE_COL_END] + 6
    for r in INDICATOR_ROWS:
        y0, y1 = grid.h[r] - 4, grid.h[r + 1] + 4
        sub = gray[y0:y1, x0:x1]
        rows.append(np.repeat(sub[:, :, None], 3, axis=2))
    return rows


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
            for row_idx, rgb in enumerate(crop_rows_for_llm(gray, grid)):
                jobs.append((len(results) - 1, row_idx, rgb))
        if progress:
            progress(page_no, page_no)

    if llm_client is not None and jobs:
        collected: dict[tuple[int, int], int | None] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(llm_client.read_row, rgb): (pi, ri)
                       for pi, ri, rgb in jobs}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    col = fut.result()
                except Exception:  # noqa: BLE001
                    col = None
                collected[key] = col
        by_page: dict[int, list[int | None]] = {}
        for (pi, ri), score in collected.items():
            by_page.setdefault(pi, [None] * len(INDICATOR_ROWS))[ri] = score
        for pi, scores in by_page.items():
            _apply_llm(results[pi], scores)

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
