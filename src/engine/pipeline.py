"""串起整条流水线：PDF → 每页的分数与表头字段。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grid import Grid, find_grid
from .header import Header, header_crop, parse_header
from .marks import RowReading, read_indicator_rows, sum_scores
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

    @property
    def scores(self) -> list[int | None]:
        return [r.score for r in self.readings]


def process_pdf(pdf_path: str, dpi: int = 300, progress: Progress | None = None) -> list[PageResult]:
    """处理整个 PDF，返回逐页结果（按页序）。"""
    results: list[PageResult] = []
    cache: dict[tuple[str, str, str], Header] = {}
    total_pages = 0

    for page_no, gray in render_pages(pdf_path, dpi=dpi):
        total_pages = page_no
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
        flags = [f"{r.row - 3}项：{msg}" for r in readings for msg in r.flags]

        crop = header_crop(gray, grid)
        header = parse_header(crop)
        cached = cache.get(header.key())
        if cached is not None:
            header = cached
        else:
            cache[header.key()] = header
            if not header.project:
                flags.append("表头项目名称未识别出来")
            if not header.assistant:
                flags.append("表头组长助手未识别出来")

        results.append(
            PageResult(page=page_no, header=header, readings=readings,
                       total=sum_scores(readings), grid_ok=True, flags=flags)
        )
        if progress:
            progress(page_no, page_no)

    if progress and total_pages:
        progress(total_pages, total_pages)
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
