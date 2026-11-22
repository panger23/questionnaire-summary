"""串起整条流水线：PDF → 每页的分数与表头字段（可选 LLM 交叉校验）。

双通道的分工：
  通道 A（墨迹）——确定性、可复现，负责判"勾在哪一格"，但它分不清 √ 和 ×。
  通道 B（LLM）——负责语义判断：多标记时哪个是 √、哪个是作废的 ×；也处理"直接写数字"。
两者不一致时，只有在"代码通道自己就看到两格墨量接近"时才允许 LLM 的结论生效，
否则保持代码结论。理由：勾的尾巴扫过竖线也会造成两格有墨，那不是真的两个标记。
"""
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
    top_two,
)
from .render import count_pages, ink_mask, render_pages

Progress = Callable[[int, int], None]

# 两格墨量比达到这个值，才认为"两个格子里真有东西"（而不是勾尾巴扫过）
AUTO_ADOPT_RATIO = 0.5


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
    llm_marks: list[int] | None = None
    llm_total: int | None = None
    agree: bool | None = None
    adopted: list[str] = field(default_factory=list)
    grid_h: list[int] = field(default_factory=list)
    grid_v: list[int] = field(default_factory=list)

    @property
    def scores(self) -> list[int | None]:
        return [r.score for r in self.readings]


def crop_rows_for_llm(gray: np.ndarray, grid: Grid) -> list[np.ndarray]:
    """按行裁出给模型看的图：每行只含 10 个分值格，一行一张图。

    实测整块发会翻车（模型在窄列网格上会输出 10/9/8/7/6 这种伪规律），
    一行一张图 + 格子上方印分值标签之后，成功率接近 100%。
    """
    rows: list[np.ndarray] = []
    x0, x1 = grid.v[SCORE_COL_START] - 6, grid.v[SCORE_COL_END] + 6
    for r in INDICATOR_ROWS:
        y0, y1 = grid.h[r] - 4, grid.h[r + 1] + 4
        sub = gray[y0:y1, x0:x1]
        rows.append(np.repeat(sub[:, :, None], 3, axis=2))
    return rows


def _apply_llm(r: PageResult, scores: list[int | None], marks: list[int]) -> None:
    r.llm_scores = scores
    r.llm_marks = marks
    if all(s is None for s in scores):
        r.flags.append("LLM 没有给出任何分数")
        return

    diffs: list[str] = []
    for i, (a, b) in enumerate(zip(r.scores, scores)):
        if a == b:
            continue
        name = INDICATOR_NAMES[i]
        if a is None or b is None:
            diffs.append(f"{name}：代码 {a if a is not None else '?'} / "
                         f"LLM {b if b is not None else '?'}")
            continue
        ratios = r.readings[i].ratios
        best, second, _best_i, second_i = top_two(ratios)
        ratio = second / best if best > 0 else 0.0
        second_score = 10 - second_i
        if ratio >= AUTO_ADOPT_RATIO and marks[i] >= 2 and b == second_score:
            r.readings[i].score = b
            r.adopted.append(f"{name} {a}→{b}")
            continue
        diffs.append(
            f"{name}：代码 {a} / LLM {b}"
            f"（两格墨量 {best:.2%} / {second:.2%}，更像笔迹尾巴，未自动采用）"
        )

    r.total = sum_scores(r.readings)
    if all(s is not None for s in scores):
        r.llm_total = int(sum(s for s in scores if s is not None))

    if r.adopted:
        r.flags.append("已按『多个标记取√、×作废』规则采用 LLM 判断：" + "；".join(r.adopted))
    if diffs:
        r.flags.append("双通道不一致 —— " + "；".join(diffs))
    # 采用过 LLM 判断的页，本质仍是"两通道不一致"，只是已经按规则裁决过了
    r.agree = not (diffs or r.adopted)


def process_pdf(pdf_path: str, dpi: int = 300, progress: Progress | None = None,
                llm_client=None, workers: int = 4,
                llm_progress: Callable[[int, int], None] | None = None,
                llm_scope: str = "all") -> list[PageResult]:
    """处理整个 PDF。llm_client 不为 None 时，额外跑一遍 LLM 通道做交叉校验。"""
    results: list[PageResult] = []
    jobs: list[tuple[int, int, np.ndarray]] = []
    header_cache: dict[tuple[str, str, str], Header] = {}
    total_pages = count_pages(pdf_path)

    for page_no, gray in render_pages(pdf_path, dpi=dpi):
        mask = ink_mask(gray)
        grid = find_grid(mask)

        if not grid.ok:
            results.append(
                PageResult(page=page_no, header=Header("", "", ""), readings=[],
                           total=None, grid_ok=False, flags=[grid.note],
                           grid_h=list(grid.h), grid_v=list(grid.v))
            )
            if progress:
                progress(page_no, total_pages)
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
                       total=sum_scores(readings), grid_ok=True, flags=flags,
                       grid_h=list(grid.h), grid_v=list(grid.v))
        )
        # llm_scope = "flagged" 时，只把这一批里有问题的页送去复核，省时省钱
        if llm_client is not None and (llm_scope == "all" or flags or sum_scores(readings) is None):
            for row_idx, rgb in enumerate(crop_rows_for_llm(gray, grid)):
                jobs.append((len(results) - 1, row_idx, rgb))
        if progress:
            progress(page_no, total_pages)

    if llm_client is not None and jobs:
        collected: dict[tuple[int, int], object] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(llm_client.read_row, rgb): (pi, ri)
                       for pi, ri, rgb in jobs}
            done = 0
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    collected[key] = fut.result()
                except Exception:  # noqa: BLE001
                    collected[key] = None
                done += 1
                if llm_progress is not None and (done % 10 == 0 or done == len(futures)):
                    llm_progress(done, len(futures))
        by_page: dict[int, dict] = {}
        for (pi, ri), verdict in collected.items():
            slot = by_page.setdefault(pi, {"scores": [None] * len(INDICATOR_ROWS),
                                           "marks": [0] * len(INDICATOR_ROWS)})
            if verdict is not None:
                slot["scores"][ri] = getattr(verdict, "score", None)
                slot["marks"][ri] = getattr(verdict, "marks", 0)
        for pi, slot in by_page.items():
            _apply_llm(results[pi], slot["scores"], slot["marks"])

        # 只有"一次都没成功"才判定整批不可用；偶发超时只作为提示，
        # 逐页的失败仍保留在标黄信息里。
        total_calls = len(jobs)
        failures = llm_client.usage.failures
        if failures >= total_calls:
            llm_client.channel_error = (llm_client.last_error
                                        or "LLM 通道不可用（额度/密钥/网络问题）")
            for r in results:
                r.flags = [f for f in r.flags if f != "LLM 没有给出任何分数"]
        elif failures:
            llm_client.partial_warning = (
                f"{failures}/{total_calls} 次 LLM 调用失败（多为超时/限流），"
                f"这些行按代码通道结果处理并标黄")

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
