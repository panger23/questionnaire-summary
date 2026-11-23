"""串起整条流水线：PDF → 每页的分数与表头字段（可选 LLM 交叉校验）。

双通道的分工：
  通道 A（墨迹）——确定性、可复现，负责判"勾在哪一格"，但它分不清 √ 和 ×。
  通道 B（LLM）——负责语义判断：多标记时哪个是 √、哪个是作废的 ×；也处理"直接写数字"。
两者不一致时，只有在"代码通道自己就看到两格墨量接近"时才允许 LLM 的结论生效，
否则保持代码结论。理由：勾的尾巴扫过竖线也会造成两格有墨，那不是真的两个标记。
"""
from __future__ import annotations

import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grid import Grid, find_grid
from .header import Header, header_crop, parse_header, read_score_layout
from .marks import (
    RowReading,
    default_layout,
    indicator_name,
    read_indicator_rows,
    sum_scores,
    top_two,
)
from .render import count_pages, deskew, estimate_skew, ink_mask, render_pages

Progress = Callable[[int, int], None]

# 两格墨量比达到这个值，才认为"两个格子里真有东西"（而不是勾尾巴扫过）
AUTO_ADOPT_RATIO = 0.5

# 整页墨迹占比低于此值 = 扫描仪夹进来的空白页。
# 实测：正常问卷约 4.2%~4.5%，空白页只有 0.03%~0.04%，差 100 倍。
BLANK_PAGE_MAX_INK = 0.01


@dataclass
class ProcessOutcome:
    results: list["PageResult"]
    blank_pages: list[int]
    total_pages: int


@dataclass(frozen=True)
class Flag:
    """一条待确认项：short 给用户看，detail 是技术依据（悬停显示）。"""
    short: str
    detail: str = ""


@dataclass
class PageResult:
    page: int
    header: Header
    readings: list[RowReading]
    total: int | None
    grid_ok: bool
    flags: list[Flag] = field(default_factory=list)
    # LLM 通道（未启用时为 None）
    llm_scores: list[int | None] | None = None
    llm_marks: list[int] | None = None
    llm_total: int | None = None
    agree: bool | None = None
    adopted: list[str] = field(default_factory=list)
    grid_h: list[int] = field(default_factory=list)
    grid_v: list[int] = field(default_factory=list)
    skew: float = 0.0  # 这一页被纠正的角度，原图预览时要按同样角度转
    crop_box: tuple[int, int, int, int] | None = None  # 表格区域，供"看原图"裁切

    @property
    def scores(self) -> list[int | None]:
        return [r.score for r in self.readings]


def crop_rows_for_llm(gray: np.ndarray, grid: Grid,
                      layout: list[tuple[int, int]] | None = None) -> list[np.ndarray]:
    """按行裁出给模型看的图：每行只含 10 个分值格，一行一张图。

    实测整块发会翻车（模型在窄列网格上会输出 10/9/8/7/6 这种伪规律），
    一行一张图 + 格子上方印分值标签之后，成功率接近 100%。
    """
    columns = layout or default_layout(grid)
    rows: list[np.ndarray] = []
    # 注意用 score_col_band：columns 里存的是"分值列序号"，不是原始竖线下标
    x0 = grid.score_col_band(columns[0][0])[0] - 6
    x1 = grid.score_col_band(columns[-1][0])[1] + 6
    for r in grid.data_rows:
        y0, y1 = grid.h[r] - 4, grid.h[r + 1] + 4
        sub = gray[y0:y1, x0:x1]
        rows.append(np.repeat(sub[:, :, None], 3, axis=2))
    return rows


def _apply_llm(r: PageResult, scores: list[int | None], marks: list[int]) -> None:
    r.llm_scores = scores
    r.llm_marks = marks
    if all(s is None for s in scores):
        r.flags.append(Flag("LLM 复核失败，这页按本地识别结果处理"))
        return

    diffs = 0
    for i, (a, b) in enumerate(zip(r.scores, scores)):
        if a == b:
            continue
        name = indicator_name(i)
        if a is None or b is None:
            r.flags.append(Flag(f"{name}：两个通道结论不一致，请确认",
                                f"本地 {a if a is not None else '未识别'} / "
                                f"LLM {b if b is not None else '未识别'}"))
            diffs += 1
            continue
        ratios = r.readings[i].ratios
        best, second, _best_i, second_i = top_two(ratios)
        ratio = second / best if best > 0 else 0.0
        second_score = 10 - second_i
        if ratio >= AUTO_ADOPT_RATIO and marks[i] >= 2 and b == second_score:
            r.readings[i].score = b
            r.adopted.append(f"{name} {a}→{b}")
            r.flags.append(Flag(
                f"{name}：两个格子都有标记，已按「打勾优先、画叉作废」取 {b} 分",
                f"本地判 {a}、LLM 判 {b}；两格墨量 {best:.2%} / {second:.2%}"))
            continue
        diffs += 1
        r.flags.append(Flag(
            f"{name}：本地 {a} 分、LLM {b} 分，暂按本地结果",
            f"两格墨量 {best:.2%} / {second:.2%}，更像勾的尾巴扫过，未自动采用 LLM 判断"
        ))

    r.total = sum_scores(r.readings)
    if all(s is not None for s in scores):
        r.llm_total = int(sum(s for s in scores if s is not None))

    # 采用过 LLM 判断的页，本质仍是"两通道不一致"，只是已经按规则裁决过了
    r.agree = not (diffs or r.adopted)


def process_pdf(pdf_path: str, dpi: int = 300, progress: Progress | None = None,
                llm_client=None, workers: int = 4,
                llm_progress: Callable[[int, int], None] | None = None,
                llm_scope: str = "all") -> ProcessOutcome:
    """处理整个 PDF。llm_client 不为 None 时，额外跑一遍 LLM 通道做交叉校验。

    空白页（扫描仪夹进来的白纸）会被跳过，不计入问卷份数和平均分，
    但页码会记录在 outcome.blank_pages 里，供界面和 Excel 提示。
    """
    results: list[PageResult] = []
    blank_pages: list[int] = []
    jobs: list[tuple[int, int, np.ndarray]] = []
    header_cache: dict[tuple[str, str, str], Header] = {}
    total_pages = count_pages(pdf_path)
    score_layout: list[tuple[int, int]] | None = None  # 一份文档只读一次表头分值

    for page_no, gray in render_pages(pdf_path, dpi=dpi):
        mask = ink_mask(gray)

        # 空白页：整页几乎没有墨迹 → 直接跳过
        if mask.mean() < BLANK_PAGE_MAX_INK:
            blank_pages.append(page_no)
            if progress:
                progress(page_no, total_pages)
            continue

        # 先按原图找网格；找不到再判断是不是扫描倾斜，纠偏后重试
        grid = find_grid(mask)
        angle = 0.0
        if not grid.ok:
            angle = estimate_skew(gray)
            if abs(angle) >= 0.1:
                gray = deskew(gray, angle)
                mask = ink_mask(gray)
                grid = find_grid(mask)
            else:
                angle = 0.0

        if not grid.ok:
            results.append(
                PageResult(page=page_no, header=Header("", "", ""), readings=[],
                           total=None, grid_ok=False,
                           flags=[Flag("这一页的表格没找到，请人工核对", grid.note)],
                           skew=angle,
                           grid_h=list(grid.h), grid_v=list(grid.v))
            )
            if progress:
                progress(page_no, total_pages)
            continue

        if score_layout is None:
            score_layout = read_score_layout(gray, grid) or default_layout(grid)
        readings = read_indicator_rows(mask, grid, score_layout)
        flags = [
            Flag(f"{indicator_name(r.index)}：{r.note}", r.detail)
            for r in readings if r.note
        ]

        header = parse_header(header_crop(gray, grid))
        cached = header_cache.get(header.key())
        if cached is not None:
            header = cached
        else:
            header_cache[header.key()] = header
            if not header.project:
                flags.append(Flag("表头的「项目名称」没认出来", "这一列的 OCR 结果为空"))
            if not header.assistant:
                flags.append(Flag("表头的「组长助手」没认出来", "这一列的 OCR 结果为空"))

        results.append(
            PageResult(page=page_no, header=header, readings=readings,
                       total=sum_scores(readings), grid_ok=True, flags=flags,
                       grid_h=list(grid.h), grid_v=list(grid.v), skew=angle,
                       crop_box=(grid.v[0], grid.h[max(0, grid.header_row - 1)],
                                 grid.v[-1], grid.h[-1]))
        )
        # llm_scope = "flagged" 时，只把这一批里有问题的页送去复核，省时省钱
        if llm_client is not None and (llm_scope == "all" or flags or sum_scores(readings) is None):
            for row_idx, rgb in enumerate(crop_rows_for_llm(gray, grid, score_layout)):
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
            n_rows = len(results[pi].readings) or 1
            slot = by_page.setdefault(pi, {"scores": [None] * n_rows,
                                           "marks": [0] * n_rows})
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
                r.flags = [f for f in r.flags if "LLM 复核失败" not in f.short]
        elif failures:
            llm_client.partial_warning = (
                f"{failures}/{total_calls} 次 LLM 调用失败（多为超时/限流），"
                f"这些行按代码通道结果处理并标黄")

    return ProcessOutcome(results=results, blank_pages=blank_pages,
                          total_pages=total_pages)


# 引号类字符在扫描件里经常被 OCR 成半角/全角/直角引号等不同形态，
# 但它们不该把同一个项目拆成两组 → 分组时忽略这些字符，显示仍用首次出现的写法。
_QUOTE_CHARS = '"\'""\'\'「」『』《》〈〉'


def _branch_key(name: str) -> str:
    return "".join(ch for ch in name if ch not in _QUOTE_CHARS)


def _same_project(a: str, b: str) -> bool:
    """判断两个项目名是不是同一个（容忍 OCR 漏字/多字）。

    OCR 偶尔会把「菏泽」读成「泽」，一个项目就被拆成两组。
    这里只在"长度差 ≤2 且相似度 ≥93%"时才认为是同一个——
    像「日照分行」和「济宁分行」这种正规差异相似度只有 80% 出头，不会被误合并。
    """
    if a == b:
        return True
    if min(len(a), len(b)) < 10 or abs(len(a) - len(b)) > 2:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.93


def group_by_branch(results: list[PageResult]) -> list[tuple[str, list[PageResult]]]:
    """按项目/分行分组，顺序 = 在 PDF 中首次出现的顺序。

    分组键做了两层容错：忽略引号形态差异、容忍 OCR 漏字（见 _same_project）。
    显示名优先用"更完整"的那种写法。
    """
    order: list[str] = []
    buckets: dict[str, list[PageResult]] = {}
    display: dict[str, str] = {}

    for r in results:
        name = r.header.project or "未识别分行"
        key = _branch_key(name)
        matched = None
        for existing in order:
            if _same_project(existing, key):
                matched = existing
                break
        if matched is None:
            buckets[key] = [r]
            order.append(key)
            display[key] = name
            continue
        buckets[matched].append(r)
        if len(key) > len(matched):  # 保留更完整的写法
            display[matched] = name
    return [(display[key], buckets[key]) for key in order]


def average(results: list[PageResult]) -> float | None:
    vals = [r.total for r in results if r.total is not None]
    return sum(vals) / len(vals) if vals else None
