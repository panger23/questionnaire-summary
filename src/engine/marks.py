"""判断每个评价指标勾在哪一列 —— 靠像素墨量，不靠 OCR。

为什么不用 OCR 找勾：勾不是字符，OCR 会把无意义的笔迹当噪声丢掉。
实测整页 OCR 在 270 个评分格里只吐出 22 个识别框，绝大多数勾根本没被识别；
而墨迹法 270/270 全对，速度快两个数量级。

判定规则：
  1. 每行的 10 个分值格各算一个"墨量占比"（墨迹像素 / 格内像素）；
  2. 最高的一格就是勾所在，分值 = 10 - 列序号；
  3. 若最高格墨量 < 1%，说明这行没勾 → 标黄；
  4. 若第二高格 ≥ 最高格的 25%，说明墨迹跨了两格（画叉又打勾 / 勾尾越界 / 压在分界线上）
     → 仍然给出取值，但标黄交人确认。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Grid

# 评分列在 14 列里的下标区间；列 c 对应分值 10-(c-2)
SCORE_COL_START = 2
SCORE_COL_END = 12  # 不含
# 五个评价指标所在行（0 起的表格行号），第 9 行是"总体评分"，不读
INDICATOR_ROWS = (4, 5, 6, 7, 8)

CELL_INSET = 8  # 格内缩多少像素，避开印刷网格线
HAS_MARK_MIN_RATIO = 0.010  # 最高格墨量低于 1% 视为没勾
AMBIGUOUS_SECOND_RATIO = 0.25  # 次高格 ≥ 最高格 25% 视为跨格

SCORE_LABELS = ("10分", "9分", "8分", "7分", "6分", "5分", "4分", "3分", "2分", "1分")
INDICATOR_NAMES = ("任务分工", "沟通协调", "团结协作", "专业技术", "关爱扶持")


def _ink_ratio(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    ys, ye = y0 + CELL_INSET, y1 - CELL_INSET
    xs, xe = x0 + CELL_INSET, x1 - CELL_INSET
    if ye <= ys or xe <= xs:
        return 0.0
    return float(mask[ys:ye, xs:xe].mean())


def _ratios_for_row(mask: np.ndarray, grid: Grid, row: int) -> list[float]:
    y0, y1 = grid.row_band(row)
    ratios = []
    for c in range(SCORE_COL_START, SCORE_COL_END):
        x0, x1 = grid.col_band(c)
        ratios.append(_ink_ratio(mask, y0, y1, x0, x1))
    return ratios


@dataclass
class RowReading:
    row: int
    score: int | None  # None = 这行没找到勾
    ratios: list[float] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def read_indicator_rows(mask: np.ndarray, grid: Grid) -> list[RowReading]:
    """读出 5 个评价指标各得几分。"""
    out: list[RowReading] = []
    for row in INDICATOR_ROWS:
        ratios = _ratios_for_row(mask, grid, row)
        best_i = int(np.argmax(ratios))
        best = ratios[best_i]
        rest = [r for i, r in enumerate(ratios) if i != best_i]
        second = max(rest) if rest else 0.0
        flags: list[str] = []

        if best < HAS_MARK_MIN_RATIO:
            out.append(RowReading(row=row, score=None, ratios=ratios,
                                  flags=["该行没有找到勾"]))
            continue

        if second >= AMBIGUOUS_SECOND_RATIO * best:
            flags.append(
                f"墨迹跨格：{SCORE_LABELS[best_i]} 最重"
                f"（{best:.2%}），{SCORE_LABELS[ratios.index(second)]} 次之（{second:.2%}）"
            )

        out.append(RowReading(row=row, score=10 - best_i, ratios=ratios, flags=flags))
    return out


def sum_scores(readings: list[RowReading]) -> int | None:
    """五项求和；任何一项缺失则返回 None。"""
    if any(r.score is None for r in readings):
        return None
    return int(sum(r.score for r in readings if r.score is not None))


def top_two(ratios: list[float]) -> tuple[float, float, int, int]:
    """返回 (最高墨量, 次高墨量, 最高格下标, 次高格下标)。"""
    order = sorted(range(len(ratios)), key=lambda i: ratios[i], reverse=True)
    best_i, second_i = order[0], order[1]
    return ratios[best_i], ratios[second_i], best_i, second_i
