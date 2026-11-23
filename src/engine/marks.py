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
# 指标名称（仅用于给人看的提示文字，识别本身不依赖它）
INDICATOR_NAMES = ("任务分工", "沟通协调", "团结协作", "专业技术", "关爱扶持")

CELL_INSET = 8  # 格内缩多少像素，避开印刷网格线
HAS_MARK_MIN_RATIO = 0.010  # 最高格墨量低于 1% 视为没勾
AMBIGUOUS_SECOND_RATIO = 0.25  # 次高格 ≥ 最高格 25% 视为跨格

# 分值列从左到右对应的分（与表头印刷的 10分…1分 一致）
SCORE_LABELS = tuple(f"{n}分" for n in range(10, 0, -1))

def _ink_ratio(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    ys, ye = y0 + CELL_INSET, y1 - CELL_INSET
    xs, xe = x0 + CELL_INSET, x1 - CELL_INSET
    if ye <= ys or xe <= xs:
        return 0.0
    return float(mask[ys:ye, xs:xe].mean())


def default_layout(grid: Grid) -> list[tuple[int, int]]:
    """拿不到表头分值时的兜底：从左到右按 N, N-1, … 排列。"""
    count = grid.score_column_count
    return [(index, count - index) for index in range(count)]


def _ratios_for_row(mask: np.ndarray, grid: Grid, row: int,
                    columns: list[tuple[int, int]]) -> list[float]:
    """这一行每个分值格的墨量占比（顺序与 columns 一致）。"""
    y0, y1 = grid.row_band(row)
    ratios = []
    for index, _score in columns:
        x0, x1 = grid.score_col_band(index)
        ratios.append(_ink_ratio(mask, y0, y1, x0, x1))
    return ratios


@dataclass
class RowReading:
    row: int
    score: int | None  # None = 这行没找到勾
    ratios: list[float] = field(default_factory=list)
    note: str = ""    # 一句话：这行有什么问题
    detail: str = ""  # 技术细节（墨量等），界面上悬停才显示
    index: int = 0    # 是第几个指标（0 起），用于取指标名称


def read_indicator_rows(mask: np.ndarray, grid: Grid,
                        layout: list[tuple[int, int]] | None = None) -> list[RowReading]:
    """读出 5 个评价指标各得几分 —— 就认"墨量最大的那一格"。

    如果一行里有两个格子都有明显笔迹（画叉又打勾、勾尾巴扫过、有人画了个圈），
    不去猜是什么意思，只标黄交给人看。分数由人确认后手工改。
    """
    columns = layout or default_layout(grid)
    out: list[RowReading] = []
    for position, row in enumerate(grid.data_rows):
        ratios = _ratios_for_row(mask, grid, row, columns)
        best_i = int(np.argmax(ratios))
        best = ratios[best_i]
        rest = [r for i, r in enumerate(ratios) if i != best_i]
        second = max(rest) if rest else 0.0

        if best < HAS_MARK_MIN_RATIO:
            out.append(RowReading(row=row, score=None, ratios=ratios,
                                  note="这一行没找到标记", index=position))
            continue

        note = detail = ""
        if second >= AMBIGUOUS_SECOND_RATIO * best:
            best_label = f"{columns[best_i][1]}分"
            second_label = f"{columns[ratios.index(second)][1]}分"
            note = f"两个格子都有笔迹，暂取 {best_label}，请确认"
            detail = f"{best_label} 墨量 {best:.2%}，{second_label} 墨量 {second:.2%}"

        out.append(RowReading(row=row, score=columns[best_i][1], ratios=ratios,
                              note=note, detail=detail, index=position))
    return out


def indicator_name(index: int) -> str:
    """指标名称（仅用于提示文字；换版式时名称对不上也不影响识别）。"""
    if 0 <= index < len(INDICATOR_NAMES):
        return INDICATOR_NAMES[index]
    return f"第{index + 1}项"


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
