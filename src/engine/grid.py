"""定位问卷表格的横竖网格线。

问卷版式固定为 11 条横线 x 15 条竖线，围出 10 行 x 14 列的表格：
    列 0  序号
    列 1  评价指标
    列 2..11  分值 10,9,8,7,6,5,4,3,2,1
    列 12 小计（不读）
    列 13 备注（不读）
    行 0  项目名称 / 审计组长 / 组长助手 / 备注
    行 1  评价等级
    行 2  序号 / 评价指标 / 等级名
    行 3  分值表头
    行 4..8  五个评价指标
    行 9  总体评分（不读，手写内容一律忽略）

检测方式：统计每行/每列的墨迹像素数。网格线是贯穿整张表的直线，
黑像素数远高于文字；再按连通段合并成线。若阈值下数不出期望条数，
就逐级降阈值重试 —— 实测本批 54 页有 3 页（42/51/52）格线有断裂。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 只作为"理想值"用于挑最接近的检测结果；真正的行列数由下面 infer 出来
EXPECTED_H_LINES = 11
EXPECTED_V_LINES = 15

# 横线贯穿表宽（约占页宽 87%），竖线只贯穿表高（约占页高 47%）
_H_FRACTIONS = (0.55, 0.45, 0.35, 0.28, 0.22, 0.16, 0.10)
_V_FRACTIONS = (0.30, 0.25, 0.20, 0.16, 0.13, 0.10, 0.07)


@dataclass
class Grid:
    h: list[int]  # 横线的 y 坐标，自上而下
    v: list[int]  # 竖线的 x 坐标，自左向右
    ok: bool
    note: str = ""
    # 以下三项是"从图像里推断出来的结构"，不写死，换版式也能用：
    score_v: list[int] = field(default=None)   # 分值列对应的竖线下标（长度 = 分值列数+1）
    header_row: int = 3                        # 分值表头（10分/9分…）所在行
    data_rows: list[int] = field(default=None)  # 打分的那几行

    def __post_init__(self):
        if self.score_v is None:
            self.score_v = infer_score_columns(self.v)
        if self.data_rows is None:
            self.data_rows = list(range(self.header_row + 1, max(0, len(self.h) - 2)))

    @property
    def score_column_count(self) -> int:
        return max(0, len(self.score_v) - 1)

    def row_band(self, r: int) -> tuple[int, int]:
        return self.h[r], self.h[r + 1]

    def col_band(self, c: int) -> tuple[int, int]:
        return self.v[c], self.v[c + 1]

    def score_col_band(self, index: int) -> tuple[int, int]:
        """第 index 个分值列（0 起，从左到右）的 x 区间。"""
        return self.v[self.score_v[index]], self.v[self.score_v[index + 1]]


def _merge_runs(indices: np.ndarray, gap: int = 4) -> list[int]:
    """把相邻的线像素下标合并成线段中心。"""
    if indices.size == 0:
        return []
    centers: list[int] = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        i = int(raw)
        if i - prev <= gap:
            prev = i
            continue
        centers.append((start + prev) // 2)
        start = prev = i
    centers.append((start + prev) // 2)
    return centers


def infer_score_columns(v: list[int]) -> list[int]:
    """从竖线里找出"分值列"——它们是等宽、连续的一段。

    做法：算相邻竖线的间距，找出最长的一段"间距基本一致"的区间（±25% 以内）。
    这段区间围出来的列就是打分格。这样不管是 10 档、5 档还是别的列数都能识别。
    """
    if len(v) < 4:
        return list(range(len(v)))
    gaps = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    best_len, best_start, best_end = 1, 0, 0
    i = 0
    while i < len(gaps):
        median = gaps[i]
        j = i
        while j + 1 < len(gaps) and abs(gaps[j + 1] - median) <= 0.25 * median:
            j += 1
        if (j - i + 1) > best_len:
            best_len, best_start, best_end = j - i + 1, i, j
        i = j + 1
    return list(range(best_start, best_end + 2))


def _detect(mask: np.ndarray, h_frac: float, v_frac: float) -> tuple[list[int], list[int]]:
    height, width = mask.shape
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    h = _merge_runs(np.where(row_counts > h_frac * width)[0])
    v = _merge_runs(np.where(col_counts > v_frac * height)[0])
    return h, v


def find_grid(mask: np.ndarray) -> Grid:
    """检出表格网格；失败时 ok=False，调用方应把该页转人工。

    阈值是逐个试的：一旦有组合正好 11 横 x 15 竖就用它；都不正好时，
    退而取"数量最接近"的那一组（而不是盲取最后一组，那样会挑到一堆文字线）。
    """
    best: tuple[list[int], list[int]] = ([], [])
    best_gap = 10 ** 6
    for h_frac in _H_FRACTIONS:
        for v_frac in _V_FRACTIONS:
            h, v = _detect(mask, h_frac, v_frac)
            if len(h) == EXPECTED_H_LINES and len(v) == EXPECTED_V_LINES:
                return _with_header_row(mask, Grid(h=h, v=v, ok=True))
            gap = abs(len(h) - EXPECTED_H_LINES) + abs(len(v) - EXPECTED_V_LINES)
            if gap < best_gap:
                best_gap, best = gap, (h, v)
    h, v = best
    return Grid(
        h=h,
        v=v,
        ok=False,
        note=f"网格线数量异常：横线 {len(h)}/11，竖线 {len(v)}/15",
    )


def _with_header_row(mask: np.ndarray, grid: Grid) -> Grid:
    """找出"分值表头"那一行并据此确定数据行。

    分值表头行（10分/9分/…）的特征：每个分值格里都印着字，墨迹几乎满格；
    而数据行每行只有一两个格子有笔迹。用这个差异定位，不依赖固定行号。
    """
    if not grid.score_v or len(grid.h) < 3:
        return grid
    inset = 8
    best_row, best_frac = grid.header_row, 0.0
    for row in range(len(grid.h) - 1):
        y0, y1 = grid.h[row], grid.h[row + 1]
        if y1 - y0 < 20:
            continue
        filled = 0
        for index in range(len(grid.score_v) - 1):
            x0, x1 = grid.score_col_band(index)
            cell = mask[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
            if cell.size and cell.mean() >= 0.01:
                filled += 1
        frac = filled / max(1, len(grid.score_v) - 1)
        if frac > best_frac:
            best_frac, best_row = frac, row
    if best_frac >= 0.6:
        return Grid(h=grid.h, v=grid.v, ok=True, note=grid.note,
                    score_v=grid.score_v, header_row=best_row)
    return grid
