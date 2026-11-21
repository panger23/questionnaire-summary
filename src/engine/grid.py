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

from dataclasses import dataclass

import numpy as np

EXPECTED_H_LINES = 11
EXPECTED_V_LINES = 15

# 横线贯穿表宽（约占页宽 87%），竖线只贯穿表高（约占页高 47%）
_H_FRACTIONS = (0.55, 0.45, 0.35, 0.28, 0.22, 0.16, 0.10)
_V_FRACTIONS = (0.30, 0.25, 0.20, 0.16, 0.13, 0.10, 0.07)


@dataclass
class Grid:
    h: list[int]  # 横线的 y 坐标，自上而下
    v: list[int]  # 竖线的 x 坐标，自左向右
    ok: bool  # 是否检出了期望条数
    note: str = ""

    def row_band(self, r: int) -> tuple[int, int]:
        return self.h[r], self.h[r + 1]

    def col_band(self, c: int) -> tuple[int, int]:
        return self.v[c], self.v[c + 1]


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


def _detect(mask: np.ndarray, h_frac: float, v_frac: float) -> tuple[list[int], list[int]]:
    height, width = mask.shape
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    h = _merge_runs(np.where(row_counts > h_frac * width)[0])
    v = _merge_runs(np.where(col_counts > v_frac * height)[0])
    return h, v


def find_grid(mask: np.ndarray) -> Grid:
    """检出表格网格；失败时 ok=False，调用方应把该页转人工。"""
    last: tuple[list[int], list[int]] = ([], [])
    for h_frac in _H_FRACTIONS:
        for v_frac in _V_FRACTIONS:
            h, v = _detect(mask, h_frac, v_frac)
            last = (h, v)
            if len(h) == EXPECTED_H_LINES and len(v) == EXPECTED_V_LINES:
                return Grid(h=h, v=v, ok=True)
    h, v = last
    return Grid(
        h=h,
        v=v,
        ok=False,
        note=f"网格线数量异常：横线 {len(h)}/11，竖线 {len(v)}/15",
    )
