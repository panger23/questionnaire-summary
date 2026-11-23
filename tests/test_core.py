"""核心逻辑的单元测试（不依赖真实问卷 PDF）。

跑法：.venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.grid import Grid, _merge_runs  # noqa: E402
from src.engine.marks import (  # noqa: E402
    AMBIGUOUS_SECOND_RATIO,
    CELL_INSET,
    INDICATOR_ROWS,
    SCORE_COL_END,
    SCORE_COL_START,
    read_indicator_rows,
    sum_scores,
    top_two,
)
from src.llm.client import parse_verdict  # noqa: E402
from src.llm.config import masked_key  # noqa: E402

# 一份合成网格：11 条横线、15 条竖线，尺寸和真实扫描件同一量级
H = [765, 945, 1020, 1096, 1285, 1390, 1495, 1600, 1705, 1810, 1914]
V = [202, 348, 1322, 1456, 1589, 1721, 1854, 1987, 2119, 2252,
     2385, 2518, 2651, 2784, 3278]


# 测试用表格是 10 档（10分…1分），显式给出布局，不依赖表头 OCR
LAYOUT = [(i, 10 - i) for i in range(10)]


def make_mask() -> np.ndarray:
    return np.zeros((2458, 3520), dtype=bool)


def paint_cell(mask: np.ndarray, row: int, score_col_index: int, thickness: int = 20) -> None:
    """在 (row, 第 score_col_index 个分值列) 里画一块墨。"""
    y0, y1 = H[row], H[row + 1]
    col = SCORE_COL_START + score_col_index
    x0, x1 = V[col], V[col + 1]
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    mask[cy - thickness:cy + thickness, cx - thickness:cx + thickness] = True


class GridTests(unittest.TestCase):
    def test_merge_runs(self):
        self.assertEqual(_merge_runs(np.array([5, 6, 7, 20, 21])), [6, 20])
        self.assertEqual(_merge_runs(np.array([], dtype=int)), [])


class MarkTests(unittest.TestCase):
    def test_single_mark_is_read(self):
        mask = make_mask()
        for row in INDICATOR_ROWS:
            paint_cell(mask, row, 0)  # 全部勾在 10 分
        grid = Grid(h=H, v=V, ok=True)
        readings = read_indicator_rows(mask, grid, LAYOUT)
        self.assertEqual([r.score for r in readings], [10] * 5)
        self.assertEqual(sum_scores(readings), 50)
        self.assertFalse(any(r.note for r in readings))

    def test_mark_in_middle_column(self):
        mask = make_mask()
        for row in INDICATOR_ROWS:
            paint_cell(mask, row, 2)  # 8 分
        grid = Grid(h=H, v=V, ok=True)
        self.assertEqual([r.score for r in read_indicator_rows(mask, grid, LAYOUT)], [8] * 5)

    def test_two_marks_is_flagged(self):
        mask = make_mask()
        for row in INDICATOR_ROWS:
            paint_cell(mask, row, 0)
        paint_cell(mask, INDICATOR_ROWS[0], 3, thickness=18)  # 再多画一个 7 分
        grid = Grid(h=H, v=V, ok=True)
        readings = read_indicator_rows(mask, grid, LAYOUT)
        self.assertTrue(readings[0].note, "两个格子都有墨时必须给出提示")
        best, second, _, _ = top_two(readings[0].ratios)
        self.assertGreater(second / best, AMBIGUOUS_SECOND_RATIO)

    def test_empty_row_gives_none(self):
        mask = make_mask()
        for row in INDICATOR_ROWS[1:]:
            paint_cell(mask, row, 0)
        grid = Grid(h=H, v=V, ok=True)
        readings = read_indicator_rows(mask, grid, LAYOUT)
        self.assertIsNone(readings[0].score)
        self.assertIsNone(sum_scores(readings))


class LlmParseTests(unittest.TestCase):
    def test_parse(self):
        verdict = parse_verdict('```json\n{"score": 7, "marks": 2}\n```')
        self.assertEqual(verdict.score, 7)
        self.assertEqual(verdict.marks, 2)

    def test_parse_without_marks(self):
        self.assertEqual(parse_verdict('{"score": 10}').marks, 1)

    def test_parse_null_score(self):
        verdict = parse_verdict('{"score": null, "marks": 0}')
        self.assertIsNone(verdict.score)
        self.assertEqual(verdict.marks, 0)

    def test_parse_garbage(self):
        self.assertIsNone(parse_verdict("模型不知道"))
        self.assertIsNone(parse_verdict('{"score": 11, "marks": 1}'))

    def test_masked_key(self):
        self.assertEqual(masked_key("sk-1234567890abcd"), "sk-****abcd")
        self.assertEqual(masked_key(""), "")


class WebSerializationTests(unittest.TestCase):
    """手工改分 / 采用 LLM 之后，标记列表里必须都是 Flag 对象，序列化不能崩。

    之前这里踩过坑：往 r.flags 里塞纯字符串，编辑分数后整个接口 500。
    """

    @staticmethod
    def _make_row():
        from src.engine.header import Header
        from src.engine.marks import RowReading
        from src.engine.pipeline import Flag, PageResult

        readings = [
            RowReading(row=row, score=10, ratios=[0.04] + [0.0] * 9)
            for row in (4, 5, 6, 7, 8)
        ]
        return PageResult(
            page=3,
            header=Header("2025年日照分行常规审计", "陈霄", "綦光磊"),
            readings=readings,
            total=50,
            grid_ok=True,
            flags=[Flag("短句说明", "技术细节")],
        )

    def test_row_serializes(self):
        from src.web.app import _row

        data = _row(self._make_row())
        self.assertEqual(data["flags"][0]["short"], "短句说明")
        self.assertEqual(data["flags"][0]["detail"], "技术细节")

    def test_add_flag_keeps_flag_objects(self):
        from src.web.app import _add_flag, _row

        r = self._make_row()
        _add_flag(r, "这一行手动改过分数")
        data = _row(r)  # 以前这步会 AttributeError
        self.assertEqual(data["flags"][-1]["short"], "这一行手动改过分数")
        _add_flag(r, "这一行手动改过分数")  # 重复添加不叠加
        self.assertEqual(len(r.flags), 2)


class LayoutConstantTests(unittest.TestCase):
    def test_geometry_expectations(self):
        self.assertEqual(len(H), 11)
        self.assertEqual(len(V), 15)
        self.assertEqual(len(INDICATOR_ROWS), 5)
        self.assertEqual(SCORE_COL_END - SCORE_COL_START, 10)
        self.assertLess(CELL_INSET, (H[1] - H[0]) / 4)


if __name__ == "__main__":
    unittest.main()
