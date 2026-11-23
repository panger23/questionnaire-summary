"""读表头：项目名称、审计组长、组长助手（→代职人）。

表头是印刷体，用 rapidocr（离线，模型随包）识别，只裁第一行那一窄条。

关键难点：不同表格的列宽、合并单元格都不一样，OCR 还会把相邻两格粘成一块
（例如「2025年山东省分行征信管」和「审计组长」被读成一段）。
所以这里不再依赖 OCR 文本框的相对位置，而是：
  1. 先在表头这一条里找出竖线，把表头切成若干「单元格」；
  2. 把 OCR 出的每段文字按 x 坐标归到对应单元格；
  3. 再按「标签格 → 右边那格是值」的方式取字段。
切不出单元格时，退回按标签相邻关系解析的旧办法。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .grid import _merge_runs

_OCR = None

PROJECT_RE = re.compile(r"\d{4}\s*年.*?(分行|审计|项目)")
SCORE_RE = re.compile(r"(\d+)\s*分")
CLEAN_RE = re.compile(r"[\s:：·．.]+")


def _ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
    return _OCR


@dataclass(frozen=True)
class Header:
    project: str
    lead: str  # 审计组长
    assistant: str  # 组长助手

    def key(self) -> tuple[str, str, str]:
        return (self.project, self.lead, self.assistant)


def header_crop(gray: np.ndarray, grid) -> np.ndarray:
    """表格第一行（项目名称那一行）的裁图。"""
    y0, y1 = grid.row_band(0)
    x0, x1 = grid.v[0], grid.v[-1]
    pad = 4
    h, w = gray.shape
    return gray[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]


def _clean(text: str) -> str:
    return CLEAN_RE.sub("", text or "")


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def _center(box) -> tuple[float, float]:
    pts = np.array(box, dtype=float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _merge_stacked(items: list[dict], x_overlap_ratio: float = 0.4, v_gap: int = 30) -> list[dict]:
    """把被 OCR 拆成上下两块的竖排标签合回一个，例如「审计组」+「长」→「审计组长」。"""
    merged: list[dict] = []
    for it in sorted(items, key=lambda d: d["y0"]):
        target = None
        for m in merged:
            overlap = min(m["x1"], it["x1"]) - max(m["x0"], it["x0"])
            width = min(m["x1"] - m["x0"], it["x1"] - it["x0"])
            if width <= 0:
                continue
            if overlap > x_overlap_ratio * width and -5 < (it["y0"] - m["y1"]) < v_gap:
                target = m
                break
        if target is None:
            merged.append(dict(it))
            continue
        target["text"] += it["text"]
        target["raw"] = target["text"]
        target["x0"] = min(target["x0"], it["x0"])
        target["x1"] = max(target["x1"], it["x1"])
        target["y0"] = min(target["y0"], it["y0"])
        target["y1"] = max(target["y1"], it["y1"])
        target["cx"] = (target["x0"] + target["x1"]) / 2
        target["cy"] = (target["y0"] + target["y1"]) / 2
    return merged


def _cell_bounds(crop: np.ndarray) -> list[tuple[int, int]]:
    """在表头这条裁图里找竖线，切成若干单元格的 x 区间。"""
    mask = crop < 128
    height, width = mask.shape
    if height < 20:
        return []
    columns = mask.sum(axis=0)
    # 表头竖线贯穿这一条的高度，用 0.55 倍高度即可和文字区分
    lines = _merge_runs(np.where(columns > 0.55 * height)[0])
    if len(lines) < 4:  # 没有足够的内部分隔线，切了也没意义
        return []
    edges = [0, *lines, width]
    bounds: list[tuple[int, int]] = []
    for left, right in zip(edges, edges[1:]):
        if right - left >= 24:
            bounds.append((left, right))
    return bounds if len(bounds) >= 3 else []


def _ocr_cell_text(cell: np.ndarray) -> str:
    """对单个单元格做 OCR，按阅读顺序拼成一串文字（不含空白）。"""
    result, _ = _ocr()(_to_bgr(cell))
    items = []
    for box, text, score in result or []:
        cleaned = _clean(text)
        if not cleaned:
            continue
        cx, cy = _center(box)
        items.append((cy, cx, cleaned))
    items.sort(key=lambda t: (round(t[0] / 12), t[1]))
    return "".join(t[2] for t in items)


def _assign_cells(items: list[dict], bounds: list[tuple[int, int]]) -> list[str]:
    """把整行 OCR 的文本按 x 坐标归到单元格（整行 OCR 兜底时用）。"""
    buckets: list[list[dict]] = [[] for _ in bounds]
    for it in items:
        for index, (left, right) in enumerate(bounds):
            if left - 5 <= it["cx"] < right + 5:
                buckets[index].append(it)
                break
    cells: list[str] = []
    for bucket in buckets:
        bucket.sort(key=lambda d: (round(d["cy"] / 12), d["cx"]))
        cells.append("".join(d["text"] for d in bucket))
    return cells


def _strip_label(text: str, label: str) -> str:
    return text.replace(label, "").strip()


NAME_RE = re.compile(r"^[\u4e00-\u9fa5·]{2,6}$")


def _looks_project(text: str) -> bool:
    return bool(re.search(r"\d{4}\s*年", text)) and ("审计" in text or "分行" in text)


def _looks_name(text: str) -> bool:
    return bool(NAME_RE.match(text or ""))


def _ocr_selected_cells(crop: np.ndarray, bounds: list[tuple[int, int]],
                        indices: list[int], gap: int = 24) -> list[str]:
    """把选中的几个单元格上下拼成一张图，只调用一次 OCR（比逐格调快 3 倍）。"""
    slices = [crop[:, bounds[i][0]:bounds[i][1]] for i in indices]
    max_width = max(sl.shape[1] for sl in slices)
    total_height = sum(sl.shape[0] for sl in slices) + gap * (len(slices) - 1)
    canvas = np.full((total_height, max_width), 255, dtype=np.uint8)
    bands: list[tuple[int, int]] = []
    offset = 0
    for sl in slices:
        canvas[offset:offset + sl.shape[0], 0:sl.shape[1]] = sl
        bands.append((offset, offset + sl.shape[0]))
        offset += sl.shape[0] + gap

    result, _ = _ocr()(_to_bgr(canvas))
    found: list[list[tuple[float, float, str]]] = [[] for _ in indices]
    for box, text, score in result or []:
        cleaned = _clean(text)
        if not cleaned:
            continue
        cx, cy = _center(box)
        for k, (top, bottom) in enumerate(bands):
            if top - 6 <= cy < bottom + 6:
                found[k].append((cy, cx, cleaned))
                break
    # 阅读顺序：先按行（上下），再按列（左右）——项目名经常折成两行
    return ["".join(t for _, _, t in sorted(bucket, key=lambda b: (round(b[0] / 14), b[1])))
            for bucket in found]


def _parse_from_cells(cells: list[str]) -> Header:
    project = lead = assistant = ""
    for index, text in enumerate(cells):
        nxt = cells[index + 1] if index + 1 < len(cells) else ""
        if "项目名称" in text:
            project = _strip_label(text, "项目名称") or nxt
        if "审计组长" in text:
            lead = _strip_label(text, "审计组长") or nxt
        if "组长助手" in text:
            assistant = _strip_label(text, "组长助手") or nxt
    return Header(project=project.strip(), lead=lead.strip(), assistant=assistant.strip())


def _parse_by_labels(items: list[dict]) -> Header:
    """兜底：不切单元格，靠标签文字的相对位置取值。"""
    if not items:
        return Header(project="", lead="", assistant="")
    line = sorted(items, key=lambda it: (round(it["cy"] / 24), it["cx"]))

    project = ""
    for it in line:
        if PROJECT_RE.search(it["text"]):
            project = it["text"]
            break
    if not project:
        for index, it in enumerate(line):
            if "项目名称" in it["text"]:
                if index + 1 < len(line):
                    project = line[index + 1]["text"]
                break

    def value_after(keyword: str) -> str:
        for index, it in enumerate(line):
            if keyword in it["text"]:
                rest = _strip_label(it["text"], keyword)
                if rest:
                    return rest
                for nxt in line[index + 1:]:
                    if nxt["x0"] >= it["x1"] - 30:
                        return nxt["text"]
        return ""

    return Header(project=project, lead=value_after("审计组长"),
                  assistant=value_after("组长助手"))


def read_score_layout(gray: np.ndarray, grid) -> list[tuple[int, int]] | None:
    """读表头那一行，识别出每个分值格印的是几分。

    返回 [(分值列序号, 分值), ...]，按从左到右排序；识别不出来返回 None。
    有了它就不用假设"一定是 10 档、从左到右 10..1"——
    换成 5 档、或者中间夹了「小计」列，都能自动对上。
    """
    if not grid.score_v or len(grid.score_v) < 2:
        return None
    y0, y1 = grid.row_band(grid.header_row)
    x_start = grid.v[grid.score_v[0]]
    x_end = grid.v[grid.score_v[-1]]
    strip = gray[y0:y1, x_start:x_end]
    if strip.size == 0:
        return None

    result, _ = _ocr()(_to_bgr(strip))
    found: dict[int, int] = {}
    for box, text, score in result or []:
        match = SCORE_RE.search(_clean(text) or "")
        if not match:
            continue
        value = int(match.group(1))
        if not 1 <= value <= 100:
            continue
        cx = _center(box)[0] + x_start
        for index in range(len(grid.score_v) - 1):
            left, right = grid.score_col_band(index)
            if left <= cx < right:
                found[index] = value
                break
    if len(found) < 3:
        return None
    return sorted(found.items())


def parse_header(crop: np.ndarray) -> Header:
    """对表头裁图做 OCR 并解析三个字段。"""
    result, _ = _ocr()(_to_bgr(crop))
    items = []
    for box, text, score in result or []:
        cleaned = _clean(text)
        if cleaned:
            cx, cy = _center(box)
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            items.append({"text": cleaned, "raw": text, "cx": cx, "cy": cy,
                          "x0": min(xs), "x1": max(xs),
                          "y0": min(ys), "y1": max(ys)})
    if not items:
        return Header(project="", lead="", assistant="")
    items = _merge_stacked(items)

    # 首选：按单元格切分。这张表家族固定是 7 格：
    # [项目名称][项目值][审计组长][组长值][组长助手][助手值][备注]
    # 7 格时只 OCR 第 1/3/5 格（省时间），并用形如"20xx年…审计""姓名"来校验；
    # 校验不过或格数不是 7，再退回逐格全 OCR、最后退回整行解析。
    bounds = _cell_bounds(crop)
    if len(bounds) == 7:
        project, lead, assistant = _ocr_selected_cells(crop, bounds, [1, 3, 5])
        if _looks_project(project) and _looks_name(assistant):
            return Header(project=project, lead=lead, assistant=assistant)

    if bounds:
        cell_texts = [_ocr_cell_text(crop[:, left:right]) for left, right in bounds]
        by_cells = _parse_from_cells(cell_texts)
        if by_cells.project and by_cells.assistant:
            return by_cells

    # 兜底：整行 OCR + 标签相邻关系
    return _parse_by_labels(items)
