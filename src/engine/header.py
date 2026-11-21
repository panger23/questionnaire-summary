"""读表头：项目名称（→分行）、审计组长、组长助手（→代职人）。

表头是印刷体，用 rapidocr（离线，模型随包）识别，只裁第一行那一窄条，
比整页 OCR 快一个数量级。同分行的十几页表头完全一样，所以按"表头指纹"
聚类，每个分行只真正 OCR 一次。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .grid import Grid

_OCR = None

PROJECT_RE = re.compile(r"\d{4}\s*年.*?(分行|审计|项目)")
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


def header_crop(gray: np.ndarray, grid: Grid) -> np.ndarray:
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
    """把被 OCR 拆成上下两块的竖排标签合回一个，例如「审计组」+「长」→「审计组长」。

    表格里「项目名称」「审计组长」这类标签是竖着写成两行的，
    rapidocr 会把每行当成一个独立文本块。
    """
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
    items.sort(key=lambda it: (it["cy"], it["cx"]))
    row_y = np.median([it["cy"] for it in items]) if items else 0.0
    line = [it for it in items if abs(it["cy"] - row_y) <= 40] or items
    line.sort(key=lambda it: it["cx"])

    project = ""
    for it in line:
        if PROJECT_RE.search(it["text"]):
            project = it["text"]
            break
    if not project:
        label = next((i for i, it in enumerate(line) if "项目名称" in it["text"]), None)
        if label is not None and label + 1 < len(line):
            project = line[label + 1]["text"]

    def value_after(keyword: str) -> str:
        for i, it in enumerate(line):
            if keyword in it["text"]:
                # 值在右边第一个文本块；同一块里冒号后也算
                if ":" in it["raw"] or "：" in it["raw"]:
                    tail = re.split(r"[:：]", it["raw"])[-1]
                    tail = _clean(tail)
                    if tail:
                        return tail
                for nxt in line[i + 1:]:
                    if nxt["x0"] >= it["x1"] - 30:
                        return nxt["text"]
        return ""

    return Header(project=project, lead=value_after("审计组长"),
                  assistant=value_after("组长助手"))
