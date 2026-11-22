"""把扫描 PDF 的每一页渲染成灰度 numpy 图。

统一用 300dpi：实测该分辨率下网格线宽约 4px、单元格约 133x105px，
墨迹与印刷线信噪比最好。分辨率变了下面所有像素常数都要跟着变。
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pymupdf

DPI = 300
# 灰度 < INK_THRESHOLD 视为墨迹（印刷线 + 笔迹）
INK_THRESHOLD = 128


def render_pages(pdf_path: str, dpi: int = DPI) -> Iterator[tuple[int, np.ndarray]]:
    """逐页产出 (页码从 1 开始, 灰度数组 HxW uint8)。"""
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            yield i + 1, np.array(img, dtype=np.uint8)
    finally:
        doc.close()


def ink_mask(gray: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    """二值化：墨迹=True。"""
    return gray < threshold


def count_pages(pdf_path: str) -> int:
    """PDF 总页数（进度条要用）。"""
    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()
