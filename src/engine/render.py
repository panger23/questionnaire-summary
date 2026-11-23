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


def estimate_skew(gray: np.ndarray, max_angle: float = 2.0, step: float = 0.1,
                  target_width: int = 900) -> float:
    """估计扫描倾斜角（度）。正值 = 需要逆时针转回来。

    做法：把图缩小后，按不同角度旋转，看哪个角度下"最强的那些横线"最粗。
    表格线一旦被摊到好几行像素上，行投影就会变弱；转正了投影最强。
    实测同一份 PDF 里各页可能差 0.2°~0.5°，不纠偏的话横线会检测不全。
    """
    from PIL import Image

    height, width = gray.shape
    scale = target_width / float(width)
    small = Image.fromarray(gray).resize(
        (target_width, max(1, int(height * scale))), Image.BILINEAR)
    base = np.array(small)

    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        rotated = np.array(
            Image.fromarray(base).rotate(float(angle), resample=Image.BILINEAR,
                                         fillcolor=255))
        ink = rotated < INK_THRESHOLD
        if not ink.any():
            continue
        strongest = np.sort(ink.sum(axis=1))[-15:]  # 表格有 11 条横线，取最强的一批
        score = float(strongest.mean())
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return round(best_angle, 1)


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    """按给定角度把图转正（白底填充，不引入额外墨迹）。"""
    from PIL import Image

    if abs(angle) < 0.05:
        return gray
    rotated = Image.fromarray(gray).rotate(angle, resample=Image.BILINEAR, fillcolor=255)
    return np.array(rotated)


def count_pages(pdf_path: str) -> int:
    """PDF 总页数（进度条要用）。"""
    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()
