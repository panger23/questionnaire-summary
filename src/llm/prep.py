"""给模型看的行图预处理：在 10 个格子正上方印上分值标签（10、9、…、1）。

为什么要加这排数字：实测模型在"数格子"这件事上会出错（勾在最左格时会
判成第二格，输出 10/9/8/7/6 这种伪规律）。印上数字后，它从"数格子"
变成"认数字"，同一批数据里的误判全部消失。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

BAND_HEIGHT = 46
FONT_SIZE = 40


@lru_cache(maxsize=4)
def _font(size: int = FONT_SIZE):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def annotate_row(rgb: np.ndarray, ncol: int = 10) -> np.ndarray:
    """在一行评分图的上方加一条白带，把 10、9、…、1 印在各个格子正上方。"""
    image = Image.fromarray(rgb)
    width, height = image.size
    canvas = Image.new("RGB", (width, height + BAND_HEIGHT), "white")
    canvas.paste(image, (0, BAND_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    cell_width = width / ncol
    font = _font(FONT_SIZE)
    for i in range(ncol):
        label = str(10 - i)
        center_x = cell_width * i + cell_width / 2
        text_width = draw.textlength(label, font=font)
        draw.text((center_x - text_width / 2, 2), label, fill="black", font=font)
    return np.array(canvas)
