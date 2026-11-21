"""调用视觉模型，让它独立回答"5 个指标各勾在哪一分"。

只发评分区那一小块（不含备注列、不含手写总分行），既省钱又不干扰模型。
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

PROMPT = """这是一张纸质调查问卷《审计项目组组员满意度调查问卷》的评分表局部截图。

表格结构：
- 左侧是「序号」和「评价指标」两列，评价指标从上到下共 5 行，依次是：任务分工、沟通协调、团结协作、专业技术、关爱扶持
- 右侧是 10 个分值列，从左到右依次是：10分、9分、8分、7分、6分、5分、4分、3分、2分、1分
- 填写人在每一行的某一个分值格内手写了勾（√），个别格子里写的是数字

任务：判断上述 5 行，每一行被标记的是哪一个分值。

规则：
- 一行里只有一个格子有笔迹时，就是它。
- 一行里有两个格子都有笔迹时（例如一个打了叉、另一个打了勾），选填写人最终想表达的那个——打勾通常代表最终选择，画叉通常代表作废。
- 一行里完全没有笔迹，返回 null。
- 只根据图片判断，不要猜测规律。

只输出 JSON，不要输出任何解释文字。格式如下（数组长度必须是 5，顺序为任务分工、沟通协调、团结协作、专业技术、关爱扶持）：
{"scores": [10, 9, 8, 7, 6]}"""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens


def _png_bytes(rgb: np.ndarray) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def parse_scores(text: str) -> list[int | None] | None:
    """从模型回复里抠出 5 个分数；格式不对就返回 None。"""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    raw = data.get("scores")
    if not isinstance(raw, list) or len(raw) != 5:
        return None
    out: list[int | None] = []
    for value in raw:
        if value is None:
            out.append(None)
        elif isinstance(value, (int, float)) and 1 <= int(value) <= 10:
            out.append(int(value))
        else:
            out.append(None)
    return out


class VisionClient:
    """OpenAI 兼容接口的视觉模型客户端（默认百炼 qwen3-vl-plus）。"""

    def __init__(self, api_base: str, api_key: str, model: str = "qwen3-vl-plus",
                 timeout: float = 90.0):
        self.model = model
        self.usage = Usage()
        self._client = OpenAI(base_url=api_base, api_key=api_key,
                              timeout=timeout, max_retries=2)

    def read_scores(self, rgb: np.ndarray) -> list[int | None] | None:
        b64 = base64.b64encode(_png_bytes(rgb)).decode("ascii")
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
            )
        except Exception:  # noqa: BLE001 - 网络/额度/模型问题都按"这一页失败"处理
            self.usage.failures += 1
            return None
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        self.usage.add(getattr(usage, "prompt_tokens", 0) or 0,
                       getattr(usage, "completion_tokens", 0) or 0)
        return parse_scores(text)

    def estimated_cost(self, input_price_per_m: float = 1.0,
                       output_price_per_m: float = 10.0) -> float:
        return (self.usage.prompt_tokens / 1_000_000 * input_price_per_m
                + self.usage.completion_tokens / 1_000_000 * output_price_per_m)
