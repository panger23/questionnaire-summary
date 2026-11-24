"""调用视觉模型，让它独立回答"这一行勾在第几格"。

关键经验（实测）：把整块 5 行 × 10 列一次性发给模型会翻车——它在窄列网格上
做空间计数不可靠，勾落在最左侧一列时会输出 10/9/8/7/6 这种"等差数列"。
改成**一行一张图、只判一个勾**之后，同一批数据的错误全部消失。

所以这里的做法是每行一次调用，图片只有 1330×113，既便宜又准。
"""
from __future__ import annotations

import base64
import json
import random
import re
import time
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from .prep import annotate_row

ROW_PROMPT = """这是一行调查问卷的评分格截图（来自《审计项目组组员满意度调查问卷》）。

图片上方标着一排数字，每个数字正下方是一个格子，从左到右依次是 10、9、8、7、6、5、4、3、2、1。

填写人只会在其中一个格子里填写。请找出被填写的那一格。
也有极小的概率写完一列，把这一列划了。重新在别的格子打勾或者打分。

判断规则（按顺序执行）：
1. 先把"有笔迹的格子"全部找出来，不要只看最明显的那个。
2. 只有一个格子有笔迹 → 就是它。
3. 笔迹跨了两个格子时：
   - 有完整打勾（√）的那一格才算数；画叉（×）、划掉、斜杠都不算。
   - 如果一个格子里是完整的勾，相邻格子里只有一条细长的线头/尾巴，取有完整勾的那一格。
   - 勾的转折点（左下角的尖）在哪一格，就算哪一格。
4. 如果某条线/圈从格子顶边一直画到底边（贯穿整格），那是标记或圈注，不是打分，不要选它，去找真正打勾的格子。
5. 整行都没有笔迹 → score 填 null。

只输出一行 JSON，不要任何解释：
{"score": 10, "marks": 1}"""


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


@dataclass
class RowVerdict:
    score: int | None
    marks: int


def parse_verdict(text: str) -> RowVerdict | None:
    """从模型回复里抠出 {"score": 7, "marks": 2}；格式不对返回 None。"""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    # 有的模型会自作主张把字段名写成中文（智谱 glm-4.5v 实测如此），一并兼容
    raw_score = data.get("score", data.get("分数"))
    raw_marks = data.get("marks", data.get("标记数", 1))
    if raw_score is None:
        return RowVerdict(score=None, marks=0)
    if isinstance(raw_score, float) and raw_score.is_integer():
        raw_score = int(raw_score)
    if not (isinstance(raw_score, int) and 1 <= raw_score <= 10):
        return None
    marks = raw_marks if isinstance(raw_marks, int) and 0 <= raw_marks <= 10 else 1
    return RowVerdict(score=raw_score, marks=marks)


def _png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


class VisionClient:
    """OpenAI 兼容接口的视觉模型客户端（默认百炼 qwen3-vl-plus）。"""

    MAX_ATTEMPTS = 3  # 429/超时最多重试 3 次（180s×5 会卡 15 分钟，实测过的坑）

    def __init__(self, api_base: str, api_key: str, model: str = "qwen3-vl-plus",
                 timeout: float = 60.0, max_attempts: int = MAX_ATTEMPTS):
        self.model = model
        self.max_attempts = max_attempts
        self.usage = Usage()
        self.last_error = ""      # 最后一次调用的报错原文
        self.channel_error = ""   # 整批失败时置位，表示"LLM 通道整体不可用"
        self.partial_warning = "" # 部分调用失败时的提示
        # 重试由我们自己控制（要指数退避 + 区分可重试的错），所以关掉 SDK 的重试
        self._client = OpenAI(base_url=api_base, api_key=api_key,
                              timeout=timeout, max_retries=0)

    def read_row(self, rgb: np.ndarray, labels: list[str] | None = None) -> RowVerdict | None:
        """读一行评分：返回 (分值, 标记数)，无法判断返回 None。

        429 限流和超时会做指数退避重试 —— 免费档模型并发很低，不重试基本跑不完。
        """
        b64 = base64.b64encode(_png_bytes(annotate_row(rgb, labels))).decode("ascii")
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": ROW_PROMPT},
        ]
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                )
            except (RateLimitError, APITimeoutError, APIConnectionError,
                    InternalServerError) as exc:
                # 429 限流 / 超时 / 5xx：退避后重试
                last_exc = exc
                if attempt < self.max_attempts:
                    delay = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.6)
                    time.sleep(delay)
                continue
            except Exception as exc:  # noqa: BLE001 - 密钥错/模型不存在等，重试无意义
                last_exc = exc
                break

            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            self.usage.add(getattr(usage, "prompt_tokens", 0) or 0,
                           getattr(usage, "completion_tokens", 0) or 0)
            return parse_verdict(text)

        self.usage.failures += 1
        self.last_error = f"{type(last_exc).__name__}: {str(last_exc)[:300]}"
        return None

    def estimated_cost(self, input_price_per_m: float = 1.0,
                       output_price_per_m: float = 10.0) -> float:
        return (self.usage.prompt_tokens / 1_000_000 * input_price_per_m
                + self.usage.completion_tokens / 1_000_000 * output_price_per_m)
