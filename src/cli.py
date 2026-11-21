"""命令行入口：python -m src.cli <扫描件.pdf> [-o 输出.xlsx] [--llm]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .engine.pipeline import average, group_by_branch, process_pdf
from .export.xlsx import build_workbook
from .llm.client import VisionClient
from .llm.config import config_path, load_config, masked_key


def _fmt_avg(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _make_llm_client(args):
    cfg = load_config()
    api_base = args.api_base or cfg["api_base"]
    model = args.model or cfg["model"]
    api_key = cfg.get("api_key", "")
    if not api_key:
        print("启用了 LLM 复核，但没有找到 API Key。", file=sys.stderr)
        print(f"请把 key 写进配置：{config_path()}", file=sys.stderr)
        print('格式：{"api_base": "...", "model": "qwen3-vl-plus", "api_key": "sk-..."}',
              file=sys.stderr)
        return None
    print(f"LLM 复核通道：{model}（key {masked_key(api_key)}）")
    return VisionClient(api_base=api_base, api_key=api_key, model=model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="把扫描的调查问卷 PDF 识别成汇总 Excel",
    )
    parser.add_argument("pdf", help="扫描件 PDF 路径")
    parser.add_argument("-o", "--output", help="输出 xlsx 路径（默认 output/ 下自动命名）")
    parser.add_argument("--dpi", type=int, default=300, help="渲染分辨率，默认 300")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 复核通道（需要 API Key）")
    parser.add_argument("--model", help="覆盖配置里的模型名，例如 qwen3-vl-plus")
    parser.add_argument("--api-base", help="覆盖配置里的 API 地址")
    parser.add_argument("--workers", type=int, default=6, help="LLM 并发数，默认 6")
    parser.add_argument("-q", "--quiet", action="store_true", help="不打印逐页进度")
    args = parser.parse_args(argv)

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"找不到文件：{pdf}", file=sys.stderr)
        return 2

    client = None
    if args.llm:
        client = _make_llm_client(args)
        if client is None:
            return 2

    started = time.time()
    last = {"shown": 0}

    def progress(page: int, total: int) -> None:
        if args.quiet or page == total:
            return
        if page - last["shown"] >= 10:
            last["shown"] = page
            print(f"  已处理 {page} 页…", flush=True)

    print(f"正在处理：{pdf.name}")
    results = process_pdf(str(pdf), dpi=args.dpi, progress=progress,
                          llm_client=client, workers=args.workers)
    elapsed = time.time() - started

    meta: dict = {}
    if client is not None:
        meta = {
            "mode": f"纯代码 + LLM 复核（{client.model}）",
            "model": client.model,
            "llm_calls": client.usage.calls,
            "llm_failures": client.usage.failures,
            "prompt_tokens": client.usage.prompt_tokens,
            "completion_tokens": client.usage.completion_tokens,
            "cost": client.estimated_cost(),
        }

    out = Path(args.output) if args.output else (
        Path("output") / f"{pdf.stem}_汇总_{time.strftime('%Y%m%d_%H%M')}.xlsx"
    )
    build_workbook(results, str(pdf), out, meta=meta)

    print()
    print(f"共 {len(results)} 份问卷，用时 {elapsed:.1f}s")
    print(f"{'分行':<28}{'份数':>5}{'代职人':>10}{'平均分':>12}")
    print("-" * 58)
    for name, items in group_by_branch(results):
        print(f"{name:<28}{len(items):>5}{items[0].header.assistant:>10}"
              f"{_fmt_avg(average(items)):>12}")

    if client is not None:
        agree = sum(1 for r in results if r.agree is True)
        disagree = sum(1 for r in results if r.agree is False)
        failed = client.usage.failures
        print()
        print(f"LLM 复核：一致 {agree} 份 / 不一致 {disagree} 份 / 失败 {failed} 份"
              f"；用量 {client.usage.prompt_tokens}+{client.usage.completion_tokens} tokens"
              f"，约 ¥{client.estimated_cost():.2f}")

    flagged = [r for r in results if r.flags or r.total is None]
    print()
    if flagged:
        print(f"需要人工复核 {len(flagged)} 份：")
        for r in flagged:
            print(f"  第 {r.page} 页（{r.header.project or '未知'}）：{'；'.join(r.flags) or '分数不全'}")
    else:
        print("没有需要人工复核的问卷。")

    print()
    print(f"已生成：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
