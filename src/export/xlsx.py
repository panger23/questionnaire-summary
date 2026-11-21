"""生成汇总 Excel（XlsxWriter）。

关键设计：
  1. 汇总表里每个人的分数 = 对「明细」合计列的公式引用，平均分 = AVERAGE 公式。
     用户在「明细」改任何一个勾的分，合计、汇总表里那个人、该分行平均分都自动重算。
  2. 每个公式都同时写入缓存值，所以用任何软件打开（含没有计算引擎的预览器）
     都能立刻看到正确数字，不会显示 0 或空白。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import xlsxwriter

from ..engine.pipeline import PageResult, average, group_by_branch

DETAIL_COLUMNS = (
    "序号", "页码", "分行", "审计组长", "组长助手",
    "任务分工", "沟通协调", "团结协作", "专业技术", "关爱扶持",
    "合计", "LLM合计", "LLM标记数", "双通道一致", "标记",
)
COL_TOTAL = 10  # 0 起：K 列，合计
COL_SCORE_FIRST = 5
COL_SCORE_LAST = 9


def _col_letter(index: int) -> str:
    return xlsxwriter.utility.xl_col_to_name(index)


def build_workbook(results: list[PageResult], pdf_path: str, out_path: str | Path,
                   meta: dict | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(out_path))

    head_fmt = wb.add_format({"bold": True, "bg_color": "#D9E2F3", "border": 1,
                              "align": "center", "valign": "vcenter", "text_wrap": True})
    cell_fmt = wb.add_format({"border": 1, "align": "center", "valign": "vcenter"})
    left_fmt = wb.add_format({"border": 1, "align": "left", "valign": "vcenter", "text_wrap": True})
    warn_fmt = wb.add_format({"border": 1, "align": "center", "valign": "vcenter",
                              "bg_color": "#FFF2CC"})
    warn_left_fmt = wb.add_format({"border": 1, "align": "left", "valign": "vcenter",
                                   "bg_color": "#FFF2CC", "text_wrap": True})
    bold_fmt = wb.add_format({"bold": True, "border": 1, "align": "center", "valign": "vcenter"})
    bold_head_fmt = wb.add_format({"bold": True, "bg_color": "#D9E2F3", "border": 1,
                                   "align": "center", "valign": "vcenter", "text_wrap": True})
    avg_fmt = wb.add_format({"bold": True, "border": 1, "align": "center",
                             "valign": "vcenter", "num_format": "0.00"})
    int_fmt = wb.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": "0"})
    info_key = wb.add_format({"bold": True, "border": 1, "align": "left", "valign": "vcenter"})
    info_val = wb.add_format({"border": 1, "align": "left", "valign": "vcenter", "text_wrap": True})

    branches = group_by_branch(results)
    detail_row_of = {id(r): i + 2 for i, r in enumerate(results)}  # Excel 行号（1 起）

    # ---------------- 明细 ----------------
    ws = wb.add_worksheet("明细")
    for c, title in enumerate(DETAIL_COLUMNS):
        ws.write(0, c, title, head_fmt)
    for i, r in enumerate(results):
        excel_row = i + 2
        flagged = bool(r.flags) or r.total is None
        row_fmt = warn_fmt if flagged else cell_fmt
        row_left = warn_left_fmt if flagged else left_fmt
        ws.write_number(i + 1, 0, i + 1, row_fmt)
        ws.write_number(i + 1, 1, r.page, row_fmt)
        ws.write_string(i + 1, 2, r.header.project, row_left)
        ws.write_string(i + 1, 3, r.header.lead, row_fmt)
        ws.write_string(i + 1, 4, r.header.assistant, row_fmt)
        for k, score in enumerate(r.scores):
            if score is None:
                ws.write_blank(i + 1, COL_SCORE_FIRST + k, None, row_fmt)
            else:
                ws.write_number(i + 1, COL_SCORE_FIRST + k, score, row_fmt)
        first = _col_letter(COL_SCORE_FIRST)
        last = _col_letter(COL_SCORE_LAST)
        formula = f"=SUM({first}{excel_row}:{last}{excel_row})"
        if r.total is None:
            ws.write_blank(i + 1, COL_TOTAL, None, row_fmt)
        else:
            ws.write_formula(i + 1, COL_TOTAL, formula, row_fmt, r.total)
        if r.llm_total is not None:
            ws.write_number(i + 1, 11, r.llm_total, row_fmt)
        elif r.llm_scores is not None:
            ws.write_string(i + 1, 11, "—", row_fmt)
        else:
            ws.write_blank(i + 1, 11, None, row_fmt)
        if r.llm_marks is not None:
            ws.write_string(i + 1, 12, "/".join(str(m) for m in r.llm_marks), row_fmt)
        else:
            ws.write_blank(i + 1, 12, None, row_fmt)
        agree_text = "" if r.agree is None else ("一致" if r.agree else "不一致")
        ws.write_string(i + 1, 13, agree_text, row_fmt)
        ws.write_string(i + 1, 14, "；".join(r.flags) if r.flags else "", row_left)

    widths = {"A": 6, "B": 6, "C": 26, "D": 10, "E": 10,
              "F": 9, "G": 9, "H": 9, "I": 9, "J": 9, "K": 8, "L": 9,
              "M": 12, "N": 10, "O": 52}
    for letter, width in widths.items():
        ws.set_column(f"{letter}:{letter}", width)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(results), len(DETAIL_COLUMNS) - 1)

    # ---------------- 汇总 ----------------
    sm = wb.add_worksheet("汇总")
    max_n = max((len(items) for _, items in branches), default=0)
    sm.write(0, 0, "序号", bold_head_fmt)
    sm.write(1, 0, "组员数量", bold_head_fmt)
    sm.write(2, 0, "代职人", bold_head_fmt)

    for j, (name, items) in enumerate(branches):
        col = 1 + j
        sm.write_string(0, col, name, bold_head_fmt)
        sm.write_number(1, col, len(items), int_fmt)
        sm.write_string(2, col, items[0].header.assistant, cell_fmt)
        for i, r in enumerate(items):
            detail_row = detail_row_of[id(r)]
            formula = f"='明细'!{_col_letter(COL_TOTAL)}{detail_row}"
            cached = r.total
            if cached is None:
                sm.write_blank(3 + i, col, None, cell_fmt)
            else:
                sm.write_formula(3 + i, col, formula, cell_fmt, cached)

    for i in range(max_n):
        sm.write_number(3 + i, 0, i + 1, cell_fmt)

    avg_row = 3 + max_n
    sm.write(avg_row, 0, "平均分", bold_fmt)
    for j, (_, items) in enumerate(branches):
        col = 1 + j
        letter = _col_letter(col)
        first_row, last_row = 4, 3 + len(items)
        formula = f"=AVERAGE({letter}{first_row}:{letter}{last_row})"
        sm.write_formula(avg_row, col, formula, avg_fmt, average(items))

    sm.set_column("A:A", 12)
    for j in range(len(branches)):
        sm.set_column(1 + j, 1 + j, 24)
    sm.freeze_panes(1, 1)

    # ---------------- 说明 ----------------
    nf = wb.add_worksheet("说明")
    meta = meta or {}
    flagged = [r for r in results if r.flags or r.total is None]
    mode = meta.get("mode", "纯代码模式（离线，墨迹判列；未启用 LLM 复核）")
    rows: list[tuple[str, str]] = [
        ("项", "值"),
        ("源文件", Path(pdf_path).name),
        ("处理时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("问卷份数", str(len(results))),
        ("识别模式", mode),
        ("需人工复核份数", str(len(flagged))),
        ("需复核页码", "、".join(str(r.page) for r in flagged) or "无"),
    ]
    if meta.get("llm_calls") is not None:
        agree = sum(1 for r in results if r.agree is True)
        disagree = sum(1 for r in results if r.agree is False)
        rows += [
            ("LLM 模型", str(meta.get("model", ""))),
            ("双通道一致 / 不一致", f"{agree} 份 / {disagree} 份"),
            ("LLM 调用次数", f"{meta.get('llm_calls', 0)} 次，失败 {meta.get('llm_failures', 0)} 次"),
            ("LLM 用量", f"输入 {meta.get('prompt_tokens', 0)} tokens，"
                         f"输出 {meta.get('completion_tokens', 0)} tokens"),
            ("LLM 预估花费", f"约 ¥{meta.get('cost', 0):.2f}"),
        ]
    for name, items in branches:
        avg = average(items)
        rows.append((name, f"{len(items)} 份，平均分 {avg:.6f}" if avg is not None else "—"))
    rows += [
        ("分数规则", "5 项勾选分值之和；手写总分、小计列、备注一律不读"),
        ("平均分规则", "五项齐全的有效问卷做算术平均"),
        ("勾选分值", "10/9/8/7/6/5/4/3/2/1 分"),
        ("标黄含义", "该行存在需要人工确认的情况（墨迹跨格 / 缺勾 / 表头未识别 / 双通道不一致）"),
        ("多标记规则", "一行多个标记时取打勾（√）的那个，画叉（×）视为作废；"
                       "仅当代码通道两格墨量都较重时才自动采用 LLM 判断"),
    ]
    for i, (key, value) in enumerate(rows):
        fmt_k = head_fmt if i == 0 else info_key
        fmt_v = head_fmt if i == 0 else info_val
        nf.write_string(i, 0, key, fmt_k)
        nf.write_string(i, 1, value, fmt_v)
    nf.set_column("A:A", 26)
    nf.set_column("B:B", 70)

    wb.close()
    return out_path
