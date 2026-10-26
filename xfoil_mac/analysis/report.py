"""Small self-contained reports.

Include explicit links to scientific source files.
"""

import html
import json
from pathlib import Path
from urllib.parse import quote

from ..data import atomic_text

VERDICT_LABELS = {
    "ok": ("可信", "#10b981", "#064e3b"),
    "suspect": ("超出有效域", "#f59e0b", "#451a03"),
    "invalid": ("求解失败", "#f43f5e", "#4c0519"),
    "retry": ("孤立失败，可重试", "#a855f7", "#3b0764"),
    "unchecked": ("无证据", "#94a3b8", "#1e293b"),
}
"""Verdict wording, and the colours it is shown in.

The wording deliberately avoids "stall": the solver leaving its
documented domain is not evidence that the real flow separated.
"""

TRUST_EXPLANATION = (
    "XFOIL 报告 status=ok 且 convergence_rate=1.0 的曲线，边界层仍可能已经"
    "大面积分离，超出其文档声明支持的范围（仅限有限后缘分离，不支持大分离"
    "与激波）。判定依据是分离拓扑：同一表面上同时出现前缘泡与后缘分离、"
    "或单个分离段覆盖大部分弦长。判定描述的是求解器可信域，不是真实失速。"
)


def trust_card(summary: dict) -> str:
    """A verdict banner for a report, or an empty string.

    Rendered from a summary produced by the trust layer, so a report
    never states a conclusion the evidence did not support.
    """
    if not summary or not summary.get("points"):
        return ""
    counts = summary.get("counts") or {}
    order = ("ok", "suspect", "invalid", "retry", "unchecked")
    chips = []
    for name in order:
        count = counts.get(name)
        if not count:
            continue
        label, colour, background = VERDICT_LABELS.get(
            name, (name, "#94a3b8", "#1e293b")
        )
        chips.append(
            f'<span class="chip" style="color:{colour};'
            f'background:{background}">{html.escape(label)}'
            f" {count}</span>"
        )
    limit = summary.get("last_trusted_alpha")
    if limit is None:
        boundary = "没有可信角度的证据。"
    else:
        boundary = f"可信角度上界约 {limit:g}°。"
    flag = summary.get("first_suspect_alpha")
    if flag is not None:
        boundary += f" 首次标记出现在 {flag:g}°。"
    return (
        '<section class="data-card verdict"><div class="verdict-body">'
        '<p class="verdict-title">数据可信度</p>'
        f'<p class="verdict-line">{escape_text(boundary)}</p>'
        f'<p class="verdict-chips">{"".join(chips)}</p>'
        f"<details><summary>判定依据</summary>"
        f"<pre>{escape_text(TRUST_EXPLANATION)}</pre></details>"
        "</div></section>"
    )


def escape_text(value) -> str:
    return html.escape(str(value))


def report_page(
    destination: Path,
    title: str,
    description: str,
    rows: list[dict],
    links=(),
    images=(),
    trust=None,
):
    columns = list(dict.fromkeys(k for row in rows for k in row))

    def escape(value):
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        return html.escape(str(value if value is not None else "—"))

    def cell(value):
        if isinstance(value, dict) or (
            isinstance(value, list) and len(str(value)) > 100
        ):
            detail = json.dumps(value, ensure_ascii=False, indent=2)
            return (
                "<details><summary>查看详细数据</summary>"
                f"<pre>{escape(detail)}</pre></details>"
            )
        return escape(value)

    cells = "".join(
        "<tr>"
        + "".join(f"<td>{cell(row.get(k))}</td>" for k in columns)
        + "</tr>"
        for row in rows
    )
    navigation = "".join(
        f'<a href="{quote(str(path), safe="/._-")}">{escape(label)}</a> '
        for label, path in links
    )
    pictures = "".join(
        f'<figure><img src="{quote(str(path), safe="/._-")}" '
        f'alt="{escape(label)}" loading="lazy">'
        f"<figcaption>{escape(label)}</figcaption>"
        "</figure>"
        for label, path in images
    )
    headings = "".join(
        f'<th scope="col">{escape(column)}</th>' for column in columns
    )
    table = (
        '<section class="data-card"><div class="scroll" tabindex="0" '
        'role="region" aria-label="计算数据表"><table><thead><tr>'
        f"{headings}"
        f"</tr></thead><tbody>{cells}</tbody></table></div></section>"
        if columns
        else ""
    )
    verdict_card = trust_card(trust) if trust else ""
    ttf_path = Path(__file__).resolve().parent.parent / "assets" / "xfoil_hud_5x7.ttf"
    font_face_css = ""
    if ttf_path.exists():
        import base64

        b64 = base64.b64encode(ttf_path.read_bytes()).decode("ascii")
        font_face_css = (
            '@font-face{font-family:"XFOIL HUD 5x7";'
            f'src:url("data:font/ttf;base64,{b64}") format("truetype");'
            "font-weight:normal;font-style:normal;font-display:swap}\n"
        )
    atomic_text(
        destination,
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">\n'
        '<meta name="viewport" '
        'content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title>\n"
        f"<style>\n{font_face_css}"
        """:root{color-scheme:dark;font:13px/1.65 "XFOIL HUD 5x7","PingFang SC","Heiti TC",ui-monospace,
"SFMono-Regular",Menlo,monospace;background:#04070e;color:#e2e8f0}
*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;
padding:42px 28px 28px}.eyebrow{font-size:12px;font-weight:600;
color:#38bdf8;letter-spacing:.035em;margin:0 0 12px}
h1{font-size:28px;line-height:1.3;font-weight:600;letter-spacing:.02em;
margin:0 0 16px;overflow-wrap:anywhere;color:#38bdf8}.description{color:#94a3b8;
max-width:880px;margin:0;white-space:pre-line}
a{color:#38bdf8;text-decoration:none}a:hover{text-decoration:underline}
nav{display:flex;flex-wrap:wrap;gap:9px;margin:26px 0}
nav a{display:inline-flex;align-items:center;padding:8px 13px;
border:1px solid #2ea8ff;border-radius:6px;background:#090d16;color:#e2e8f0;font-size:12px;
overflow-wrap:anywhere}nav a:hover{background:#162032;border-color:#38bdf8}
a:focus-visible,summary:focus-visible,.scroll:focus-visible{
outline:3px solid #38bdf855;outline-offset:3px}
.data-card{background:#090d16;border:1px solid #2ea8ff;border-radius:8px;
overflow:hidden;box-shadow:0 4px 16px #00000066}.scroll{overflow:auto}
table{border-collapse:collapse;font-size:12px;width:100%;
font-variant-numeric:tabular-nums}td,th{padding:12px 16px;text-align:left;
border-bottom:1px solid #162032;vertical-align:top}th{white-space:nowrap;
font-size:11px;font-weight:600;color:#38bdf8;background:#060a12}
tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#16203288}
td{max-width:600px;overflow-wrap:anywhere;color:#e2e8f0}details{min-width:120px}
summary{cursor:pointer;color:#38bdf8;font-size:12px}
pre{font:11px/1.6 "XFOIL HUD 5x7","PingFang SC",ui-monospace,"SFMono-Regular",Menlo,monospace;
white-space:pre-wrap;overflow-wrap:anywhere;max-width:480px;
padding:12px;background:#04070e;color:#e2e8f0;border:1px solid #162032;border-radius:6px;margin:10px 0 0}
figure{margin:24px 0;background:#090d16;border:1px solid #2ea8ff;
border-radius:8px;overflow:hidden}img{max-width:100%;display:block;
margin:auto;background:#04070e}figcaption{padding:12px 18px;
color:#94a3b8;font-size:12px;border-top:1px solid #162032}
footer{color:#64748b;font-size:12px;padding:28px 0 6px}
.verdict{margin:0 0 22px}.verdict-body{padding:18px 20px}
.verdict-title{margin:0 0 8px;font-size:12px;font-weight:600;
color:#38bdf8;letter-spacing:.035em}
.verdict-line{margin:0 0 12px;font-size:13px;color:#e2e8f0}
.verdict-chips{margin:0;display:flex;flex-wrap:wrap;gap:8px}
.chip{display:inline-flex;align-items:center;padding:5px 11px;
border-radius:4px;font-size:11px;font-weight:600;
font-variant-numeric:tabular-nums}
.verdict details{margin-top:12px}
@media(max-width:640px){main{padding:28px 16px 20px}h1{font-size:22px}
nav{margin:22px 0;gap:8px}nav a{padding:8px 11px}td,th{padding:11px 13px}
.data-card,figure{border-radius:6px}}
@media print{body{background:#04070e}main{padding:0;max-width:none}
.data-card,figure{box-shadow:none;break-inside:avoid}
.scroll{overflow:visible}nav a{border:0;padding:0;margin-right:12px}}
</style>\n"""
        '<body><main><header><p class="eyebrow">XFOIL · 计算报告</p>'
        f'<h1>{escape(title)}</h1><p class="description">'
        f"{escape(description)}</p></header>"
        f'<nav aria-label="结果与原始数据">{navigation}</nav>'
        f"{verdict_card}{table}{pictures}<footer>保留实际求解结果。"
        "缺失或无效数据不以默认值补齐。</footer></main></body></html>",
    )
    return destination
