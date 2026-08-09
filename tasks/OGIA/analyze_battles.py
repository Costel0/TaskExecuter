from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence


DEFAULT_INPUT = Path("data/OGIA/random_battles/merged_battles_1.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/OGIA/Analisis")

# Stable ranges make reports from different datasets directly comparable.
POINT_EDGES = (
    0.0,
    500.0,
    1_000.0,
    2_000.0,
    5_000.0,
    10_000.0,
    20_000.0,
    50_000.0,
    100_000.0,
    200_000.0,
    500_000.0,
    float("inf"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogia-analyze-battles",
        description=(
            "Analyze an OGIA merged JSONL battle dataset and generate a "
            "self-contained HTML report."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL dataset (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional full report path. By default the filename is "
            "<input_stem>_analysis.html inside --output-dir."
        ),
    )
    return parser


def _mapping(container: dict, key: str, line_number: int) -> dict:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(
            f"Line {line_number}: expected '{key}' to be a JSON object."
        )
    return value


def _profit_total(result: dict, key: str, line_number: int) -> int:
    values = _mapping(result, key, line_number)
    return int(sum(int(value) for value in values.values()))


def _percentage(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def _format_points(value: float) -> str:
    if value == float("inf"):
        return "∞"
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def _bin_label(lower: float, upper: float) -> str:
    if upper == float("inf"):
        return f"≥ {_format_points(lower)}"
    return f"{_format_points(lower)}–{_format_points(upper)}"


def _point_bin_index(points: float) -> int:
    for index in range(len(POINT_EDGES) - 1):
        if POINT_EDGES[index] <= points < POINT_EDGES[index + 1]:
            return index
    return len(POINT_EDGES) - 2


def analyze_file(source: Path) -> dict:
    outcome_counts = {"attacker": 0, "defender": 0, "draw": 0}
    bin_battles = [0] * (len(POINT_EDGES) - 1)
    bin_wins = [0] * (len(POINT_EDGES) - 1)
    total = 0
    profitable_minimum = 0
    profitable_maximum = 0
    minimum_profit_sum = 0
    maximum_profit_sum = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number}: battle record must be a JSON object."
                )

            result = _mapping(record, "result", line_number)
            composition = _mapping(record, "composition_details", line_number)
            defender = _mapping(composition, "defender", line_number)

            try:
                winner = str(result["winner"])
                defender_points = float(defender["actual_points"])
            except KeyError as exc:
                raise ValueError(
                    f"Line {line_number}: missing field {exc.args[0]!r}."
                ) from exc

            minimum_profit = _profit_total(
                result,
                "minimum_profit",
                line_number,
            )
            maximum_profit = _profit_total(
                result,
                "maximum_profit",
                line_number,
            )

            total += 1
            outcome_counts[winner] = outcome_counts.get(winner, 0) + 1
            minimum_profit_sum += minimum_profit
            maximum_profit_sum += maximum_profit

            if minimum_profit > 0:
                profitable_minimum += 1
            if maximum_profit > 0:
                profitable_maximum += 1

            bin_index = _point_bin_index(defender_points)
            bin_battles[bin_index] += 1
            if winner == "attacker":
                bin_wins[bin_index] += 1

    if total == 0:
        raise ValueError(f"No battle records found in {source}.")

    return {
        "total": total,
        "outcome_counts": outcome_counts,
        "profitable_minimum": profitable_minimum,
        "profitable_maximum": profitable_maximum,
        "minimum_profit_sum": minimum_profit_sum,
        "maximum_profit_sum": maximum_profit_sum,
        "bin_battles": bin_battles,
        "bin_wins": bin_wins,
    }


def _svg_histogram(labels: list[str], wins: list[int]) -> str:
    width = 1120
    height = 430
    left, right, top, bottom = 72, 24, 34, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(1, max(wins, default=0))
    slot = plot_width / max(1, len(wins))
    bar_width = slot * 0.68

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Victorias del atacante por rango de puntos del defensor">'
    ]

    for tick in range(6):
        value = round(y_max * tick / 5)
        y = top + plot_height - (value / y_max) * plot_height
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
            f'class="axis">{value}</text>'
        )

    for index, (label, value) in enumerate(zip(labels, wins)):
        x = left + index * slot + (slot - bar_width) / 2
        bar_height = value / y_max * plot_height
        y = top + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" rx="4" class="bar"/>'
        )
        parts.append(
            f'<text x="{x + bar_width/2:.1f}" y="{max(top+12, y-7):.1f}" '
            f'text-anchor="middle" class="value">{value}</text>'
        )
        parts.append(
            f'<text transform="translate({x + bar_width/2:.1f},'
            f'{top + plot_height + 18:.1f}) rotate(35)" '
            f'text-anchor="start" class="axis">{html.escape(label)}</text>'
        )

    parts.append(
        f'<text x="{left + plot_width/2:.1f}" y="{height-10}" '
        'text-anchor="middle" class="axis-title">Puntos reales del defensor</text>'
    )
    parts.append(
        f'<text transform="translate(18,{top + plot_height/2:.1f}) rotate(-90)" '
        'text-anchor="middle" class="axis-title">Victorias del atacante</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _render_report(source: Path, stats: dict) -> str:
    total = stats["total"]
    outcomes = stats["outcome_counts"]
    attacker_wins = outcomes.get("attacker", 0)
    defender_wins = outcomes.get("defender", 0)
    draws = outcomes.get("draw", 0)
    profitable_minimum = stats["profitable_minimum"]
    profitable_maximum = stats["profitable_maximum"]

    labels = [
        _bin_label(POINT_EDGES[index], POINT_EDGES[index + 1])
        for index in range(len(POINT_EDGES) - 1)
    ]
    chart = _svg_histogram(labels, stats["bin_wins"])

    rows = []
    for label, battles, wins in zip(
        labels,
        stats["bin_battles"],
        stats["bin_wins"],
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{battles:,}</td>"
            f"<td>{wins:,}</td>"
            f"<td>{_percentage(wins, battles):.1f}%</td>"
            "</tr>"
        )

    avg_min = stats["minimum_profit_sum"] / total
    avg_max = stats["maximum_profit_sum"] / total
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OGIA · Análisis de batallas</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--text:#17202a;--muted:#65717e;--accent:#2962ff;--border:#dfe5eb;--good:#137333}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text)}}
main{{max-width:1220px;margin:auto;padding:34px 24px 56px}}h1{{margin:0 0 6px;font-size:32px}}h2{{margin:0 0 18px;font-size:21px}}
.meta,.note{{color:var(--muted);line-height:1.55}}.meta{{margin-bottom:26px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:14px;margin-bottom:22px}}
.card,.panel{{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:0 1px 2px #00000009}}.card{{padding:18px}}.label{{color:var(--muted);font-size:13px;margin-bottom:7px}}.number{{font-size:28px;font-weight:700}}.detail{{color:var(--muted);font-size:13px;margin-top:5px}}.good{{color:var(--good)}}
.panel{{padding:22px;margin-top:18px;overflow:auto}}svg{{display:block;width:100%;min-width:760px;height:auto}}.grid{{stroke:#e8edf2;stroke-width:1}}.bar{{fill:var(--accent)}}.axis{{fill:#5d6975;font-size:12px}}.axis-title{{fill:#3e4852;font-size:13px;font-weight:600}}.value{{fill:#2c3640;font-size:12px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--border)}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:600}}.note{{font-size:13px;margin-top:12px}}code{{background:#eef1f4;border-radius:4px;padding:2px 5px}}
</style>
</head>
<body><main>
<h1>OGIA · Análisis de batallas</h1>
<div class="meta">Fuente: <code>{html.escape(str(source))}</code><br>Reporte generado: {html.escape(generated)}</div>
<div class="cards">
<div class="card"><div class="label">Batallas analizadas</div><div class="number">{total:,}</div></div>
<div class="card"><div class="label">Victorias atacante</div><div class="number">{attacker_wins:,}</div><div class="detail">{_percentage(attacker_wins,total):.1f}% del total</div></div>
<div class="card"><div class="label">Victorias defensor</div><div class="number">{defender_wins:,}</div><div class="detail">{_percentage(defender_wins,total):.1f}% del total</div></div>
<div class="card"><div class="label">Empates</div><div class="number">{draws:,}</div><div class="detail">{_percentage(draws,total):.1f}% del total</div></div>
<div class="card"><div class="label">Beneficio mínimo &gt; 0</div><div class="number good">{profitable_minimum:,}</div><div class="detail">{_percentage(profitable_minimum,total):.1f}% del total</div></div>
<div class="card"><div class="label">Beneficio máximo &gt; 0</div><div class="number">{profitable_maximum:,}</div><div class="detail">{_percentage(profitable_maximum,total):.1f}% del total</div></div>
</div>
<section class="panel"><h2>Victorias por rango de puntos del defensor</h2>{chart}<div class="note">Las barras muestran victorias absolutas del atacante usando <strong>los puntos reales de la defensa</strong>.</div></section>
<section class="panel"><h2>Detalle por rango</h2><table><thead><tr><th>Rango de puntos</th><th>Batallas</th><th>Victorias atacante</th><th>Win rate</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="panel"><h2>Beneficio</h2><table><tbody><tr><th>Beneficio mínimo medio por batalla</th><td>{avg_min:,.0f}</td></tr><tr><th>Beneficio máximo medio por batalla</th><td>{avg_max:,.0f}</td></tr></tbody></table><div class="note"><strong>Beneficio mínimo</strong> = loot − coste de naves atacantes destruidas. <strong>Beneficio máximo</strong> = beneficio mínimo + todos los escombros generados. La cifra principal de batalla beneficiosa usa <code>minimum_profit_total &gt; 0</code>.</div></section>
</main></body></html>"""


def run(args: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    options = parser.parse_args(list(args or []))

    source = options.input.expanduser()
    if not source.is_file():
        parser.error(f"Input dataset not found: {source}")

    output = (
        options.output.expanduser()
        if options.output is not None
        else options.output_dir.expanduser() / f"{source.stem}_analysis.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    stats = analyze_file(source)
    output.write_text(_render_report(source, stats), encoding="utf-8")

    attacker_wins = stats["outcome_counts"].get("attacker", 0)
    print(f"Battles analyzed: {stats['total']:,}")
    print(
        "Profitable battles (minimum profit > 0): "
        f"{stats['profitable_minimum']:,} "
        f"({_percentage(stats['profitable_minimum'], stats['total']):.1f}%)"
    )
    print(
        f"Attacker wins: {attacker_wins:,} "
        f"({_percentage(attacker_wins, stats['total']):.1f}%)"
    )
    print(f"Report: {output}")
    return 0
