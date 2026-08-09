from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence


DEFAULT_INPUT = Path("data/OGIA/random_battles/merged_battles_1.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/OGIA/Analisis")

# Multiplier buckets use the real attacker/defender fleet points.
# The random generator normally targets 1.2x-5.0x, but approximate integer
# fleet construction can move the realised ratio slightly outside that range.
RATIO_STEP = 0.1
RATIO_MIN = 1.0
RATIO_MAX = 5.0
RATIO_EDGES = tuple(
    round(RATIO_MIN + index * RATIO_STEP, 10)
    for index in range(int(round((RATIO_MAX - RATIO_MIN) / RATIO_STEP)) + 1)
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


def _ratio_bin_labels() -> list[str]:
    labels = [f"< {RATIO_MIN:.1f}x"]
    labels.extend(
        f"{RATIO_EDGES[index]:.1f}x–{RATIO_EDGES[index + 1]:.1f}x"
        for index in range(len(RATIO_EDGES) - 1)
    )
    labels.append(f"≥ {RATIO_MAX:.1f}x")
    return labels


def _ratio_bin_index(ratio: float) -> int:
    if ratio < RATIO_MIN:
        return 0
    if ratio >= RATIO_MAX:
        return len(RATIO_EDGES)

    regular_index = int((ratio - RATIO_MIN) / RATIO_STEP)
    regular_index = min(regular_index, len(RATIO_EDGES) - 2)
    return 1 + regular_index


def _ratio_bin_center(index: int) -> float:
    if index == 0:
        return RATIO_MIN - RATIO_STEP / 2
    if index == len(RATIO_EDGES):
        return RATIO_MAX + RATIO_STEP / 2

    lower = RATIO_EDGES[index - 1]
    upper = RATIO_EDGES[index]
    return (lower + upper) / 2


def analyze_file(source: Path) -> dict:
    labels = _ratio_bin_labels()
    outcome_counts = {"attacker": 0, "defender": 0, "draw": 0}
    bin_battles = [0] * len(labels)
    bin_wins = [0] * len(labels)
    profitable_max_profit_sum = [0] * len(labels)
    profitable_counts = [0] * len(labels)

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
            attacker = _mapping(composition, "attacker", line_number)
            defender = _mapping(composition, "defender", line_number)

            try:
                winner = str(result["winner"])
                attacker_points = float(attacker["actual_points"])
                defender_points = float(defender["actual_points"])
            except KeyError as exc:
                raise ValueError(
                    f"Line {line_number}: missing field {exc.args[0]!r}."
                ) from exc

            if defender_points <= 0:
                raise ValueError(
                    f"Line {line_number}: defender actual_points must be > 0."
                )

            power_ratio = attacker_points / defender_points
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

            bin_index = _ratio_bin_index(power_ratio)
            bin_battles[bin_index] += 1
            if winner == "attacker":
                bin_wins[bin_index] += 1

            # "Profitable" keeps the conservative definition used by the summary:
            # loot minus destroyed attacker fleet cost must already be positive.
            if minimum_profit > 0:
                profitable_minimum += 1
                profitable_counts[bin_index] += 1
                profitable_max_profit_sum[bin_index] += maximum_profit

            if maximum_profit > 0:
                profitable_maximum += 1

    if total == 0:
        raise ValueError(f"No battle records found in {source}.")

    profitable_max_profit_mean = [
        (
            profit_sum / count
            if count
            else None
        )
        for profit_sum, count in zip(
            profitable_max_profit_sum,
            profitable_counts,
        )
    ]

    return {
        "total": total,
        "outcome_counts": outcome_counts,
        "profitable_minimum": profitable_minimum,
        "profitable_maximum": profitable_maximum,
        "minimum_profit_sum": minimum_profit_sum,
        "maximum_profit_sum": maximum_profit_sum,
        "bin_labels": labels,
        "bin_battles": bin_battles,
        "bin_wins": bin_wins,
        "profitable_counts": profitable_counts,
        "profitable_max_profit_mean": profitable_max_profit_mean,
    }


def _svg_histogram(labels: list[str], wins: list[int]) -> str:
    width = max(1400, 72 + len(wins) * 38)
    height = 470
    left, right, top, bottom = 72, 24, 34, 125
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(1, max(wins, default=0))
    slot = plot_width / max(1, len(wins))
    bar_width = slot * 0.72

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Victorias del atacante por multiplicador de puntos">'
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
            f'height="{bar_height:.1f}" rx="3" class="bar"/>'
        )
        if value:
            parts.append(
                f'<text x="{x + bar_width/2:.1f}" '
                f'y="{max(top+12, y-6):.1f}" text-anchor="middle" '
                f'class="value">{value}</text>'
            )
        parts.append(
            f'<text transform="translate({x + bar_width/2:.1f},'
            f'{top + plot_height + 16:.1f}) rotate(55)" '
            f'text-anchor="start" class="axis">{html.escape(label)}</text>'
        )

    parts.append(
        f'<text x="{left + plot_width/2:.1f}" y="{height-10}" '
        'text-anchor="middle" class="axis-title">'
        'Multiplicador real de puntos atacante / defensor</text>'
    )
    parts.append(
        f'<text transform="translate(18,{top + plot_height/2:.1f}) rotate(-90)" '
        'text-anchor="middle" class="axis-title">Victorias del atacante</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _format_resource_axis(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _svg_profit_line(
    labels: list[str],
    mean_profits: list[float | None],
    profitable_counts: list[int],
) -> str:
    points = [
        (_ratio_bin_center(index), float(value), profitable_counts[index])
        for index, value in enumerate(mean_profits)
        if value is not None
    ]

    width = 1160
    height = 460
    left, right, top, bottom = 90, 28, 34, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    if not points:
        return (
            '<div class="empty-chart">'
            "No hay batallas con beneficio mínimo positivo para representar."
            "</div>"
        )

    x_min = min(point[0] for point in points)
    x_max = max(point[0] for point in points)
    if x_max <= x_min:
        x_max = x_min + RATIO_STEP

    y_values = [point[1] for point in points]
    y_min = min(0.0, min(y_values))
    y_max = max(y_values)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Beneficio máximo medio frente a multiplicador de puntos">'
    ]

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_coord(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" '
            f'class="axis">{html.escape(_format_resource_axis(value))}</text>'
        )

    first_tick = int(x_min * 10 + 0.999999) / 10
    last_tick = int(x_max * 10 + 1e-9) / 10
    tick_value = first_tick
    while tick_value <= last_tick + 1e-9:
        x = x_coord(tick_value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
            f'y2="{top + plot_height}" class="grid vertical"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 22}" '
            f'text-anchor="middle" class="axis">{tick_value:.1f}x</text>'
        )
        tick_value = round(tick_value + 0.5, 10)

    polyline_points = " ".join(
        f"{x_coord(x):.1f},{y_coord(y):.1f}"
        for x, y, _count in points
    )
    parts.append(
        f'<polyline points="{polyline_points}" class="profit-line" '
        'fill="none"/>'
    )

    for x_value, y_value, count in points:
        x = x_coord(x_value)
        y = y_coord(y_value)
        title = (
            f"{x_value:.2f}x · beneficio máximo medio "
            f"{y_value:,.0f} · {count:,} batallas rentables"
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
            f'class="profit-point"><title>{html.escape(title)}</title></circle>'
        )

    parts.append(
        f'<text x="{left + plot_width/2:.1f}" y="{height-10}" '
        'text-anchor="middle" class="axis-title">'
        'Multiplicador real de puntos atacante / defensor</text>'
    )
    parts.append(
        f'<text transform="translate(20,{top + plot_height/2:.1f}) rotate(-90)" '
        'text-anchor="middle" class="axis-title">'
        'Beneficio máximo medio</text>'
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

    labels = stats["bin_labels"]
    histogram = _svg_histogram(labels, stats["bin_wins"])
    profit_line = _svg_profit_line(
        labels,
        stats["profitable_max_profit_mean"],
        stats["profitable_counts"],
    )

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
.panel{{padding:22px;margin-top:18px;overflow:auto}}svg{{display:block;width:100%;min-width:760px;height:auto}}.grid{{stroke:#e8edf2;stroke-width:1}}.vertical{{stroke:#f1f4f7}}.bar{{fill:var(--accent)}}.profit-line{{stroke:var(--accent);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}}.profit-point{{fill:var(--accent);stroke:white;stroke-width:1.5}}.axis{{fill:#5d6975;font-size:12px}}.axis-title{{fill:#3e4852;font-size:13px;font-weight:600}}.value{{fill:#2c3640;font-size:11px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--border)}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:600}}.note{{font-size:13px;margin-top:12px}}code{{background:#eef1f4;border-radius:4px;padding:2px 5px}}.empty-chart{{padding:48px;color:var(--muted);text-align:center}}
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

<section class="panel">
<h2>Victorias por superioridad de puntos del atacante</h2>
{histogram}
<div class="note">Cada rango usa el multiplicador <strong>puntos reales del atacante / puntos reales del defensor</strong>. Por ejemplo, 1.1x–1.2x significa que el atacante tiene aproximadamente entre un 10% y un 20% más de puntos.</div>
</section>

<section class="panel">
<h2>Detalle por multiplicador de puntos</h2>
<table><thead><tr><th>Atacante / defensor</th><th>Batallas</th><th>Victorias atacante</th><th>Win rate</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</section>

<section class="panel">
<h2>Beneficio</h2>
<table><tbody><tr><th>Beneficio mínimo medio por batalla</th><td>{avg_min:,.0f}</td></tr><tr><th>Beneficio máximo medio por batalla</th><td>{avg_max:,.0f}</td></tr></tbody></table>
<div class="note"><strong>Beneficio mínimo</strong> = loot − coste de naves atacantes destruidas. <strong>Beneficio máximo</strong> = beneficio mínimo + todos los escombros generados. La cifra principal de batalla beneficiosa usa <code>minimum_profit_total &gt; 0</code>.</div>
</section>

<section class="panel">
<h2>Beneficio máximo frente a superioridad de puntos</h2>
{profit_line}
<div class="note">Sólo se incluyen batallas realmente rentables según el criterio conservador <code>minimum_profit_total &gt; 0</code>. Cada punto de la línea representa el <strong>beneficio máximo medio</strong> de las batallas rentables dentro de un intervalo de 0.1x del multiplicador real atacante/defensor. Pasa el cursor sobre un punto para ver su valor y número de batallas.</div>
</section>
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
