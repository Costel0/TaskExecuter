from __future__ import annotations

import argparse
import html
import json
import statistics
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from .OgameData import UNIT_SPECS


DEFAULT_INPUT = Path("data/OGIA/random_battles/merged_battles_1.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/OGIA/Analisis")

# Multiplier buckets use real attacker/defender points.
RATIO_STEP = 0.1
RATIO_MIN = 1.0
RATIO_MAX = 5.0
RATIO_EDGES = tuple(
    round(RATIO_MIN + index * RATIO_STEP, 10)
    for index in range(int(round((RATIO_MAX - RATIO_MIN) / RATIO_STEP)) + 1)
)

# These are the ships currently sampled by random_battles.py.
ATTACK_SHIPS = (
    "small_cargo",
    "large_cargo",
    "light_fighter",
    "heavy_fighter",
    "cruiser",
    "battleship",
    "bomber",
    "destroyer",
    "battlecruiser",
    "reaper",
    "pathfinder",
)

REGULAR_DEFENSES = (
    "rocket_launcher",
    "light_laser",
    "heavy_laser",
    "gauss_cannon",
    "ion_cannon",
    "plasma_turret",
)

# Log-like defender-size buckets, in OGame points.
DEFENDER_POINT_EDGES = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)

TECH_BUCKET_LABELS = (
    "≤ -12",
    "-11 a -4",
    "-3 a +3",
    "+4 a +11",
    "≥ +12",
)

MIN_RATIO_SAMPLE = 20
MIN_RATIO_SUCCESS_SAMPLE = 10


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogia-analyze-battles",
        description=(
            "Analyze an OGIA merged JSONL battle dataset and generate an "
            "optimizer-oriented HTML + JSON report."
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
            "Optional full HTML report path. By default the filename is "
            "<input_stem>_analysis.html inside --output-dir."
        ),
    )
    return parser


def _mapping(container: Mapping, key: str, line_number: int) -> dict:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(
            f"Line {line_number}: expected '{key}' to be a JSON object."
        )
    return value


def _profit_total(result: dict, key: str, line_number: int) -> int:
    values = _mapping(result, key, line_number)
    return int(sum(int(value) for value in values.values()))


def _percentage(part: int | float, total: int | float) -> float:
    return 100.0 * float(part) / float(total) if total else 0.0


def _unit_points(unit_name: str) -> float:
    spec = UNIT_SPECS[unit_name]
    return float(spec.metal + spec.crystal + spec.deuterium) / 1_000.0


def _composition_point_shares(
    composition: Mapping[str, int],
    *,
    allowed_units: Sequence[str],
) -> dict[str, float]:
    point_values: dict[str, float] = {}
    total_points = 0.0

    for unit_name in allowed_units:
        quantity = int(composition.get(unit_name, 0))
        if quantity <= 0:
            continue
        value = quantity * _unit_points(unit_name)
        point_values[unit_name] = value
        total_points += value

    if total_points <= 0:
        return {}

    return {
        unit_name: value / total_points
        for unit_name, value in point_values.items()
    }


def _dominant_defense(defender: Mapping[str, int]) -> str:
    best_name = REGULAR_DEFENSES[0]
    best_points = -1.0

    for unit_name in REGULAR_DEFENSES:
        points = int(defender.get(unit_name, 0)) * _unit_points(unit_name)
        if points > best_points:
            best_name = unit_name
            best_points = points

    return best_name


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

    # Half-open buckets [lower, upper): exact 1.2x belongs to 1.2x–1.3x.
    return bisect_right(RATIO_EDGES, ratio)


def _ratio_bin_center(index: int) -> float:
    if index == 0:
        return RATIO_MIN - RATIO_STEP / 2
    if index == len(RATIO_EDGES):
        return RATIO_MAX + RATIO_STEP / 2

    lower = RATIO_EDGES[index - 1]
    upper = RATIO_EDGES[index]
    return (lower + upper) / 2


def _size_bin_labels() -> list[str]:
    edges = DEFENDER_POINT_EDGES
    labels = [f"< {edges[0]:,}"]
    labels.extend(
        f"{edges[index]:,}–{edges[index + 1]:,}"
        for index in range(len(edges) - 1)
    )
    labels.append(f"≥ {edges[-1]:,}")
    return labels


def _size_bin_index(points: float) -> int:
    return bisect_right(DEFENDER_POINT_EDGES, points)


def _tech_delta_bucket_index(delta: int) -> int:
    if delta <= -12:
        return 0
    if delta <= -4:
        return 1
    if delta <= 3:
        return 2
    if delta <= 11:
        return 3
    return 4


def _combat_tech_total(tech: Mapping[str, int]) -> int:
    return sum(int(tech.get(key, 0)) for key in ("weapons", "shielding", "armour"))


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _safe_median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def analyze_file(source: Path) -> dict:
    ratio_labels = _ratio_bin_labels()
    size_labels = _size_bin_labels()

    ratio_battles = [0] * len(ratio_labels)
    ratio_wins = [0] * len(ratio_labels)
    ratio_successes = [0] * len(ratio_labels)
    ratio_success_min_profit_sum = [0.0] * len(ratio_labels)
    ratio_success_max_profit_sum = [0.0] * len(ratio_labels)
    ratio_success_roi_sum = [0.0] * len(ratio_labels)

    size_battles = [0] * len(size_labels)
    size_wins = [0] * len(size_labels)
    size_successes = [0] * len(size_labels)
    size_success_ratio_sum = [0.0] * len(size_labels)

    tech_battles = [0] * len(TECH_BUCKET_LABELS)
    tech_successes = [0] * len(TECH_BUCKET_LABELS)

    ship_present = {ship: 0 for ship in ATTACK_SHIPS}
    ship_success_present = {ship: 0 for ship in ATTACK_SHIPS}
    ship_success_share_sum = {ship: 0.0 for ship in ATTACK_SHIPS}

    defense_profiles = {
        defense: {
            "battles": 0,
            "wins": 0,
            "successes": 0,
            "success_ratio_sum": 0.0,
            "success_ship_share_sum": {ship: 0.0 for ship in ATTACK_SHIPS},
        }
        for defense in REGULAR_DEFENSES
    }

    outcome_counts = {"attacker": 0, "defender": 0, "draw": 0}
    total = 0
    profitable_minimum = 0
    profitable_maximum = 0
    profitable_wins = 0
    minimum_profit_sum = 0.0
    maximum_profit_sum = 0.0
    tech_delta_sum = 0.0
    success_tech_delta_sum = 0.0

    success_ratios: list[float] = []
    success_rois: list[float] = []
    # Kept compact: only successful battles, only the information needed for
    # top-efficiency composition signals.
    successful_records: list[tuple[float, float, dict[str, float]]] = []

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
            attacker_details = _mapping(composition, "attacker", line_number)
            defender_details = _mapping(composition, "defender", line_number)
            inputs = _mapping(record, "inputs", line_number)
            attacker_fleet = _mapping(inputs, "attacker", line_number)
            defender_fleet = _mapping(inputs, "defender", line_number)
            attacker_tech = _mapping(inputs, "attacker_tech", line_number)
            defender_tech = _mapping(inputs, "defender_tech", line_number)

            try:
                winner = str(result["winner"])
                attacker_points = float(attacker_details["actual_points"])
                defender_points = float(defender_details["actual_points"])
            except KeyError as exc:
                raise ValueError(
                    f"Line {line_number}: missing field {exc.args[0]!r}."
                ) from exc

            if attacker_points <= 0:
                raise ValueError(
                    f"Line {line_number}: attacker actual_points must be > 0."
                )
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
            conservative_roi = minimum_profit / (attacker_points * 1_000.0)

            attacker_shares = _composition_point_shares(
                attacker_fleet,
                allowed_units=ATTACK_SHIPS,
            )
            dominant_defense = _dominant_defense(defender_fleet)
            tech_delta = (
                _combat_tech_total(attacker_tech)
                - _combat_tech_total(defender_tech)
            )

            # "Successful" is deliberately stricter than a simple combat win:
            # the attacker must win AND already be profitable without counting
            # debris. This is the subset most useful for seeding the optimizer.
            success = winner == "attacker" and minimum_profit > 0

            total += 1
            outcome_counts[winner] = outcome_counts.get(winner, 0) + 1
            minimum_profit_sum += minimum_profit
            maximum_profit_sum += maximum_profit
            tech_delta_sum += tech_delta

            if minimum_profit > 0:
                profitable_minimum += 1
            if maximum_profit > 0:
                profitable_maximum += 1
            if success:
                profitable_wins += 1

            ratio_index = _ratio_bin_index(power_ratio)
            ratio_battles[ratio_index] += 1
            if winner == "attacker":
                ratio_wins[ratio_index] += 1
            if success:
                ratio_successes[ratio_index] += 1
                ratio_success_min_profit_sum[ratio_index] += minimum_profit
                ratio_success_max_profit_sum[ratio_index] += maximum_profit
                ratio_success_roi_sum[ratio_index] += conservative_roi

            size_index = _size_bin_index(defender_points)
            size_battles[size_index] += 1
            if winner == "attacker":
                size_wins[size_index] += 1
            if success:
                size_successes[size_index] += 1
                size_success_ratio_sum[size_index] += power_ratio

            tech_index = _tech_delta_bucket_index(tech_delta)
            tech_battles[tech_index] += 1
            if success:
                tech_successes[tech_index] += 1

            profile = defense_profiles[dominant_defense]
            profile["battles"] += 1
            if winner == "attacker":
                profile["wins"] += 1

            for ship in ATTACK_SHIPS:
                if int(attacker_fleet.get(ship, 0)) > 0:
                    ship_present[ship] += 1

            if success:
                success_ratios.append(power_ratio)
                success_rois.append(conservative_roi)
                success_tech_delta_sum += tech_delta
                successful_records.append(
                    (conservative_roi, power_ratio, dict(attacker_shares))
                )

                profile["successes"] += 1
                profile["success_ratio_sum"] += power_ratio

                for ship in ATTACK_SHIPS:
                    share = float(attacker_shares.get(ship, 0.0))
                    ship_success_share_sum[ship] += share
                    profile["success_ship_share_sum"][ship] += share
                    if int(attacker_fleet.get(ship, 0)) > 0:
                        ship_success_present[ship] += 1

    if total == 0:
        raise ValueError(f"No battle records found in {source}.")

    success_count = profitable_wins

    # Top 25% of successful battles by conservative ROI. These are not
    # "optimal" attacks, but they are a useful high-quality seed signal.
    top_efficiency_share_sum = {ship: 0.0 for ship in ATTACK_SHIPS}
    top_efficiency_ratios: list[float] = []
    top_efficiency_count = 0
    roi_q75: float | None = None

    if successful_records:
        ordered_rois = sorted(record[0] for record in successful_records)
        q75_index = int(0.75 * (len(ordered_rois) - 1))
        roi_q75 = float(ordered_rois[q75_index])

        for roi, ratio, shares in successful_records:
            if roi + 1e-15 < roi_q75:
                continue
            top_efficiency_count += 1
            top_efficiency_ratios.append(ratio)
            for ship in ATTACK_SHIPS:
                top_efficiency_share_sum[ship] += float(shares.get(ship, 0.0))

    ship_rows = []
    for ship in ATTACK_SHIPS:
        present = ship_present[ship]
        present_success = ship_success_present[ship]
        absent = total - present
        absent_success = success_count - present_success

        present_success_rate = _percentage(present_success, present)
        absent_success_rate = _percentage(absent_success, absent)
        mean_success_share = (
            100.0 * ship_success_share_sum[ship] / success_count
            if success_count
            else 0.0
        )
        mean_top_share = (
            100.0 * top_efficiency_share_sum[ship] / top_efficiency_count
            if top_efficiency_count
            else 0.0
        )

        ship_rows.append(
            {
                "ship": ship,
                "present": present,
                "present_rate": _percentage(present, total),
                "success_when_present": present_success,
                "success_rate_when_present": present_success_rate,
                "success_rate_when_absent": absent_success_rate,
                "success_rate_uplift_pp": present_success_rate - absent_success_rate,
                "mean_point_share_success_pct": mean_success_share,
                "mean_point_share_top_efficiency_pct": mean_top_share,
            }
        )

    defense_rows = []
    for defense in REGULAR_DEFENSES:
        profile = defense_profiles[defense]
        successes = int(profile["successes"])
        mean_shares = {
            ship: (
                100.0 * float(profile["success_ship_share_sum"][ship]) / successes
                if successes
                else 0.0
            )
            for ship in ATTACK_SHIPS
        }
        defense_rows.append(
            {
                "defense": defense,
                "battles": int(profile["battles"]),
                "wins": int(profile["wins"]),
                "successes": successes,
                "success_rate": _percentage(successes, int(profile["battles"])),
                "mean_success_multiplier": (
                    float(profile["success_ratio_sum"]) / successes
                    if successes
                    else None
                ),
                "mean_success_ship_point_shares_pct": mean_shares,
            }
        )

    ratio_rows = []
    for index, label in enumerate(ratio_labels):
        successes = ratio_successes[index]
        ratio_rows.append(
            {
                "label": label,
                "battles": ratio_battles[index],
                "wins": ratio_wins[index],
                "win_rate": _percentage(ratio_wins[index], ratio_battles[index]),
                "successes": successes,
                "success_rate": _percentage(successes, ratio_battles[index]),
                "mean_min_profit_success": (
                    ratio_success_min_profit_sum[index] / successes
                    if successes
                    else None
                ),
                "mean_max_profit_success": (
                    ratio_success_max_profit_sum[index] / successes
                    if successes
                    else None
                ),
                "mean_conservative_roi_success": (
                    ratio_success_roi_sum[index] / successes
                    if successes
                    else None
                ),
            }
        )

    size_rows = []
    for index, label in enumerate(size_labels):
        successes = size_successes[index]
        size_rows.append(
            {
                "label": label,
                "battles": size_battles[index],
                "wins": size_wins[index],
                "win_rate": _percentage(size_wins[index], size_battles[index]),
                "successes": successes,
                "success_rate": _percentage(successes, size_battles[index]),
                "mean_success_multiplier": (
                    size_success_ratio_sum[index] / successes
                    if successes
                    else None
                ),
            }
        )

    tech_rows = []
    for index, label in enumerate(TECH_BUCKET_LABELS):
        tech_rows.append(
            {
                "label": label,
                "battles": tech_battles[index],
                "successes": tech_successes[index],
                "success_rate": _percentage(
                    tech_successes[index],
                    tech_battles[index],
                ),
            }
        )

    eligible_success_rate_bins = [
        row for row in ratio_rows if row["battles"] >= MIN_RATIO_SAMPLE
    ]
    best_success_ratio = (
        max(
            eligible_success_rate_bins,
            key=lambda row: (row["success_rate"], -ratio_rows.index(row)),
        )
        if eligible_success_rate_bins
        else None
    )

    eligible_roi_bins = [
        row
        for row in ratio_rows
        if row["successes"] >= MIN_RATIO_SUCCESS_SAMPLE
        and row["mean_conservative_roi_success"] is not None
    ]
    best_roi_ratio = (
        max(
            eligible_roi_bins,
            key=lambda row: float(row["mean_conservative_roi_success"]),
        )
        if eligible_roi_bins
        else None
    )

    return {
        "source": str(source),
        "total": total,
        "outcome_counts": outcome_counts,
        "profitable_minimum": profitable_minimum,
        "profitable_maximum": profitable_maximum,
        "profitable_wins": profitable_wins,
        "minimum_profit_sum": minimum_profit_sum,
        "maximum_profit_sum": maximum_profit_sum,
        "ratio_rows": ratio_rows,
        "size_rows": size_rows,
        "ship_rows": ship_rows,
        "defense_rows": defense_rows,
        "tech_rows": tech_rows,
        "success_summary": {
            "count": success_count,
            "mean_multiplier": _safe_mean(success_ratios),
            "median_multiplier": _safe_median(success_ratios),
            "mean_conservative_roi": _safe_mean(success_rois),
            "median_conservative_roi": _safe_median(success_rois),
            "top_efficiency_q75_roi_cutoff": roi_q75,
            "top_efficiency_count": top_efficiency_count,
            "top_efficiency_mean_multiplier": _safe_mean(top_efficiency_ratios),
            "mean_tech_delta": (
                success_tech_delta_sum / success_count if success_count else None
            ),
        },
        "dataset_mean_tech_delta": tech_delta_sum / total,
        "heuristics": {
            "best_success_ratio_bin": best_success_ratio,
            "best_roi_ratio_bin": best_roi_ratio,
            "minimum_ratio_bin_sample": MIN_RATIO_SAMPLE,
            "minimum_ratio_success_sample": MIN_RATIO_SUCCESS_SAMPLE,
        },
    }


def _svg_histogram(
    labels: Sequence[str],
    values: Sequence[int],
    *,
    aria_label: str,
    x_title: str,
    y_title: str,
) -> str:
    width = max(1160, 72 + len(values) * 50)
    height = 470
    left, right, top, bottom = 72, 24, 34, 125
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(1, max(values, default=0))
    slot = plot_width / max(1, len(values))
    bar_width = slot * 0.72

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(aria_label)}">'
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

    for index, (label, value) in enumerate(zip(labels, values)):
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
        f'text-anchor="middle" class="axis-title">{html.escape(x_title)}</text>'
    )
    parts.append(
        f'<text transform="translate(18,{top + plot_height/2:.1f}) rotate(-90)" '
        f'text-anchor="middle" class="axis-title">{html.escape(y_title)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _format_number(value: float | None, *, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}"


def _format_pct_fraction(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100.0 * value:.2f}%"


def _render_report(source: Path, stats: dict) -> str:
    total = stats["total"]
    outcomes = stats["outcome_counts"]
    attacker_wins = outcomes.get("attacker", 0)
    defender_wins = outcomes.get("defender", 0)
    draws = outcomes.get("draw", 0)
    successful = stats["profitable_wins"]

    ratio_rows = stats["ratio_rows"]
    size_rows = stats["size_rows"]
    ship_rows = stats["ship_rows"]
    defense_rows = stats["defense_rows"]
    tech_rows = stats["tech_rows"]
    success_summary = stats["success_summary"]
    heuristics = stats["heuristics"]

    ratio_histogram = _svg_histogram(
        [row["label"] for row in ratio_rows],
        [row["wins"] for row in ratio_rows],
        aria_label="Victorias del atacante por multiplicador de puntos",
        x_title="Multiplicador real de puntos atacante / defensor",
        y_title="Victorias del atacante",
    )
    size_histogram = _svg_histogram(
        [row["label"] for row in size_rows],
        [row["wins"] for row in size_rows],
        aria_label="Victorias del atacante por tamaño de defensa",
        x_title="Puntos reales de la defensa",
        y_title="Victorias del atacante",
    )

    ratio_table_rows = []
    for row in ratio_rows:
        roi = row["mean_conservative_roi_success"]
        ratio_table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{row['battles']:,}</td>"
            f"<td>{row['wins']:,}</td>"
            f"<td>{row['win_rate']:.1f}%</td>"
            f"<td>{row['successes']:,}</td>"
            f"<td>{row['success_rate']:.1f}%</td>"
            f"<td>{_format_pct_fraction(roi)}</td>"
            "</tr>"
        )

    size_table_rows = []
    for row in size_rows:
        multiplier = (
            f"{row['mean_success_multiplier']:.2f}x"
            if row["mean_success_multiplier"] is not None
            else "—"
        )
        size_table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{row['battles']:,}</td>"
            f"<td>{row['wins']:,}</td>"
            f"<td>{row['win_rate']:.1f}%</td>"
            f"<td>{row['successes']:,}</td>"
            f"<td>{row['success_rate']:.1f}%</td>"
            f"<td>{multiplier}</td>"
            "</tr>"
        )

    ship_table_rows = []
    for row in sorted(
        ship_rows,
        key=lambda item: item["mean_point_share_top_efficiency_pct"],
        reverse=True,
    ):
        ship_name = UNIT_SPECS[row["ship"]].name
        uplift = row["success_rate_uplift_pp"]
        uplift_class = "good" if uplift > 0 else ("bad" if uplift < 0 else "")
        ship_table_rows.append(
            "<tr>"
            f"<td>{html.escape(ship_name)}</td>"
            f"<td>{row['present_rate']:.1f}%</td>"
            f"<td>{row['success_rate_when_present']:.1f}%</td>"
            f"<td>{row['success_rate_when_absent']:.1f}%</td>"
            f'<td class="{uplift_class}">{uplift:+.1f} pp</td>'
            f"<td>{row['mean_point_share_success_pct']:.1f}%</td>"
            f"<td>{row['mean_point_share_top_efficiency_pct']:.1f}%</td>"
            "</tr>"
        )

    defense_summary_rows = []
    for row in defense_rows:
        defense_name = UNIT_SPECS[row["defense"]].name
        shares = row["mean_success_ship_point_shares_pct"]
        top_components = sorted(
            shares.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top_text = ", ".join(
            f"{UNIT_SPECS[ship].name} {share:.1f}%"
            for ship, share in top_components
        )
        multiplier = (
            f"{row['mean_success_multiplier']:.2f}x"
            if row["mean_success_multiplier"] is not None
            else "—"
        )
        defense_summary_rows.append(
            "<tr>"
            f"<td>{html.escape(defense_name)}</td>"
            f"<td>{row['battles']:,}</td>"
            f"<td>{row['successes']:,}</td>"
            f"<td>{row['success_rate']:.1f}%</td>"
            f"<td>{multiplier}</td>"
            f"<td>{html.escape(top_text) if top_text else '—'}</td>"
            "</tr>"
        )

    matrix_header = "".join(
        f"<th>{html.escape(UNIT_SPECS[ship].name)}</th>"
        for ship in ATTACK_SHIPS
    )
    matrix_rows = []
    for row in defense_rows:
        cells = "".join(
            f"<td>{row['mean_success_ship_point_shares_pct'][ship]:.1f}%</td>"
            for ship in ATTACK_SHIPS
        )
        matrix_rows.append(
            "<tr>"
            f"<td>{html.escape(UNIT_SPECS[row['defense']].name)}</td>"
            f"{cells}"
            "</tr>"
        )

    tech_table_rows = []
    for row in tech_rows:
        tech_table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{row['battles']:,}</td>"
            f"<td>{row['successes']:,}</td>"
            f"<td>{row['success_rate']:.1f}%</td>"
            "</tr>"
        )

    best_success = heuristics["best_success_ratio_bin"]
    best_roi = heuristics["best_roi_ratio_bin"]
    best_success_text = (
        f"{best_success['label']} · {best_success['success_rate']:.1f}%"
        if best_success
        else "Sin muestra suficiente"
    )
    best_roi_text = (
        f"{best_roi['label']} · "
        f"{_format_pct_fraction(best_roi['mean_conservative_roi_success'])}"
        if best_roi
        else "Sin muestra suficiente"
    )

    avg_min = stats["minimum_profit_sum"] / total
    avg_max = stats["maximum_profit_sum"] / total
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    mean_success_multiplier = success_summary["mean_multiplier"]
    median_success_multiplier = success_summary["median_multiplier"]
    top_multiplier = success_summary["top_efficiency_mean_multiplier"]
    median_roi = success_summary["median_conservative_roi"]

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OGIA · Análisis de batallas</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--text:#17202a;--muted:#65717e;--accent:#2962ff;--border:#dfe5eb;--good:#137333;--bad:#b3261e}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text)}}
main{{max-width:1380px;margin:auto;padding:34px 24px 56px}}h1{{margin:0 0 6px;font-size:32px}}h2{{margin:0 0 18px;font-size:21px}}h3{{margin:22px 0 10px;font-size:16px}}
.meta,.note{{color:var(--muted);line-height:1.55}}.meta{{margin-bottom:26px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:14px;margin-bottom:22px}}
.card,.panel{{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:0 1px 2px #00000009}}.card{{padding:18px}}.label{{color:var(--muted);font-size:13px;margin-bottom:7px}}.number{{font-size:28px;font-weight:700}}.detail{{color:var(--muted);font-size:13px;margin-top:5px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}
.panel{{padding:22px;margin-top:18px;overflow:auto}}svg{{display:block;width:100%;min-width:760px;height:auto}}.grid{{stroke:#e8edf2;stroke-width:1}}.bar{{fill:var(--accent)}}.axis{{fill:#5d6975;font-size:12px}}.axis-title{{fill:#3e4852;font-size:13px;font-weight:600}}.value{{fill:#2c3640;font-size:11px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}}.note{{font-size:13px;margin-top:12px}}code{{background:#eef1f4;border-radius:4px;padding:2px 5px}}
.matrix{{font-size:12px}}.matrix th,.matrix td{{padding:8px 9px}}
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
<div class="card"><div class="label">Ataques exitosos</div><div class="number good">{successful:,}</div><div class="detail">{_percentage(successful,total):.1f}% · victoria + beneficio mínimo &gt; 0</div></div>
<div class="card"><div class="label">Beneficio máximo &gt; 0</div><div class="number">{stats['profitable_maximum']:,}</div><div class="detail">{_percentage(stats['profitable_maximum'],total):.1f}% del total</div></div>
</div>

<section class="panel">
<h2>Señales para inicializar el optimizador</h2>
<div class="cards">
<div class="card"><div class="label">Multiplicador medio · ataques exitosos</div><div class="number">{_format_number(mean_success_multiplier, decimals=2)}x</div><div class="detail">Mediana: {_format_number(median_success_multiplier, decimals=2)}x</div></div>
<div class="card"><div class="label">Mejor rango por tasa de éxito</div><div class="number" style="font-size:21px">{html.escape(best_success_text)}</div><div class="detail">Sólo bins con ≥ {MIN_RATIO_SAMPLE} batallas</div></div>
<div class="card"><div class="label">Mejor rango por ROI conservador</div><div class="number" style="font-size:21px">{html.escape(best_roi_text)}</div><div class="detail">Sólo bins con ≥ {MIN_RATIO_SUCCESS_SAMPLE} ataques exitosos</div></div>
<div class="card"><div class="label">Top 25% eficiente · multiplicador medio</div><div class="number">{_format_number(top_multiplier, decimals=2)}x</div><div class="detail">Top por beneficio mínimo / coste en puntos del ataque</div></div>
</div>
<div class="note"><strong>Importante:</strong> estas son señales para construir una población inicial, no ataques óptimos. Las batallas fueron generadas aleatoriamente y hay variables de confusión como tecnologías, tamaño de defensa y composición. El algoritmo evolutivo seguirá siendo quien busque el óptimo.</div>
</section>

<section class="panel">
<h2>Victorias por superioridad de puntos del atacante</h2>
{ratio_histogram}
<div class="note">El multiplicador usa <strong>puntos reales del atacante / puntos reales del defensor</strong>. El criterio “éxito” de las tablas es más exigente que ganar: exige <code>winner == attacker</code> y <code>minimum_profit_total &gt; 0</code>.</div>
</section>

<section class="panel">
<h2>Detalle por multiplicador de puntos</h2>
<table>
<thead><tr><th>Atacante / defensor</th><th>Batallas</th><th>Victorias</th><th>Win rate</th><th>Éxitos</th><th>Tasa de éxito</th><th>ROI conservador medio*</th></tr></thead>
<tbody>{''.join(ratio_table_rows)}</tbody>
</table>
<div class="note">* ROI conservador = <code>minimum_profit_total / coste total aproximado del ataque</code>, donde un punto OGame equivale a 1.000 recursos. Se calcula únicamente sobre ataques exitosos.</div>
</section>

<section class="panel">
<h2>Victorias por tamaño de la defensa</h2>
{size_histogram}
</section>

<section class="panel">
<h2>Detalle por tamaño de defensa</h2>
<table>
<thead><tr><th>Puntos defensa</th><th>Batallas</th><th>Victorias</th><th>Win rate</th><th>Éxitos</th><th>Tasa de éxito</th><th>Multiplicador medio en éxitos</th></tr></thead>
<tbody>{''.join(size_table_rows)}</tbody>
</table>
<div class="note">Esta sección permite comprobar si el multiplicador que funciona cambia con la escala de la defensa, algo importante antes de fijar límites del gen <code>points_multiplier</code>.</div>
</section>

<section class="panel">
<h2>Qué naves aparecen en los ataques que funcionan</h2>
<table>
<thead><tr><th>Nave</th><th>Presencia global</th><th>Éxito si está presente</th><th>Éxito si está ausente</th><th>Diferencia</th><th>% puntos medio en éxitos</th><th>% puntos medio · top 25% eficiente</th></tr></thead>
<tbody>{''.join(ship_table_rows)}</tbody>
</table>
<div class="note">La columna “Diferencia” es descriptiva y <strong>no implica causalidad</strong>: una nave puede aparecer más en ataques grandes o contra determinados tipos de defensa. Para el punto de inicio, la señal más interesante es la distribución de puntos entre naves en los ataques exitosos y, especialmente, en su cuartil más eficiente.</div>
</section>

<section class="panel">
<h2>Composición del ataque según la defensa dominante</h2>
<table>
<thead><tr><th>Defensa dominante*</th><th>Batallas</th><th>Éxitos</th><th>Tasa de éxito</th><th>Multiplicador medio</th><th>3 componentes principales del ataque exitoso</th></tr></thead>
<tbody>{''.join(defense_summary_rows)}</tbody>
</table>
<div class="note">* “Dominante” significa el tipo de defensa regular que concentra más puntos. Las cúpulas se excluyen para no convertir defensas pequeñas en una categoría artificial. Esta tabla es la señal más directa para crear inicializadores condicionados por la composición defensiva.</div>

<h3>Matriz completa: % medio de puntos del ataque en batallas exitosas</h3>
<table class="matrix">
<thead><tr><th>Defensa dominante</th>{matrix_header}</tr></thead>
<tbody>{''.join(matrix_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Efecto de la diferencia tecnológica</h2>
<table>
<thead><tr><th>Σ tecnologías atacante − defensor</th><th>Batallas</th><th>Éxitos</th><th>Tasa de éxito</th></tr></thead>
<tbody>{''.join(tech_table_rows)}</tbody>
</table>
<div class="note">Se suma armas + escudos + blindaje de cada lado. Diferencia media del dataset: {stats['dataset_mean_tech_delta']:+.2f}; diferencia media entre ataques exitosos: {_format_number(success_summary['mean_tech_delta'], decimals=2)}. Si esta tabla muestra un efecto fuerte, habrá que controlar o fijar las tecnologías cuando usemos estas batallas para generar seeds.</div>
</section>

<section class="panel">
<h2>Beneficio y eficiencia</h2>
<table>
<tbody>
<tr><th>Beneficio mínimo medio por batalla</th><td>{avg_min:,.0f}</td></tr>
<tr><th>Beneficio máximo medio por batalla</th><td>{avg_max:,.0f}</td></tr>
<tr><th>ROI conservador mediano · ataques exitosos</th><td>{_format_pct_fraction(median_roi)}</td></tr>
<tr><th>Corte ROI del top 25% eficiente</th><td>{_format_pct_fraction(success_summary['top_efficiency_q75_roi_cutoff'])}</td></tr>
<tr><th>Ataques dentro del top 25% eficiente</th><td>{success_summary['top_efficiency_count']:,}</td></tr>
</tbody>
</table>
<div class="note"><strong>Beneficio mínimo</strong> = loot − coste de naves atacantes destruidas. <strong>Beneficio máximo</strong> = beneficio mínimo + todos los escombros generados. Para orientar el optimizador damos prioridad al beneficio mínimo y a su ROI, porque no presupone que el atacante recoja todos los escombros.</div>
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

    # Keep the same analysis in a machine-readable companion file so the future
    # PopulationInitializer can consume these priors without scraping HTML.
    json_output = output.with_suffix(".json")
    json_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    attacker_wins = stats["outcome_counts"].get("attacker", 0)
    print(f"Battles analyzed: {stats['total']:,}")
    print(
        "Successful attacks (attacker win + minimum profit > 0): "
        f"{stats['profitable_wins']:,} "
        f"({_percentage(stats['profitable_wins'], stats['total']):.1f}%)"
    )
    print(
        f"Attacker wins: {attacker_wins:,} "
        f"({_percentage(attacker_wins, stats['total']):.1f}%)"
    )
    print(f"HTML report: {output}")
    print(f"JSON summary: {json_output}")
    return 0
