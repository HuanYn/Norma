from __future__ import annotations

import json
import math
import os
from typing import Any


os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, FIG_DIR, save_fig


DATA_PATH = FIG_DIR / "wikimedia_fixture_drift_audit_20260829.json"
OUTPUT_NAME = "fig6_wikimedia_fixture_drift"


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_and_validate() -> dict[str, Any]:
    data = json.loads(
        DATA_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if (
        not isinstance(data, dict)
        or data.get("schema") != "norma-wikimedia-fixture-drift-audit-v1"
    ):
        raise ValueError("unexpected Wikimedia fixture audit schema")
    counts = data["first_full_pass"]
    if (
        sum(counts["all_72"].values()) != 72
        or sum(counts["experiment_70"].values()) != 70
    ):
        raise ValueError("fixture drift counts do not close")
    metadata_files = data["metadata_only_drift"]["files"]
    pixel_rows = data["pixel_drift"]
    if len(metadata_files) != 10 or len(set(metadata_files)) != 10:
        raise ValueError("metadata-only drift records are incomplete")
    if len(pixel_rows) != 4 or len({row["file"] for row in pixel_rows}) != 4:
        raise ValueError("pixel-drift records are incomplete")
    for row in pixel_rows:
        if row["historical_raw_sha256"] == row["current_raw_sha256"]:
            raise ValueError("pixel-drift row has identical raw hashes")
        if row["historical_pixel_sha256"] == row["current_pixel_sha256"]:
            raise ValueError("pixel-drift row has identical pixel hashes")
        for key in ("mae", "rmse", "psnr_db", "changed_channel_pct"):
            value = float(row[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid pixel drift metric: {row['file']} {key}")
    archive = data["oldimage_archive_audit"]
    if (
        archive["raw_drift_files_queried"] != 14
        or archive["historical_raw_recovered"] != 0
        or archive["historical_pixels_recovered_for_pixel_drift"] != 0
    ):
        raise ValueError("archive recovery conclusion drifted")
    return data


def _draw(data: dict[str, Any]) -> None:
    counts = data["first_full_pass"]
    all_values = counts["all_72"]
    experiment_values = counts["experiment_70"]
    categories = ["raw_exact", "raw_drift_pixel_exact", "pixel_drift"]
    labels = ["Raw bytes exact", "Raw drift · RGB exact", "RGB pixels drift"]
    colors = [COLORS[0], COLORS[1], COLORS[2]]

    fig = plt.figure(figsize=(7.35, 3.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.25), wspace=0.18)

    count_ax = fig.add_subplot(grid[0, 0])
    x = np.arange(2)
    bottoms = np.zeros(2, dtype=float)
    for category, label, color in zip(categories, labels, colors, strict=True):
        values = np.asarray(
            [all_values[category], experiment_values[category]], dtype=float
        )
        bars = count_ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=label,
        )
        for bar, value, bottom in zip(bars, values, bottoms, strict=True):
            if value >= 4:
                count_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=9,
                )
        bottoms += values
    count_ax.set_ylim(0, 78)
    count_ax.set_xticks(x, ["Manifest\nall 72", "Experiment\nused 70"])
    count_ax.set_ylabel("Images in first full URL audit")
    count_ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    count_ax.text(-0.18, 1.02, "(a)", transform=count_ax.transAxes, fontweight="bold")
    count_ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.04, 1.005),
        ncol=1,
        handlelength=1.1,
        handletextpad=0.45,
    )

    drift_ax = fig.add_subplot(grid[0, 1])
    rows = data["pixel_drift"]
    ids = [row["file"][:3] for row in rows]
    psnr = [float(row["psnr_db"]) for row in rows]
    positions = np.arange(len(rows))
    drift_ax.scatter(positions, psnr, s=58, color=COLORS[2], zorder=3)
    for position, row in zip(positions, rows, strict=True):
        drift_ax.text(
            position,
            float(row["psnr_db"]) + 0.72,
            f"MAE {float(row['mae']):.2f}\n{float(row['changed_channel_pct']):.1f}% changed",
            ha="center",
            va="bottom",
            fontsize=7.5,
            linespacing=1.15,
        )
    drift_ax.set_xlim(-0.55, len(rows) - 0.45)
    drift_ax.set_ylim(28, 47)
    drift_ax.set_xticks(positions, [f"#{value}" for value in ids])
    drift_ax.set_ylabel("Historical vs current decoded RGB PSNR (dB)")
    drift_ax.set_xlabel("Pixel-drift image ID (higher PSNR = closer pixels)")
    drift_ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    drift_ax.text(0.0, 1.02, "(b)", transform=drift_ax.transAxes, fontweight="bold")
    drift_ax.text(
        0.98,
        0.05,
        "Wikimedia oldimage/archive recovery: 0 / 4",
        transform=drift_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#B8B8B8"},
    )

    save_fig(fig, OUTPUT_NAME)
    plt.close(fig)


def _write_latex(data: dict[str, Any]) -> None:
    all_counts = data["first_full_pass"]["all_72"]
    used_counts = data["first_full_pass"]["experiment_70"]
    table = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Reconstruction audit for the mutable Wikimedia thumbnail URLs used by the historical preference-study fixture. Percentages describe one sequential full pass on 2026-08-29; they are not availability guarantees.}}
\\label{{tab:wikimedia-fixture-drift}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
Scope & Raw exact & Raw drift / RGB exact & RGB drift & Archive recovery \\\\
\\midrule
All manifest images ($n=72$) & {all_counts["raw_exact"]} ({all_counts["raw_exact"] / 72 * 100:.2f}\\%) & {all_counts["raw_drift_pixel_exact"]} ({all_counts["raw_drift_pixel_exact"] / 72 * 100:.2f}\\%) & {all_counts["pixel_drift"]} ({all_counts["pixel_drift"] / 72 * 100:.2f}\\%) & raw 0 / 14; pixels 0 / 4 \\\\
Experiment-used images ($n=70$) & {used_counts["raw_exact"]} ({used_counts["raw_exact"] / 70 * 100:.2f}\\%) & {used_counts["raw_drift_pixel_exact"]} ({used_counts["raw_drift_pixel_exact"] / 70 * 100:.2f}\\%) & {used_counts["pixel_drift"]} ({used_counts["pixel_drift"] / 70 * 100:.2f}\\%) & raw 0 / 12; pixels 0 / 4 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (FIG_DIR / "TABLE_wikimedia_fixture_drift.tex").write_text(
        table, encoding="utf-8", newline="\n"
    )
    include = """% Generated by gen_fig6_wikimedia_fixture_drift.py
\\begin{figure*}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{figures/fig6_wikimedia_fixture_drift.pdf}
  \\caption{Mutable-upstream audit of the historical 72-image Wikimedia fixture. (a) Only 58 responses remained byte-identical; ten changed only JPEG metadata while preserving decoded RGB, and four changed decoded pixels. (b) The four pixel-drift cases differ measurably despite unchanged dimensions. No historical raw or pixel-drift input was recovered through Wikimedia oldimage/archive. The existing experiment keeps its historical raw-SHA provenance and the downloader fails closed; exact fresh-clone replay requires a separately licensed content-addressed archive or a new experiment ID. The public artifact freezes historical pins and the audit summary, not transient current-response binaries or the complete per-file archive transcript; current-side differences are contemporaneous audit-record observations rather than independently offline-replayable measurements.}
  \\label{fig:wikimedia-fixture-drift}
\\end{figure*}
\\input{figures/TABLE_wikimedia_fixture_drift.tex}
"""
    (FIG_DIR / "wikimedia_fixture_drift_latex_include.tex").write_text(
        include, encoding="utf-8", newline="\n"
    )


def main() -> None:
    data = _load_and_validate()
    _draw(data)
    _write_latex(data)


if __name__ == "__main__":
    main()
