from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper_plot_style import COLORS, FIG_DIR, save_fig


DATA_PATH = Path(__file__).with_name("windows_numeric_runtime_20260829.json")
OUTPUT_NAME = "figS1_windows_numeric_runtime"


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    assert data["collision"]["returncode"] != 0
    assert data["safe_runtime"]["returncode"] == 0
    assert data["unsafe_workaround_used"] is False

    baseline = data["numpy_default_without_torch"]
    safe = data["sequential_with_torch"]
    workloads = ["32×512 matvec\n20,000 calls", "67×67 solve\n300 calls"]
    baseline_values = [baseline["matvec_seconds"], baseline["solve_seconds"]]
    safe_values = [safe["matvec_seconds"], safe["solve_seconds"]]
    baseline_medians = [_median(values) for values in baseline_values]
    safe_medians = [_median(values) for values in safe_values]

    fig = plt.figure(figsize=(7.35, 3.05), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(0.88, 1.65), wspace=0.22)

    status_ax = fig.add_subplot(grid[0, 0])
    status_ax.set_xlim(0, 1)
    status_ax.set_ylim(-0.55, 1.55)
    status_ax.axis("off")
    status_ax.text(-0.05, 1.04, "(a)", transform=status_ax.transAxes, fontweight="bold")
    status_ax.scatter([0.12], [1], s=120, marker="X", color=COLORS[2], zorder=3)
    status_ax.text(
        0.24,
        1,
        "Default threaded MKL + Torch\nexit 3 · OpenMP Error #15",
        va="center",
        linespacing=1.35,
    )
    status_ax.scatter([0.12], [0], s=125, marker="o", color=COLORS[0], zorder=3)
    status_ax.text(
        0.24,
        0,
        "Sequential MKL contract + Torch\nexit 0 · timing workloads completed",
        va="center",
        linespacing=1.35,
    )
    status_ax.plot([0.12, 0.12], [0.16, 0.84], color="#B8B8B8", lw=1.1, zorder=1)

    timing_ax = fig.add_subplot(grid[0, 1])
    x = np.arange(len(workloads), dtype=float)
    width = 0.31
    baseline_color = "#777777"
    safe_color = COLORS[0]
    jitter = np.linspace(-0.055, 0.055, data["iterations"]["repeats"])
    for group, values in enumerate(baseline_values):
        center = x[group] - width / 2
        timing_ax.scatter(
            np.full(len(values), center) + jitter,
            values,
            s=27,
            facecolors=baseline_color,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        timing_ax.plot(
            [center - 0.09, center + 0.09],
            [baseline_medians[group], baseline_medians[group]],
            color=baseline_color,
            linewidth=2.2,
            zorder=4,
        )
    for group, values in enumerate(safe_values):
        center = x[group] + width / 2
        timing_ax.scatter(
            np.full(len(values), center) + jitter,
            values,
            s=27,
            facecolors=safe_color,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        timing_ax.plot(
            [center - 0.09, center + 0.09],
            [safe_medians[group], safe_medians[group]],
            color=safe_color,
            linewidth=2.2,
            zorder=4,
        )

    timing_ax.set_yscale("log")
    timing_ax.set_ylabel("Wall time for fixed workload (s, log scale)")
    timing_ax.set_xticks(x, workloads)
    timing_ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    timing_ax.text(-0.12, 1.04, "(b)", transform=timing_ax.transAxes, fontweight="bold")
    speedup = baseline_medians[0] / safe_medians[0]
    solve_reduction = 1.0 - safe_medians[1] / baseline_medians[1]
    timing_ax.text(
        x[0],
        max(baseline_values[0]) * 1.27,
        f"{speedup:.2f}× speedup",
        ha="center",
        fontsize=9,
    )
    timing_ax.text(
        x[1],
        max(baseline_values[1] + safe_values[1]) * 1.29,
        f"{solve_reduction * 100:.2f}% lower",
        ha="center",
        fontsize=9,
    )
    timing_ax.legend(
        handles=[
            Line2D(
                [0], [0], color=baseline_color, lw=6, label="Default MKL (NumPy only)"
            ),
            Line2D([0], [0], color=safe_color, lw=6, label="Sequential MKL + Torch"),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.005),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.2,
    )

    save_fig(fig, OUTPUT_NAME)
    plt.close(fig)

    table = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Windows numeric-runtime compatibility profile. Times are five-repeat medians for fixed workloads on one machine. The default arm uses NumPy without Torch because default MKL plus Torch aborts; the safe arm uses sequential MKL with Torch loaded.}}
\\label{{tab:windows-numeric-runtime}}
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Workload & Default MKL & Sequential + Torch & Change \\\\
\\midrule
32$\\times$512 matvec ($20{{,}}000$ calls) & {baseline_medians[0]:.4g} s & {safe_medians[0]:.4g} s & {speedup:.2f}$\\times$ speedup \\\\
67$\\times$67 solve ($300$ calls) & {baseline_medians[1]:.4g} s & {safe_medians[1]:.4g} s & {solve_reduction * 100:.2f}\\% lower \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (FIG_DIR / "TABLE_windows_numeric_runtime.tex").write_text(
        table, encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
