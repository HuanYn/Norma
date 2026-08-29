from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Keep Matplotlib's PDF CreationDate deterministic across replays.
os.environ["SOURCE_DATE_EPOCH"] = "0"

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, FIG_DIR, FONT_SIZE, save_fig


PROJECT_ROOT = FIG_DIR.parent


def _public_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


METHODS = (
    ("zero_feedback_cosine", "Cosine (zero feedback)", COLORS[0], "--", "o"),
    ("random_pair_contextual", "Contextual, random pairs", COLORS[1], "-", "s"),
    (
        "predictive_entropy_contextual",
        "Contextual, predictive entropy",
        COLORS[2],
        "-",
        "^",
    ),
)
PANELS = (
    ("heldout_pair_expected_log_loss", "Held-out pair log loss\n(lower is better)"),
    ("heldout_pair_order_accuracy", "Held-out pair accuracy\n(higher is better)"),
    (
        "constrained_set_regret_per_photo",
        "Constrained set regret / photo\n(lower is better)",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=FIG_DIR / "contextual_preference_controlled_20260828.json",
    )
    parser.add_argument("--name", default="fig3_contextual_preference_learning")
    return parser.parse_args()


def _metric_arrays(
    summary: dict[str, Any], method: str, budgets: list[int], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]]]:
    records = [summary[method][str(budget)][metric] for budget in budgets]
    means = np.asarray([record["mean"] for record in records], dtype=np.float64)
    lows = np.asarray(
        [record["bootstrap_95_percentile_ci"][0] for record in records],
        dtype=np.float64,
    )
    highs = np.asarray(
        [record["bootstrap_95_percentile_ci"][1] for record in records],
        dtype=np.float64,
    )
    raw = [record["raw_seed_means"] for record in records]
    return means, lows, highs, raw


def _set_limits(ax: plt.Axes, values: list[float], *, metric: str) -> None:
    low = min(values)
    high = max(values)
    span = max(high - low, 1e-3)
    lower = low - 0.09 * span
    upper = high + 0.11 * span
    if metric == "constrained_set_regret_per_photo":
        lower = 0.0
    if metric == "heldout_pair_order_accuracy":
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    ax.set_ylim(lower, upper)


def _format_interval(record: dict[str, Any], *, digits: int = 3) -> str:
    low, high = record["bootstrap_95_percentile_ci"]
    return f"{record['mean']:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _write_table(payload: dict[str, Any], budgets: list[int]) -> None:
    summary = payload["summary"]
    final_budget = str(max(budgets))
    labels = {
        "zero_feedback_cosine": "Cosine (0 feedback)",
        "random_pair_contextual": "Contextual, random pairs",
        "predictive_entropy_contextual": "Contextual, predictive entropy",
    }
    rows = []
    for method, label in labels.items():
        values = summary[method][final_budget]
        rows.append(
            " & ".join(
                (
                    _latex_escape(label),
                    _format_interval(values["heldout_pair_expected_log_loss"]),
                    _format_interval(values["heldout_pair_order_accuracy"]),
                    _format_interval(values["constrained_set_regret_per_photo"]),
                )
            )
            + r" \\"
        )
    table = "\n".join(
        (
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            (
                r"\caption{Controlled semi-synthetic preference results after 60 "
                r"feedback comparisons. Values are means and 95\% percentile-bootstrap "
                r"intervals across 10 split/choice seeds after averaging three fixed "
                r"simulated profiles within each seed. This is not human-preference evidence.}"
            ),
            r"\label{tab:controlled-contextual-preference}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Method & Pair log loss $\downarrow$ & Pair accuracy $\uparrow$ & Set regret/photo $\downarrow$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        )
    )
    (FIG_DIR / "TABLE_contextual_preference_controlled.tex").write_text(
        table, encoding="utf-8", newline="\n"
    )


def _write_latex_include(name: str) -> None:
    snippet = "\n".join(
        (
            r"\begin{figure*}[t]",
            r"  \centering",
            rf"  \includegraphics[width=0.95\textwidth]{{figures/{name}.pdf}}",
            (
                r"  \caption{Learning curves for the 67D contextual preference adapter "
                r"on 70 public Wikimedia images under three declared category-utility "
                r"simulators. Each replicate uses a category-stratified exact-file-disjoint "
                r"42/28 train/test split. Lines are means across 10 split/choice seeds "
                r"after averaging three fixed simulated profiles per seed; shaded regions "
                r"are 95\% percentile-bootstrap intervals over seed means and faint points "
                r"are the raw seed means. At 60 feedback comparisons, all paired "
                r"predictive-entropy-versus-random intervals cross zero, so the curves do "
                r"not establish superiority of entropy acquisition. The experiment uses "
                r"no human preferences and does not establish population generalization.}"
            ),
            r"  \label{fig:controlled-contextual-preference}",
            r"\end{figure*}",
            "",
        )
    )
    (FIG_DIR / "contextual_preference_latex_include.tex").write_text(
        snippet, encoding="utf-8", newline="\n"
    )


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload["claim_boundary"]["is_real_human_preference_data"] is not False:
        raise ValueError("figure input must retain the non-human claim boundary")
    budgets = [int(value) for value in payload["assumptions"]["feedback_budgets"]]
    summary = payload["summary"]
    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharex=True)
    panel_letters = ("(a)", "(b)", "(c)")
    all_panel_values: dict[str, list[float]] = {metric: [] for metric, _ in PANELS}

    for method_index, (method, label, color, linestyle, marker) in enumerate(METHODS):
        for ax, (metric, ylabel) in zip(axes, PANELS, strict=True):
            means, lows, highs, raw = _metric_arrays(summary, method, budgets, metric)
            ax.fill_between(budgets, lows, highs, color=color, alpha=0.12, linewidth=0)
            ax.plot(
                budgets,
                means,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.0,
                linewidth=1.55,
                zorder=3,
            )
            for budget_index, (budget, seed_values) in enumerate(
                zip(budgets, raw, strict=True)
            ):
                offsets = np.linspace(-0.38, 0.38, len(seed_values))
                offsets += (method_index - 1) * 0.05
                ax.scatter(
                    budget + offsets,
                    seed_values,
                    color=color,
                    marker=marker,
                    s=7,
                    alpha=0.24,
                    linewidths=0,
                    zorder=2,
                )
                all_panel_values[metric].extend(float(value) for value in seed_values)
            all_panel_values[metric].extend(lows.tolist())
            all_panel_values[metric].extend(highs.tolist())
            ax.set_ylabel(ylabel)
            ax.set_xticks(budgets)
            ax.set_xlabel("Feedback comparisons")
            ax.yaxis.grid(True, color="#d8d8d8", linewidth=0.45, alpha=0.65)
            ax.set_axisbelow(True)

    for ax, (metric, _), letter in zip(axes, PANELS, panel_letters, strict=True):
        _set_limits(ax, all_panel_values[metric], metric=metric)
        ax.text(
            -0.11,
            1.03,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.2,
    )
    figure.text(
        0.995,
        0.005,
        "raw points: n=10 seed means; 3 fixed simulated profiles averaged per seed",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    figure.subplots_adjust(left=0.085, right=0.995, bottom=0.24, top=0.79, wspace=0.36)
    save_fig(figure, args.name)
    plt.close(figure)
    _write_table(payload, budgets)
    _write_latex_include(args.name)
    print(
        json.dumps(
            {
                "path_base": "repository-root",
                "pdf": _public_path(FIG_DIR / f"{args.name}.pdf"),
                "png": _public_path(FIG_DIR / f"{args.name}.png"),
                "table": _public_path(
                    FIG_DIR / "TABLE_contextual_preference_controlled.tex"
                ),
                "n_seed_means": 10,
                "profiles_per_seed": 3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
