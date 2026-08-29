from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

# Keep Matplotlib's PDF CreationDate deterministic across replays.
os.environ["SOURCE_DATE_EPOCH"] = "0"

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import FIG_DIR, FONT_SIZE, save_fig


PROJECT_ROOT = FIG_DIR.parent


def _public_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


EXPECTED_EXPERIMENT = "capu-pdrr-controlled-wikimedia-v1-20260829"
EXPECTED_SOURCE_SHA256 = (
    "16bfdde5c61fc6dca02d19676a441fd37b265effb9bd0631dc4947ad5bb2cdbc"
)
METHODS = (
    ("zero_feedback_cosine", "Cosine (0 FB)", "#666666", "--", "o"),
    ("random_pair_contextual", "Random", "#0072B2", ":", "s"),
    ("predictive_entropy_contextual", "Entropy", "#D55E00", "-.", "^"),
    ("pdrr_mc_contextual", "CAPU-PDRR-MC", "#009E73", "-", "D"),
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
        default=FIG_DIR / "pdrr_acquisition_controlled_20260829.json",
    )
    parser.add_argument("--name", default="fig4_pdrr_acquisition_learning")
    return parser.parse_args()


def _validate_payload(payload: dict[str, Any]) -> list[int]:
    if payload.get("experiment_id") != EXPECTED_EXPERIMENT:
        raise ValueError("unexpected experiment JSON")
    if payload.get("run_kind") != "full":
        raise ValueError("Figure 4 requires the locked full protocol")
    source_audit = payload["provenance"]["source_audit"]
    if source_audit["source_sha256_before_read"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("embedding/split source SHA-256 drifted")
    if source_audit["public_data"]["image_count"] != 70:
        raise ValueError("Figure 4 requires all 70 validated public images")
    if source_audit["public_data"]["all_file_sha256_match"] is not True:
        raise ValueError("public image SHA-256 audit did not pass")
    if (
        source_audit["numeric_recomputation"]["max_contextual_feature_abs_error"]
        > 1e-12
    ):
        raise ValueError("67D feature recomputation audit did not pass")
    budgets = [int(value) for value in payload["protocol"]["feedback_budgets"]]
    if budgets != [0, 10, 30, 60]:
        raise ValueError(f"unexpected feedback budgets: {budgets}")
    if len(payload["runs"]) != 120:
        raise ValueError("full protocol must contain 120 method/profile/seed runs")
    if set(payload["paired_budget_comparisons"]) != {"10", "30", "60"}:
        raise ValueError("paired bootstrap comparisons are incomplete")
    for method, *_ in METHODS:
        for budget in budgets:
            for metric, _ in PANELS:
                record = payload["summary"][method][str(budget)][metric]
                raw = np.asarray(record["raw_seed_means"], dtype=np.float64)
                if raw.shape != (10,) or not np.all(np.isfinite(raw)):
                    raise ValueError(
                        f"invalid seed means for {method}/{budget}/{metric}"
                    )
                if not math.isclose(
                    float(np.mean(raw)),
                    float(record["mean"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("summary mean does not match raw seed means")
                low, high = record["bootstrap_95_percentile_ci"]
                if not low <= record["mean"] <= high:
                    raise ValueError(
                        "bootstrap interval does not contain the point mean"
                    )
    return budgets


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


def _format_delta(record: dict[str, Any], *, accuracy: bool = False) -> str:
    scale = 100.0 if accuracy else 1.0
    digits = 2 if accuracy else 3
    mean = scale * float(record["paired_mean_improvement"])
    low, high = (
        scale * float(value) for value in record["paired_bootstrap_95_percentile_ci"]
    )
    value = f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"
    if record["supports_challenger_better"]:
        return rf"\textbf{{{value}}}"
    return value


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _write_table(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    method_labels = {
        "zero_feedback_cosine": "Cosine (zero feedback)",
        "random_pair_contextual": "Contextual, random pairs",
        "predictive_entropy_contextual": "Contextual, predictive entropy",
        "pdrr_mc_contextual": "Contextual, CAPU-PDRR-MC",
    }
    mean_rows: list[str] = []
    for budget in (10, 30, 60):
        for method, *_ in METHODS:
            values = summary[method][str(budget)]
            mean_rows.append(
                " & ".join(
                    (
                        _latex_escape(method_labels[method]),
                        str(budget),
                        _format_interval(values["heldout_pair_expected_log_loss"]),
                        _format_interval(values["heldout_pair_order_accuracy"]),
                        _format_interval(values["constrained_set_regret_per_photo"]),
                    )
                )
                + r" \\"
            )
        if budget != 60:
            mean_rows.append(r"\addlinespace[2pt]")

    comparison_labels = {
        "pdrr_vs_random": "PDRR improvement vs random",
        "pdrr_vs_entropy": "PDRR improvement vs entropy",
    }
    paired_rows: list[str] = []
    for budget in (10, 30, 60):
        comparisons = payload["paired_budget_comparisons"][str(budget)]
        for comparison in ("pdrr_vs_random", "pdrr_vs_entropy"):
            metrics = comparisons[comparison]["metrics"]
            paired_rows.append(
                " & ".join(
                    (
                        comparison_labels[comparison],
                        str(budget),
                        _format_delta(metrics["heldout_pair_expected_log_loss"]),
                        _format_delta(
                            metrics["heldout_pair_order_accuracy"], accuracy=True
                        ),
                        _format_delta(metrics["constrained_set_regret_per_photo"]),
                    )
                )
                + r" \\"
            )
        if budget != 60:
            paired_rows.append(r"\addlinespace[2pt]")

    table = "\n".join(
        (
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            (
                r"\caption{Controlled semi-synthetic acquisition comparison. Panel A "
                r"reports means and 95\% percentile-bootstrap intervals across 10 "
                r"split/choice seeds after averaging three fixed simulated profiles "
                r"within seed. Panel B reports paired improvements where positive means "
                r"CAPU-PDRR-MC is better; accuracy differences are percentage points. "
                r"Bold paired intervals exclude zero in the favorable direction. This "
                r"table is not evidence from human preferences.}"
            ),
            r"\label{tab:controlled-pdrr-acquisition}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Method / comparison & Budget & Pair loss $\downarrow$ & Pair accuracy $\uparrow$ & Set regret/photo $\downarrow$ \\",
            r"\midrule",
            r"\multicolumn{5}{l}{\textit{Panel A: method means [95\% CI]}} \\",
            *mean_rows,
            r"\midrule",
            r"\multicolumn{5}{l}{\textit{Panel B: paired PDRR improvement [95\% CI]}} \\",
            *paired_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        )
    )
    (FIG_DIR / "TABLE_pdrr_acquisition_controlled.tex").write_text(
        table, encoding="utf-8", newline="\n"
    )


def _write_latex_include(name: str) -> None:
    snippet = "\n".join(
        (
            r"\begin{figure*}[t]",
            r"  \centering",
            rf"  \includegraphics[width=0.95\textwidth]{{figures/{name}.pdf}}",
            (
                r"  \caption{Query-efficiency comparison for the 67D contextual "
                r"preference adapter on 70 public Wikimedia images. Each replicate "
                r"uses the source-pinned 42/28 exact-file-disjoint split; acquisition "
                r"sees only the 42 training images. Lines are means across 10 "
                r"split/choice seeds after averaging three fixed simulated profiles, "
                r"shading gives 95\% percentile-bootstrap intervals, and faint points "
                r"are raw seed means. CAPU-PDRR-MC uses $B=64$ posterior draws, a "
                r"16-pair shortlist, and an exact partition-constrained action re-solve. "
                r"At budget 10, all paired PDRR-versus-random and PDRR-versus-entropy "
                r"intervals favor PDRR; at budgets 30 and 60 all corresponding intervals "
                r"cross zero. The evidence therefore supports low-budget sample "
                r"efficiency, not universal or asymptotic superiority. Preferences are "
                r"simulated rather than human-provided.}"
            ),
            r"  \label{fig:controlled-pdrr-acquisition}",
            r"\end{figure*}",
            "",
        )
    )
    (FIG_DIR / "pdrr_acquisition_latex_include.tex").write_text(
        snippet, encoding="utf-8", newline="\n"
    )


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    budgets = _validate_payload(payload)
    summary = payload["summary"]
    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), sharex=True)
    panel_letters = ("(a)", "(b)", "(c)")
    all_panel_values: dict[str, list[float]] = {metric: [] for metric, _ in PANELS}

    for method_index, (method, label, color, linestyle, marker) in enumerate(METHODS):
        for ax, (metric, ylabel) in zip(axes, PANELS, strict=True):
            means, lows, highs, raw = _metric_arrays(summary, method, budgets, metric)
            ax.fill_between(
                budgets,
                lows,
                highs,
                color=color,
                alpha=0.10 if method != "pdrr_mc_contextual" else 0.14,
                linewidth=0,
            )
            ax.plot(
                budgets,
                means,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.1,
                linewidth=1.85 if method == "pdrr_mc_contextual" else 1.35,
                zorder=4 if method == "pdrr_mc_contextual" else 3,
            )
            for budget, seed_values in zip(budgets, raw, strict=True):
                offsets = np.linspace(-0.38, 0.38, len(seed_values))
                offsets += (method_index - 1.5) * 0.045
                ax.scatter(
                    budget + offsets,
                    seed_values,
                    color=color,
                    marker=marker,
                    s=7,
                    alpha=0.20,
                    linewidths=0,
                    zorder=2,
                )
                all_panel_values[metric].extend(float(value) for value in seed_values)
            all_panel_values[metric].extend(lows.tolist())
            all_panel_values[metric].extend(highs.tolist())
            ax.set_ylabel(ylabel)
            ax.set_xticks(budgets)
            ax.set_xlabel("Pair-query opportunities")
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
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handlelength=2.0,
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
    _write_table(payload)
    _write_latex_include(args.name)
    print(
        json.dumps(
            {
                "path_base": "repository-root",
                "pdf": _public_path(FIG_DIR / f"{args.name}.pdf"),
                "png": _public_path(FIG_DIR / f"{args.name}.png"),
                "table": _public_path(
                    FIG_DIR / "TABLE_pdrr_acquisition_controlled.tex"
                ),
                "latex_include": _public_path(
                    FIG_DIR / "pdrr_acquisition_latex_include.tex"
                ),
                "n_seed_means": 10,
                "profiles_per_seed": 3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
