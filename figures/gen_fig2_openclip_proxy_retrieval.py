from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import FIG_DIR, save_fig


DATA_PATH = FIG_DIR / "openclip_raw_v2_proxy_eval_20260828.json"
FIGURE_NAME = "fig2_openclip_raw_v2_proxy_retrieval"
TABLE_PATH = FIG_DIR / "TABLE_openclip_raw_v2_proxy_retrieval.tex"

LIGHTWEIGHT = "lightweight-semantic-v1"
LEGACY = "openclip-xlm-roberta-base-vit-b-32-laion5b-zh-bridge-v1"
RAW_V2 = "openclip-xlm-roberta-base-vit-b-32-laion5b-raw-v2"

METHODS = (
    (LIGHTWEIGHT, "Lightweight", "#666666", "^"),
    (LEGACY, "Legacy bridge", "#D55E00", "s"),
    (RAW_V2, "Raw-v2", "#0072B2", "o"),
)

MACRO_METRICS = (
    ("precision_at_10", "P@10"),
    ("recall_at_20", "R@20"),
    ("ndcg_at_10", "nDCG@10"),
    ("ndcg_at_20", "nDCG@20"),
)

QUERY_FAMILIES = (
    ("architecture", "Travel\narchitecture"),
    ("city_night", "City night\nphotography"),
    ("mountain_landscape", "Mountain travel\nlandscape"),
)


def _load_runs(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scope = data["scope"]
    if scope["queries_per_language"] != 3:
        raise ValueError("this figure is designed for exactly three query families")
    if scope["candidate_images"] != 81 or scope["proxy_labeled_images"] != 72:
        raise ValueError("unexpected benchmark scope; regenerate or review the caption")
    if not all(data["validation"]["raw_chinese_queries_preserved"]):
        raise ValueError("raw-v2 did not preserve every Chinese query")

    runs = {(run["query_language"], run["method"]): run for run in data["runs"]}
    required = {("chinese", method) for method, *_ in METHODS} | {
        ("english", LIGHTWEIGHT),
        ("english", RAW_V2),
    }
    missing = required - runs.keys()
    if missing:
        raise ValueError(f"missing required benchmark runs: {sorted(missing)}")

    for key in required:
        values = runs[key]["macro"].values()
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite metric in run {key}")
    return data, runs


def _write_latex_table(runs: dict[tuple[str, str], Any]) -> None:
    rows = (
        ("English", "Lightweight", runs[("english", LIGHTWEIGHT)]),
        ("English", "Raw-v2", runs[("english", RAW_V2)]),
        ("Chinese", "Lightweight", runs[("chinese", LIGHTWEIGHT)]),
        ("Chinese", "Legacy bridge", runs[("chinese", LEGACY)]),
        ("Chinese", "Raw-v2", runs[("chinese", RAW_V2)]),
    )
    metric_keys = (
        "mrr",
        "precision_at_10",
        "recall_at_20",
        "ndcg_at_10",
        "ndcg_at_20",
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Macro retrieval results under one production-equivalent "
            r"ranking run. Each row averages the same three query families over "
            r"81 candidates (72 Wikimedia search-term proxy labels; nine unjudged "
            r"synthetic derivatives count as non-relevant). These deterministic "
            r"pilot values have no confidence intervals and support no significance "
            r"claim. Legacy bridge reuses the raw-v2 image vectors and changes only "
            r"the Chinese text preparation.}"
        ),
        r"\label{tab:openclip-raw-v2-proxy}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        "Language & Provider / query path & MRR & P@10 & R@20 & nDCG@10 & nDCG@20 \\\\",
        r"\midrule",
    ]
    for index, (language, provider, run) in enumerate(rows):
        if index == 2:
            lines.append(r"\midrule")
        metrics = " & ".join(f"{float(run['macro'][key]):.3f}" for key in metric_keys)
        lines.append(f"{language} & {provider} & {metrics} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    data, runs = _load_runs(DATA_PATH)
    _write_latex_table(runs)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.25),
        gridspec_kw={"width_ratios": (1.08, 0.92), "wspace": 0.28},
    )

    metric_x = np.arange(len(MACRO_METRICS), dtype=float)
    width = 0.23
    offsets = (-width, 0.0, width)
    for offset, (method, label, color, marker) in zip(offsets, METHODS, strict=True):
        run = runs[("chinese", method)]
        values = [float(run["macro"][key]) for key, _ in MACRO_METRICS]
        axes[0].bar(
            metric_x + offset,
            values,
            width=width * 0.88,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=3,
        )
        query_jitter = np.asarray((-0.035, 0.0, 0.035))
        for metric_index, (metric_key, _) in enumerate(MACRO_METRICS):
            query_values = [
                float(run["per_query"][query_key]["metrics"][metric_key])
                for query_key, _ in QUERY_FAMILIES
            ]
            axes[0].scatter(
                metric_x[metric_index] + offset + query_jitter,
                query_values,
                s=12,
                facecolors="white",
                edgecolors=color,
                linewidths=0.8,
                zorder=4,
            )

    axes[0].set_xticks(metric_x, [label for _, label in MACRO_METRICS])
    axes[0].set_ylabel("Macro score (mean over 3 query families)")
    axes[0].set_ylim(0.0, 1.06)
    axes[0].set_yticks(np.linspace(0.0, 1.0, 6))
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    axes[0].text(
        0.01,
        0.98,
        "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
        va="top",
    )

    query_x = np.arange(len(QUERY_FAMILIES), dtype=float)
    for offset, (method, label, color, marker) in zip(
        (-0.09, 0.0, 0.09), METHODS, strict=True
    ):
        run = runs[("chinese", method)]
        values = [
            float(run["per_query"][key]["metrics"]["ndcg_at_20"])
            for key, _ in QUERY_FAMILIES
        ]
        axes[1].scatter(
            query_x + offset,
            values,
            color=color,
            marker=marker,
            s=38,
            label=label,
            zorder=3,
        )
    axes[1].set_xticks(query_x, [label for _, label in QUERY_FAMILIES])
    axes[1].set_ylabel("Per-query nDCG@20")
    axes[1].set_ylim(0.0, 1.06)
    axes[1].set_yticks(np.linspace(0.0, 1.0, 6))
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    axes[1].text(
        0.01,
        0.98,
        "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
        va="top",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.025),
        ncol=3,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.4,
    )
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.09, right=0.995)
    fig.text(
        0.5,
        0.035,
        (
            "Fixed Wikimedia proxy fixture; 3 Chinese query families; no CI, "
            "significance test, or general multilingual-retrieval claim."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#4D4D4D",
    )
    save_fig(fig, FIGURE_NAME)
    plt.close(fig)

    print(
        json.dumps(
            {
                "input": str(DATA_PATH),
                "outputs": [
                    str(FIG_DIR / f"{FIGURE_NAME}.pdf"),
                    str(FIG_DIR / f"{FIGURE_NAME}.png"),
                    str(TABLE_PATH),
                ],
                "experiment_id": data["experiment_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
