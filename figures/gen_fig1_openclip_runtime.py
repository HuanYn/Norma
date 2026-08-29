from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, FONT_SIZE, save_fig


ROOT = Path(__file__).resolve().parent
with (ROOT / "openclip_raw_v2_runtime_20260828.json").open(encoding="utf-8") as handle:
    data = json.load(handle)

fig, ax = plt.subplots(figsize=(5.25, 3.25))

for index, timing in enumerate(data["timings"]):
    values = np.asarray(timing["wall_seconds"], dtype=float)
    offsets = np.linspace(-0.09, 0.09, values.size) if values.size > 1 else np.zeros(1)
    color = COLORS[index]
    ax.scatter(
        np.full(values.size, index) + offsets,
        values,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        s=42,
        zorder=3,
    )
    median = float(np.median(values))
    ax.hlines(median, index - 0.16, index + 0.16, color=color, linewidth=2.2, zorder=2)
    ax.text(
        index,
        median * 1.38,
        f"median {median:.3g} s",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE - 1,
        color=color,
    )

ax.set_yscale("log")
ax.set_ylim(0.05, 650)
ax.set_yticks([0.1, 1, 10, 100])
ax.set_yticklabels(["0.1", "1", "10", "100"])
ax.set_ylabel("CPU wall-clock time (s, log scale)")
ax.set_xticks(range(len(data["timings"])))
ax.set_xticklabels(
    [f"{item['label']}\n(n={len(item['wall_seconds'])})" for item in data["timings"]]
)
ax.tick_params(axis="x", length=0, pad=7)
ax.spines["left"].set_bounds(0.1, 100)
fig.subplots_adjust(bottom=0.28, left=0.14, right=0.98, top=0.95)

save_fig(fig, "fig1_openclip_raw_v2_runtime")
plt.close(fig)
