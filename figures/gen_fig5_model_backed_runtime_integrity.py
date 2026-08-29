from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

from paper_plot_style import COLORS, FIG_DIR, save_fig


OUTPUT_NAME = "fig5_model_backed_runtime_integrity"
SUMMARY_PATH = FIG_DIR / "model_backed_runtime_integrity_20260829.json"
SYSTEM_PROFILE_PATH = FIG_DIR / "model_backed_runtime_system_profile_20260829.json"
ARTIFACTS = {
    "openclip_clean": (
        FIG_DIR / "openclip_pinned_v3_smoke_20260829.json",
        "9ed9caaf33f23cf2996e51a2a2e58ad57e521f8620f8f57746a5c7b9b550cad2",
    ),
    "openclip_standby": (
        FIG_DIR / "openclip_pinned_v3_smoke_standby_overlap_20260829.json",
        "178295b487ffd1737df7fc87b6e3ed631620b3a0ffee03112ba19d61c5df0080",
    ),
    "qwen_clean": (
        FIG_DIR / "qwen3vl_grounded_smoke_20260829.json",
        "2651112ba55c6d0beb1d94d87d3788f34559ab31c23716ad81ace7cee4d488ca",
    ),
    "qwen_standby": (
        FIG_DIR / "qwen3vl_grounded_smoke_v5_passed_standby_contaminated_20260829.json",
        "7a4549c287d03dfc9ab7930a5aa79015d8a82a9415dcaa7b4fcb5c52a226515b",
    ),
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _load_sources() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    sources: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for key, (path, expected_sha256) in ARTIFACTS.items():
        observed_sha256 = _sha256(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"frozen artifact drifted: {path.name}; "
                f"expected {expected_sha256}, got {observed_sha256}"
            )
        sources[key] = _load_json(path)
        hashes[key] = observed_sha256
    return sources, hashes


def _verify_contract(sources: dict[str, dict[str, Any]]) -> dict[str, bool]:
    openclip_clean = sources["openclip_clean"]
    openclip_old = sources["openclip_standby"]
    qwen_clean = sources["qwen_clean"]
    qwen_old = sources["qwen_standby"]

    for item in (openclip_clean, openclip_old, qwen_clean, qwen_old):
        if item.get("status") != "passed":
            raise ValueError("a frozen model smoke artifact is not passed")
    for item in (openclip_old, qwen_old):
        if (
            item.get("timing_valid") is not False
            or item.get("excluded_from_latency_claims") is not True
        ):
            raise ValueError("a standby-overlap artifact is not fail-closed")

    for key, clean_key in (
        ("openclip_standby", "openclip_clean"),
        ("qwen_standby", "qwen_clean"),
    ):
        exclusion = sources[key]["timing_exclusion"]
        clean_path, clean_sha256 = ARTIFACTS[clean_key]
        if (
            exclusion["clean_replacement_path"]
            != clean_path.relative_to(FIG_DIR.parent).as_posix()
            or exclusion["clean_replacement_sha256"] != clean_sha256
        ):
            raise ValueError("historical artifact clean cross-link drifted")

    openclip_vectors = ("chinese", "chinese_repeat", "english", "image")
    openclip_unchanged = (
        openclip_clean["provider_fingerprint"] == openclip_old["provider_fingerprint"]
        and openclip_clean["image"]["sha256"] == openclip_old["image"]["sha256"]
        and all(
            openclip_clean["observations"][name]["sha256_float32"]
            == openclip_old["observations"][name]["sha256_float32"]
            for name in openclip_vectors
        )
    )
    qwen_unchanged = (
        qwen_clean["generation_provider_fingerprint"]
        == qwen_old["generation_provider_fingerprint"]
        and qwen_clean["image"]["sha256"] == qwen_old["image"]["sha256"]
        and qwen_clean["provenance"] == qwen_old["provenance"]
        and [
            (run["answer"], run["claims"], run["citations"])
            for run in qwen_clean["runs"]
        ]
        == [
            (run["answer"], run["claims"], run["citations"]) for run in qwen_old["runs"]
        ]
    )
    qwen_replay = (
        qwen_clean.get("deterministic_replay") is True
        and len(qwen_clean["runs"]) == 2
        and qwen_clean["runs"][0]["answer"] == qwen_clean["runs"][1]["answer"]
        and qwen_clean["runs"][0]["claims"] == qwen_clean["runs"][1]["claims"]
        and qwen_clean["runs"][0]["citations"] == qwen_clean["runs"][1]["citations"]
    )
    checks = {
        "openclip_functional_payload_unchanged": openclip_unchanged,
        "qwen_functional_payload_unchanged": qwen_unchanged,
        "qwen_same_process_canonical_replay": qwen_replay,
        "standby_timings_excluded": True,
        "semantic_entailment_evaluated": False,
    }
    if not all(
        value for key, value in checks.items() if key != "semantic_entailment_evaluated"
    ):
        raise ValueError("functional contract comparison failed")
    return checks


def _write_summary(
    sources: dict[str, dict[str, Any]],
    hashes: dict[str, str],
    checks: dict[str, bool],
    system_profile_sha256: str,
) -> None:
    openclip = sources["openclip_clean"]
    qwen = sources["qwen_clean"]
    payload = {
        "schema": "norma-model-backed-runtime-integrity-summary-v1",
        "claim_boundary": (
            "single-machine functional/contract smokes and raw timing observations; "
            "not model-quality, semantic-entailment, inferential, cold-disk, CLI end-to-end, "
            "or cross-machine performance evidence"
        ),
        "source_artifact_sha256": hashes,
        "system_profile_sha256": system_profile_sha256,
        "checks": checks,
        "clean_observations": {
            "openclip": {
                "provider_initialization_seconds": openclip["observations"][
                    "provider_initialization_seconds"
                ],
                "first_text_load_verify_encode_seconds": openclip["observations"][
                    "cold_text_seconds"
                ],
                "repeat_text_seconds": openclip["observations"]["repeat_text_seconds"],
                "english_text_seconds": openclip["observations"][
                    "english_text_seconds"
                ],
                "image_encode_seconds": openclip["observations"]["image_seconds"],
                "timed_block_seconds": openclip["duration_seconds"],
                "sampled_peak_rss_bytes": openclip["peak_rss_bytes"],
                "sample_size": 1,
            },
            "qwen3_vl": {
                "provider_initialization_seconds": qwen[
                    "provider_initialization_seconds"
                ],
                "run_seconds": [run["duration_seconds"] for run in qwen["runs"]],
                "timed_block_seconds": qwen["duration_seconds"],
                "sampled_peak_rss_bytes": qwen["peak_rss_bytes"],
                "sample_size": len(qwen["runs"]),
            },
        },
        "historical_excluded_timed_block_seconds": {
            "openclip": sources["openclip_standby"]["duration_seconds"],
            "qwen3_vl": sources["qwen_standby"]["duration_seconds"],
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _draw(sources: dict[str, dict[str, Any]]) -> None:
    openclip = sources["openclip_clean"]
    qwen = sources["qwen_clean"]
    openclip_values = [
        openclip["observations"]["cold_text_seconds"],
        openclip["observations"]["repeat_text_seconds"],
        openclip["observations"]["english_text_seconds"],
        openclip["observations"]["image_seconds"],
    ]
    qwen_values = [run["duration_seconds"] for run in qwen["runs"]]
    latency_labels = [
        "OpenCLIP first text\n(load + verify + encode)",
        "OpenCLIP repeat Chinese text",
        "OpenCLIP English text",
        "OpenCLIP image encode",
        "Qwen3-VL run 0",
        "Qwen3-VL run 1",
    ]
    latency_values = openclip_values + qwen_values
    latency_colors = [COLORS[0]] * 4 + [COLORS[1]] * 2

    fig = plt.figure(figsize=(7.35, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.8, 0.72, 1.2), wspace=0.18)

    latency_ax = fig.add_subplot(grid[0, 0])
    positions = np.arange(len(latency_values))[::-1]
    for position, value, color in zip(
        positions, latency_values, latency_colors, strict=True
    ):
        latency_ax.scatter(value, position, s=48, color=color, zorder=3)
        latency_ax.text(value * 1.17, position, f"{value:.3f}", va="center", fontsize=8)
    latency_ax.set_xscale("log")
    latency_ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    latency_ax.xaxis.set_minor_locator(NullLocator())
    latency_ax.set_xlim(0.045, 80)
    latency_ax.set_ylim(-0.65, len(latency_values) - 0.35)
    latency_ax.set_yticks(positions, latency_labels)
    latency_ax.set_xlabel("Observed stage wall time (s, log scale)")
    latency_ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
    latency_ax.text(
        -0.2, 1.04, "(a)", transform=latency_ax.transAxes, fontweight="bold"
    )

    memory_ax = fig.add_subplot(grid[0, 1])
    memory_values = [
        openclip["peak_rss_bytes"] / (1024**3),
        qwen["peak_rss_bytes"] / (1024**3),
    ]
    memory_x = [0.0, 2.0]
    for x_value, value, color in zip(
        memory_x, memory_values, (COLORS[0], COLORS[1]), strict=True
    ):
        memory_ax.plot([x_value, x_value], [0, value], color=color, lw=1.5, alpha=0.65)
        memory_ax.scatter(x_value, value, s=55, color=color, zorder=3)
        memory_ax.text(x_value, value + 0.2, f"{value:.2f}", ha="center", fontsize=8)
    memory_ax.set_xlim(-0.6, 2.6)
    memory_ax.set_ylim(0, 6.0)
    memory_ax.set_xticks(memory_x, ["CLIP", "Qwen"])
    memory_ax.set_ylabel("Sampled peak RSS (GiB)")
    memory_ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    memory_ax.text(-0.34, 1.04, "(b)", transform=memory_ax.transAxes, fontweight="bold")

    integrity_ax = fig.add_subplot(grid[0, 2])
    clean_totals = [openclip["duration_seconds"], qwen["duration_seconds"]]
    historical_totals = [
        sources["openclip_standby"]["duration_seconds"],
        sources["qwen_standby"]["duration_seconds"],
    ]
    rows = [1, 0]
    for row, clean, historical, color in zip(
        rows, clean_totals, historical_totals, (COLORS[0], COLORS[1]), strict=True
    ):
        integrity_ax.plot([clean, historical], [row, row], color="#B8B8B8", lw=1)
        integrity_ax.scatter(clean, row, s=52, color=color, zorder=3)
        integrity_ax.scatter(
            historical,
            row,
            s=58,
            marker="X",
            facecolor="#777777",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        integrity_ax.text(clean, row + 0.17, f"{clean:.1f}", ha="center", fontsize=8)
        integrity_ax.text(
            historical,
            row - 0.19,
            f"{historical:.1f} ×",
            ha="center",
            fontsize=8,
            color="#555555",
        )
    integrity_ax.set_xscale("log")
    integrity_ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    integrity_ax.xaxis.set_minor_locator(NullLocator())
    integrity_ax.set_xlim(15, 6000)
    integrity_ax.set_ylim(-0.55, 1.55)
    integrity_ax.set_yticks(rows, ["OpenCLIP", "Qwen3-VL"])
    integrity_ax.set_xlabel("Timed block wall time (s, log scale)")
    integrity_ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
    integrity_ax.text(
        -0.24, 1.04, "(c)", transform=integrity_ax.transAxes, fontweight="bold"
    )
    integrity_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#444444",
                markeredgecolor="none",
                label="Guarded observation",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#777777",
                markeredgecolor="white",
                label="Standby overlap · excluded",
            ),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.08, 1.005),
        handletextpad=0.35,
    )

    save_fig(fig, OUTPUT_NAME)
    plt.close(fig)


def _write_table(sources: dict[str, dict[str, Any]]) -> None:
    openclip = sources["openclip_clean"]
    qwen = sources["qwen_clean"]
    openclip_rss = openclip["peak_rss_bytes"] / (1024**3)
    qwen_rss = qwen["peak_rss_bytes"] / (1024**3)
    table = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Single-machine learned-model functional smokes on an Intel Core Ultra 7 255HX (20 cores, 32 GB class RAM), Windows 11 build 26200, CPU inference. Values are raw observations without confidence intervals. OpenCLIP first text includes model load, manifest verification, and encode; Qwen run 0 excludes Python/Torch import and the pre-timing weight hash. RSS is sampled every 50 ms.}}
\\label{{tab:model-backed-runtime-integrity}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{llrrl}}
\\toprule
Model & Observation & Time (s) & Peak RSS (GiB) & Scope \\\\
\\midrule
OpenCLIP & First text (load + verify + encode) & {openclip["observations"]["cold_text_seconds"]:.3f} & {openclip_rss:.3f} & $n=1$ functional smoke \\\\
OpenCLIP & Repeat Chinese text / image encode & {openclip["observations"]["repeat_text_seconds"]:.3f} / {openclip["observations"]["image_seconds"]:.3f} & -- & same process \\\\
Qwen3-VL & Grounded generation run 0 & {qwen["runs"][0]["duration_seconds"]:.3f} & {qwen_rss:.3f} & referential validation only \\\\
Qwen3-VL & Grounded generation run 1 & {qwen["runs"][1]["duration_seconds"]:.3f} & -- & same canonical output \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (FIG_DIR / "TABLE_model_backed_runtime_integrity.tex").write_text(
        table, encoding="utf-8", newline="\n"
    )
    include = """% Generated by gen_fig5_model_backed_runtime_integrity.py
\\begin{figure*}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{figures/fig5_model_backed_runtime_integrity.pdf}
  \\caption{Real CPU inference observations for the pinned learned OpenCLIP and Qwen3-VL stack. (a) Raw clean stage timings; points are individual observations, not distributions. (b) 50-ms sampled process RSS peaks. (c) Modern-Standby-overlap observations are retained for audit but excluded from latency claims; guarded runs are not interpreted as model speedups. Functional vector/output payloads were unchanged. The smokes do not evaluate retrieval quality or semantic entailment.}
  \\label{fig:model-backed-runtime-integrity}
\\end{figure*}
\\input{figures/TABLE_model_backed_runtime_integrity.tex}
"""
    (FIG_DIR / "model_backed_runtime_integrity_latex_include.tex").write_text(
        include, encoding="utf-8", newline="\n"
    )


def main() -> None:
    sources, hashes = _load_sources()
    checks = _verify_contract(sources)
    system_profile = _load_json(SYSTEM_PROFILE_PATH)
    if system_profile.get("schema") != "norma-model-runtime-system-profile-v1":
        raise ValueError("unexpected model runtime system profile")
    _write_summary(sources, hashes, checks, _sha256(SYSTEM_PROFILE_PATH))
    _draw(sources)
    _write_table(sources)


if __name__ == "__main__":
    main()
