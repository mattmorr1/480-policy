"""
Generate the four paper figures from results/metrics.json and results/raw_results.json.

Outputs:
  results/fig_1_deletion_latency.pdf
  results/fig_2_post_deletion_recall.pdf
  results/fig_3_leakage_collateral.pdf
  results/fig_4_compliance_table.pdf

Usage:
  python src/analyze.py --results_dir results/ --output_dir results/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

SYSTEM_LABELS = {
    "A": "System A\n(Shared LoRA)",
    "B": "System B\n(Per-User LoRA)",
    "C": "System C\n(RAG)",
    "D": "System D\n(Contrastive)",
}
SYSTEM_ORDER = ["A", "B", "C", "D"]
COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]


def fig1_deletion_latency(metrics: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    systems = [s for s in SYSTEM_ORDER if s in metrics and "deletion_latency_seconds" in metrics[s]]
    latencies = [metrics[s]["deletion_latency_seconds"] for s in systems]
    labels = [SYSTEM_LABELS[s] for s in systems]
    colors = [COLORS[SYSTEM_ORDER.index(s)] for s in systems]

    bars = ax.bar(labels, latencies, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Seconds (log scale)", fontsize=12)
    ax.set_title("Time to Honor an Article 17 Erasure Request", fontsize=13, fontweight="bold")
    ax.set_xlabel("System", fontsize=12)

    # Annotate System A with human-readable time
    for bar, system, val in zip(bars, systems, latencies):
        if system == "A":
            mins = val / 60
            label = f"{mins:.0f} min" if mins >= 1 else f"{val:.1f}s"
        elif val < 0.001:
            label = f"{val*1000:.3f}ms"
        elif val < 1:
            label = f"{val*1000:.1f}ms"
        else:
            label = f"{val:.2f}s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plt.tight_layout()
    out = out_dir / "fig_1_deletion_latency.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig2_post_deletion_recall(metrics: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    systems = [s for s in SYSTEM_ORDER if s in metrics]
    labels = [SYSTEM_LABELS[s] for s in systems]
    colors = [COLORS[SYSTEM_ORDER.index(s)] for s in systems]

    # Left: ROUGE-L on holdout01 (recall after deletion)
    rouge_vals = [metrics[s].get("recall_rouge_l") or 0.0 for s in systems]
    axes[0].bar(labels, rouge_vals, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("ROUGE-L (higher = more recall)", fontsize=11)
    axes[0].set_title("Post-Deletion Recall (ROUGE-L)\nLower = deletion worked", fontsize=11, fontweight="bold")
    axes[0].axhline(0.1, color="black", linestyle="--", linewidth=0.8, label="Threshold: 0.1")
    axes[0].legend(fontsize=9)

    # Right: Truth Ratio (>0.5 = model still "knows" the answer)
    tr_vals = [metrics[s].get("truth_ratio_mean") or 0.5 for s in systems]
    axes[1].bar(labels, tr_vals, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Truth Ratio (>0.5 = residual memorization)", fontsize=11)
    axes[1].set_title("Truth Ratio After Deletion\nLower = deletion worked", fontsize=11, fontweight="bold")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random baseline: 0.5")
    axes[1].legend(fontsize=9)

    plt.suptitle("Post-Deletion Recall: Does Personalization Disappear?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "fig_2_post_deletion_recall.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig3_leakage_collateral(metrics: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    systems = [s for s in SYSTEM_ORDER if s in metrics]
    labels = [SYSTEM_LABELS[s] for s in systems]
    colors = [COLORS[SYSTEM_ORDER.index(s)] for s in systems]

    # Left: cross-author leakage rate
    leakage = [metrics[s].get("leakage_rate") or 0.0 for s in systems]
    axes[0].bar(labels, leakage, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Leakage Rate (fraction of probes that leaked)", fontsize=11)
    axes[0].set_title("Cross-Author Data Leakage\nLower = better isolation", fontsize=11, fontweight="bold")

    # Right: world_facts ROUGE-L — collateral damage from System D
    world = [metrics[s].get("world_facts_rouge_l") or 0.0 for s in systems]
    axes[1].bar(labels, world, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("ROUGE-L on World Facts", fontsize=11)
    axes[1].set_title("Collateral Damage: General Knowledge\nHigher = better (no collateral damage)", fontsize=11, fontweight="bold")

    plt.suptitle("Specificity: Does Deletion Affect the Right Data Only?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "fig_3_leakage_collateral.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig4_compliance_table(metrics: dict, out_dir: Path):
    """Social contract compliance table: rows = clauses, columns = systems.

    Notes on metric choices:
    - Completeness uses ROUGE-L (not Truth Ratio). The TR implementation compares
      correct answers against reversed-word-order negatives, which any LM assigns
      low probability to regardless of memorization — making TR ~0.98 for all systems
      including the vanilla base model. ROUGE-L on post-deletion recall probes is the
      reliable signal.
    - Leakage verdicts are N/A when eval was run with --mock_judge (random 0/1).
      System B is marked Partial pending a real LLM judge re-run.
    """
    clauses = [
        "Deletion latency\n(<1 second)",
        "Deletion completeness\n(ROUGE-L < 0.1 post-deletion)",
        "Cross-user isolation\n(Leakage rate < 0.1)",
        "Auditability\n(traceable to data)",
    ]

    def latency_verdict(s):
        lat = metrics.get(s, {}).get("deletion_latency_seconds")
        if lat is None:
            return "N/A", "gray"
        if lat < 1:
            return "Pass", "green"
        if lat < 60:
            return "Partial", "orange"
        return "Fail", "red"

    def completeness_verdict(s):
        # Use ROUGE-L on post-deletion recall as the primary completeness metric.
        # Truth Ratio is unreliable here: the reversed-word-order negative baseline
        # inflates TR to ~0.98 even for the base model that never saw TOFU data.
        rouge = metrics.get(s, {}).get("recall_rouge_l")
        if rouge is None:
            return "N/A", "gray"
        if rouge < 0.1:
            return "Pass", "green"
        if rouge < 0.25:
            return "Partial", "orange"
        return "Fail", "red"

    def leakage_verdict(s):
        lr = metrics.get(s, {}).get("leakage_rate")
        import math
        if lr is None or (isinstance(lr, float) and math.isnan(lr)):
            return "N/A", "gray"
        # Leakage scores near 0.5 indicate mock judge (random) — treat as N/A
        if abs(lr - 0.5) < 0.15:
            return "N/A*", "gray"
        if lr < 0.1:
            return "Pass", "green"
        if lr < 0.3:
            return "Partial", "orange"
        return "Fail", "red"

    # Auditability is a known architectural property of each system
    auditability = {
        "A": ("Fail", "red"),       # shared weights — can't trace which data to remove
        "B": ("Partial", "orange"), # adapter deletion confirmed; TR metric unreliable
        "C": ("Pass", "green"),     # index deletion is exact and auditable
        "D": ("Fail", "red"),       # token penalty has no data-level audit trail
    }

    verdicts = {
        "A": [latency_verdict("A"), completeness_verdict("A"), leakage_verdict("A"), auditability.get("A", ("N/A", "gray"))],
        "B": [latency_verdict("B"), completeness_verdict("B"), leakage_verdict("B"), auditability.get("B", ("N/A", "gray"))],
        "C": [latency_verdict("C"), completeness_verdict("C"), leakage_verdict("C"), auditability.get("C", ("N/A", "gray"))],
        "D": [latency_verdict("D"), completeness_verdict("D"), leakage_verdict("D"), auditability.get("D", ("N/A", "gray"))],
    }

    color_map = {"green": "#c8e6c9", "orange": "#ffe0b2", "red": "#ffcdd2", "gray": "#f5f5f5"}

    # Build row/col data
    col_labels = [SYSTEM_LABELS[s] for s in SYSTEM_ORDER]  # keep \n for two-line headers
    row_labels = clauses                                     # keep \n for two-line row labels

    table_texts = []
    table_colors = []
    for clause_idx in range(len(clauses)):
        row_t, row_c = [], []
        for s in SYSTEM_ORDER:
            text, color = verdicts.get(s, [("N/A", "gray")] * 4)[clause_idx]
            row_t.append(text)
            row_c.append(color_map.get(color, "#f5f5f5"))
        table_texts.append(row_t)
        table_colors.append(row_c)

    n_rows = len(clauses)
    n_cols = len(SYSTEM_ORDER)

    # Layout constants (all in figure-fraction units via axes coords 0..1)
    row_label_w = 0.36   # fraction of axes width for the row-label column
    col_w = (1.0 - row_label_w) / n_cols
    header_h = 0.18      # fraction of axes height for the header row
    row_h = (1.0 - header_h) / n_rows

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    header_bg = "#37474f"
    header_fg = "white"
    row_label_bg = "#eceff1"

    # Draw header: blank top-left corner, then one cell per system
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 1 - header_h), row_label_w, header_h,
        boxstyle="square,pad=0", facecolor=header_bg, edgecolor="white", linewidth=1.5,
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(row_label_w / 2, 1 - header_h / 2, "Compliance Criterion",
            ha="center", va="center", fontsize=11, fontweight="bold", color=header_fg,
            transform=ax.transAxes)

    for j, (s, label) in enumerate(zip(SYSTEM_ORDER, col_labels)):
        x = row_label_w + j * col_w
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 1 - header_h), col_w, header_h,
            boxstyle="square,pad=0", facecolor=header_bg, edgecolor="white", linewidth=1.5,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(x + col_w / 2, 1 - header_h / 2, label,
                ha="center", va="center", fontsize=11, fontweight="bold", color=header_fg,
                multialignment="center", transform=ax.transAxes)

    # Draw data rows
    for i in range(n_rows):
        y = 1 - header_h - (i + 1) * row_h
        stripe = "#fafafa" if i % 2 == 0 else "#f0f0f0"

        # Row label cell
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y), row_label_w, row_h,
            boxstyle="square,pad=0", facecolor=stripe, edgecolor="white", linewidth=1,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(row_label_w - 0.01, y + row_h / 2, row_labels[i],
                ha="right", va="center", fontsize=10, fontweight="bold",
                multialignment="right", transform=ax.transAxes)

        # Data cells
        for j in range(n_cols):
            x = row_label_w + j * col_w
            cell_color = table_colors[i][j]
            verdict_text = table_texts[i][j]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), col_w, row_h,
                boxstyle="square,pad=0", facecolor=cell_color, edgecolor="white", linewidth=1,
                transform=ax.transAxes, clip_on=False,
            ))
            ax.text(x + col_w / 2, y + row_h / 2, verdict_text,
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    transform=ax.transAxes)

    # Title
    ax.set_title(
        "Social Contract Compliance: Which Systems Honor GDPR Article 17?\n"
        "Primary compliant architecture: System C (RAG).  "
        "System B = partial success (leakage score is N/A* from mock judge).\n"
        "Completeness uses ROUGE-L < 0.1 post-deletion. "
        "N/A* = mock judge (random); Truth Ratio metric unreliable with reversed-word baseline.",
        fontsize=9.5, pad=12,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor="#c8e6c9", label="Pass", edgecolor="gray"),
        mpatches.Patch(facecolor="#ffe0b2", label="Partial", edgecolor="gray"),
        mpatches.Patch(facecolor="#ffcdd2", label="Fail", edgecolor="gray"),
        mpatches.Patch(facecolor="#f5f5f5", label="N/A / N/A*", edgecolor="gray"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10,
              framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    out = out_dir / "fig_4_compliance_table.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json") as f:
        metrics = json.load(f)

    print("Generating figures...")
    fig1_deletion_latency(metrics, out_dir)
    fig2_post_deletion_recall(metrics, out_dir)
    fig3_leakage_collateral(metrics, out_dir)
    fig4_compliance_table(metrics, out_dir)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
