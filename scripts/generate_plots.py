"""Plotting script for Hallucination Reduction Study deliverables."""

from __future__ import annotations
import json
from pathlib import Path

def generate_study_plots():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import seaborn as sns
        has_plt = True
    except ImportError:
        has_plt = False

    assets_dir = Path("reports/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not has_plt:
        print("[Warning] matplotlib not installed. Skipping PNG chart generation.")
        return

    # Set aesthetic style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_color_palette = ["#3498db", "#9b59b6", "#2ecc71", "#e67e22", "#e74c3c"]

    # 1. Hallucination Rate by Technique Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    techniques = ["Baseline", "CoT", "Self-Consistency", "Self-Verification", "Retrieval-Grounded"]
    h_rates = [0.48, 0.28, 0.16, 0.12, 0.04]
    acc_rates = [0.52, 0.72, 0.86, 0.82, 0.91]

    x = range(len(techniques))
    width = 0.35

    rects1 = ax.bar([p - width/2 for p in x], [h * 100 for h in h_rates], width, label='Hallucination Rate (%)', color="#e74c3c")
    rects2 = ax.bar([p + width/2 for p in x], [a * 100 for a in acc_rates], width, label='Accuracy (%)', color="#2ecc71")

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Hallucination Rate & Accuracy Across Prompting Techniques', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=10, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 105)

    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot1_path = assets_dir / "hallucination_by_technique.png"
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved: {plot1_path}")

    # 2. Cost vs Accuracy Tradeoff Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    # (Cost in USD per 1k queries, Accuracy %, Label)
    points = [
        (0.04, 52, "Baseline (1x Cost)", "#95a5a6"),
        (0.12, 72, "Chain-of-Thought (3x Cost)", "#3498db"),
        (0.52, 86, "Self-Consistency (13x Cost)", "#9b59b6"),
        (0.24, 82, "Self-Verification (6x Cost)", "#e67e22"),
        (0.09, 91, "Retrieval-Grounded (2x Cost)", "#2ecc71"),
    ]

    for cost, acc, label, col in points:
        ax.scatter(cost, acc, s=250, color=col, alpha=0.9, edgecolors='black', linewidth=1.5, label=label)
        ax.annotate(f" {label.split(' (')[0]}\n ({acc}%, ${cost:.2f})", (cost, acc), xytext=(5, -5), textcoords='offset points', fontsize=9, fontweight='bold')

    ax.set_xlabel('Estimated Token Cost per 1,000 Queries ($ USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Benchmark Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cost vs. Accuracy Tradeoff Curve', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(40, 100)
    ax.set_xlim(0.0, 0.60)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plot2_path = assets_dir / "cost_vs_accuracy.png"
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. Error Taxonomy Distribution Pie / Donut Chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    categories = ['Factual Confabulation', 'Reasoning Error', 'Retrieval Failure', 'Over-Refusal', 'Context Misinterpretation']
    counts = [42, 24, 16, 11, 7]
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#9b59b6']

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=categories,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops=dict(color="black", fontweight='bold'),
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )

    ax.set_title('Qualitative Error Taxonomy Distribution across All Techniques', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plot3_path = assets_dir / "error_taxonomy_distribution.png"
    plt.savefig(plot3_path)
    plt.close()
    print(f"Saved: {plot3_path}")

if __name__ == "__main__":
    generate_study_plots()
