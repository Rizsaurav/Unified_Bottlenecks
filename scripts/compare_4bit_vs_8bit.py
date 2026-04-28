"""
4-bit (Q4_K_M) vs 8-bit (Q8_0) Comparative Analysis
Generates side-by-side figures and a summary CSV for the paper.
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

base = os.path.dirname(os.path.abspath(__file__))
res  = os.path.join(base, "../data/processed_csvs")

# ── Load all 6 enriched CSVs ──
WORKLOADS = ["Text-Only_Baseline", "Vision+Text_single_image", "Video_Inference_5_frames"]
SHORT     = ["Text-Only", "Vision+Text", "Video (5fr)"]

data = {}
for wl in WORKLOADS:
    f4 = os.path.join(res, f"enriched_{wl}.csv")
    f8 = os.path.join(res, f"enriched_{wl}_8bit.csv")
    if not os.path.exists(f4) or not os.path.exists(f8):
        print(f"SKIP {wl}: missing file"); continue
    data[wl] = {"q4": pd.read_csv(f4), "q8": pd.read_csv(f8)}
    print(f"Loaded {wl}: Q4={len(data[wl]['q4']):,}  Q8={len(data[wl]['q8']):,}")

PHASE_COLORS = {"vision_encoding": "#e74c3c", "text_generation": "#3498db", "idle": "#95a5a6"}
Q_COLORS     = {"q4": "#e67e22", "q8": "#8e44ad"}
PHASES       = ["vision_encoding", "text_generation"]

# ── Helper: per-phase stats ──
def phase_stats(df, phase):
    s = df[df["phase"] == phase]
    if len(s) == 0:
        return {}
    active = s[s["gpu_freq_mhz"] > 0]
    return {
        "n": len(s),
        "combined_mean": s["combined_mw"].mean(),
        "combined_peak": s["combined_mw"].max(),
        "cpu_mean": s["cpu_mw"].mean(),
        "gpu_mean": s["gpu_mw"].mean(),
        "ane_mean": s["ane_mw"].mean() if "ane_mw" in s else 0,
        "gpu_res_mean": s["gpu_residency"].mean(),
        "gpu_freq_mean": active["gpu_freq_mhz"].mean() if len(active) else 0,
        "pct_444": (active["gpu_freq_mhz"] <= 444).mean() * 100 if len(active) else 0,
        "pct_gt800": (active["gpu_freq_mhz"] > 800).mean() * 100 if len(active) else 0,
        "sw_p1_mean": s[s["gpu_sw_p1_pct"] > 0]["gpu_sw_p1_pct"].mean() if (s["gpu_sw_p1_pct"] > 0).any() else 0,
    }

# ── Build summary table ──
rows = []
for wl, short in zip(WORKLOADS, SHORT):
    if wl not in data: continue
    for phase in PHASES:
        s4 = phase_stats(data[wl]["q4"], phase)
        s8 = phase_stats(data[wl]["q8"], phase)
        if not s4 or not s8: continue
        rows.append({
            "Workload": short, "Phase": phase.replace("_"," ").title(),
            "Q4_Combined_Mean": s4["combined_mean"], "Q8_Combined_Mean": s8["combined_mean"],
            "Delta_Combined_pct": (s8["combined_mean"]/s4["combined_mean"]-1)*100,
            "Q4_CPU_Mean": s4["cpu_mean"], "Q8_CPU_Mean": s8["cpu_mean"],
            "Q4_GPU_Res": s4["gpu_res_mean"], "Q8_GPU_Res": s8["gpu_res_mean"],
            "Q4_GPU_Freq": s4["gpu_freq_mean"], "Q8_GPU_Freq": s8["gpu_freq_mean"],
            "Q4_pct444": s4["pct_444"], "Q8_pct444": s8["pct_444"],
            "Delta_444_pp": s8["pct_444"] - s4["pct_444"],
            "Q4_pctGT800": s4["pct_gt800"], "Q8_pctGT800": s8["pct_gt800"],
            "Q4_SW_P1": s4["sw_p1_mean"], "Q8_SW_P1": s8["sw_p1_mean"],
        })

summary = pd.DataFrame(rows)
summary_path = os.path.join(res, "q4_vs_q8_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\nSaved summary → {summary_path}")
print(summary.to_string(index=False))

# ═══════════════════════════════════════════════
#  FIGURE 7: Side-by-side bar charts (6 metrics)
# ═══════════════════════════════════════════════
print("\nGenerating Figure 7: 4-bit vs 8-bit metric comparison bars...")

metrics_fn = {
    "Peak Combined\nPower (mW)":      lambda df: df["combined_mw"].max(),
    "Mean Combined\nPower (mW)":      lambda df: df[df["phase"]!="idle"]["combined_mw"].mean(),
    "Mean GPU\nResidency (%)":        lambda df: df["gpu_residency"].mean(),
    "Mean GPU Freq\n(active, MHz)":   lambda df: df[df["gpu_freq_mhz"]>0]["gpu_freq_mhz"].mean(),
    "% Time @444MHz\n(mem-stall)":    lambda df: (df[df["gpu_freq_mhz"]>0]["gpu_freq_mhz"]<=444).mean()*100,
    "Mean CPU\nPower (mW)":           lambda df: df[df["phase"]!="idle"]["cpu_mw"].mean(),
}

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("Q4_K_M (4-bit) vs Q8_0 (8-bit) — Cross-Workload Hardware Comparison\nInternVL2.5-4B on Apple M2 (8 GB Unified Memory)",
             fontsize=14, fontweight="bold")
axes = axes.flatten()

for ax, (mname, fn) in zip(axes, metrics_fn.items()):
    x = np.arange(len(SHORT))
    w = 0.35
    v4 = [fn(data[wl]["q4"]) for wl in WORKLOADS if wl in data]
    v8 = [fn(data[wl]["q8"]) for wl in WORKLOADS if wl in data]
    b4 = ax.bar(x - w/2, v4, w, label="Q4_K_M (4-bit)", color=Q_COLORS["q4"], alpha=0.85, edgecolor="white")
    b8 = ax.bar(x + w/2, v8, w, label="Q8_0 (8-bit)",   color=Q_COLORS["q8"], alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_title(mname, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(list(b4)+list(b8), v4+v8):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(v4+v8)*0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
out7 = os.path.join(res, "fig7_q4_vs_q8_bars.png")
plt.savefig(out7, dpi=200, bbox_inches="tight"); plt.close()
print(f"  Saved → {out7}")

# ═══════════════════════════════════════════════
#  FIGURE 8: Phase-resolved delta heatmap
# ═══════════════════════════════════════════════
print("Generating Figure 8: Phase-resolved delta heatmap...")

heat_cols = ["Delta_Combined_pct", "Delta_444_pp", "Q4_GPU_Res", "Q8_GPU_Res", "Q4_GPU_Freq", "Q8_GPU_Freq"]
heat_labels = ["Δ Combined\nPower (%)", "Δ %@444MHz\n(pp)", "Q4 GPU\nResidency", "Q8 GPU\nResidency", "Q4 GPU\nFreq (MHz)", "Q8 GPU\nFreq (MHz)"]

fig, ax = plt.subplots(figsize=(14, 5))
mat = summary[heat_cols].values
row_labels = [f"{r['Workload']} | {r['Phase']}" for _, r in summary.iterrows()]
im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(heat_labels))); ax.set_xticklabels(heat_labels, fontsize=9)
ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(mat[i,j]) > np.nanmax(abs(mat))*0.6 else "black")
ax.set_title("Q4 vs Q8 — Phase-Resolved Delta Heatmap\n(Negative Δ444 = less memory stall = more ALU work in Q8)",
             fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
out8 = os.path.join(res, "fig8_phase_delta_heatmap.png")
plt.savefig(out8, dpi=200, bbox_inches="tight"); plt.close()
print(f"  Saved → {out8}")

# ═══════════════════════════════════════════════
#  FIGURE 9: GPU freq distribution overlay (4 vs 8)
# ═══════════════════════════════════════════════
print("Generating Figure 9: GPU frequency distribution overlay...")

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("GPU Frequency Distribution — Q4_K_M vs Q8_0 by Workload × Phase\n"
             "Shift rightward = more ALU work | Shift leftward = more memory stall",
             fontsize=13, fontweight="bold")

for col, (wl, short) in enumerate(zip(WORKLOADS, SHORT)):
    if wl not in data: continue
    for row, phase in enumerate(PHASES):
        ax = axes[row, col]
        for qname, qlabel, qcolor in [("q4","Q4_K_M","#e67e22"),("q8","Q8_0","#8e44ad")]:
            sub = data[wl][qname]
            sub = sub[(sub["phase"]==phase) & (sub["gpu_freq_mhz"]>0)]
            if len(sub)==0: continue
            ax.hist(sub["gpu_freq_mhz"], bins=40, alpha=0.55, color=qcolor, label=qlabel, edgecolor="white", lw=0.3)
        ax.axvline(444, color="red", ls=":", lw=1.2, label="444 MHz floor")
        ax.set_title(f"{short} — {phase.replace('_',' ').title()}", fontsize=10)
        ax.set_xlabel("GPU Freq (MHz)"); ax.set_ylabel("Count")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
out9 = os.path.join(res, "fig9_freq_overlay_q4_q8.png")
plt.savefig(out9, dpi=200, bbox_inches="tight"); plt.close()
print(f"  Saved → {out9}")

# ═══════════════════════════════════════════════
#  FIGURE 10: Scatter overlay — bottleneck regime
# ═══════════════════════════════════════════════
print("Generating Figure 10: Bottleneck regime scatter overlay...")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("GPU Bottleneck Regime — Q4 vs Q8 Overlay (Residency × Frequency)\n"
             "Upper-right = ALU-bound | Lower-left = Memory-BW-bound",
             fontsize=13, fontweight="bold")

for ax, (wl, short) in zip(axes, zip(WORKLOADS, SHORT)):
    if wl not in data: continue
    for qname, qlabel, qcolor, marker in [("q4","Q4","#e67e22","o"),("q8","Q8","#8e44ad","^")]:
        sub = data[wl][qname]
        active = sub[sub["gpu_freq_mhz"]>0]
        ax.scatter(active["gpu_residency"], active["gpu_freq_mhz"],
                   c=qcolor, alpha=0.15, s=4, marker=marker, label=qlabel)
    ax.axvline(30, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.axhline(800, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.set_title(short, fontsize=11)
    ax.set_xlabel("GPU Residency (%)"); ax.set_ylabel("GPU Freq (MHz)")
    ax.set_xlim(-2, 102); ax.set_ylim(0, 1550)
    ax.legend(fontsize=9, markerscale=4); ax.grid(alpha=0.25)

plt.tight_layout()
out10 = os.path.join(res, "fig10_regime_scatter_overlay.png")
plt.savefig(out10, dpi=200, bbox_inches="tight"); plt.close()
print(f"  Saved → {out10}")

# ═══════════════════════════════════════════════
#  FIGURE 11: Decompression penalty delta bar chart
# ═══════════════════════════════════════════════
print("Generating Figure 11: Decompression penalty delta...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Decompression Penalty Proxy — % GPU Samples at 444 MHz (Memory-Stall Floor)\n"
             "Q4_K_M requires dequant ALU → expect fewer stalls than Q8 during vision encoding",
             fontsize=12, fontweight="bold")

for ax, (wl, short) in zip(axes, zip(WORKLOADS, SHORT)):
    if wl not in data: continue
    x = np.arange(len(PHASES))
    w = 0.3
    v4 = [phase_stats(data[wl]["q4"], p).get("pct_444",0) for p in PHASES]
    v8 = [phase_stats(data[wl]["q8"], p).get("pct_444",0) for p in PHASES]
    b4 = ax.bar(x-w/2, v4, w, color=Q_COLORS["q4"], label="Q4_K_M", alpha=0.85, edgecolor="white")
    b8 = ax.bar(x+w/2, v8, w, color=Q_COLORS["q8"], label="Q8_0",   alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(["Vision Enc.", "Text Gen."], fontsize=10)
    ax.set_ylabel("% Active Samples @444 MHz"); ax.set_ylim(0, 110)
    ax.set_title(short, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(list(b4)+list(b8), v4+v8):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8)
    # annotate delta
    for i in range(len(PHASES)):
        delta = v8[i] - v4[i]
        mid_y = max(v4[i], v8[i]) + 8
        ax.annotate(f"Δ={delta:+.1f}pp", xy=(i, mid_y), ha="center", fontsize=9,
                    fontweight="bold", color="#c0392b" if delta>0 else "#27ae60")

plt.tight_layout()
out11 = os.path.join(res, "fig11_decompression_penalty_delta.png")
plt.savefig(out11, dpi=200, bbox_inches="tight"); plt.close()
print(f"  Saved → {out11}")

# ═══════════════════════════════════════════════
#  FIGURE 12: Power efficiency — BW cost of quant
# ═══════════════════════════════════════════════
print("Generating Figure 12: Bandwidth cost of quantisation...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Memory Bandwidth Cost: Q4 vs Q8\nQ8 weights are 2× larger → more bytes over the bus → higher power",
             fontsize=13, fontweight="bold")

# Panel A: mean active combined power
ax = axes[0]
x = np.arange(len(SHORT)); w = 0.3
v4 = [data[wl]["q4"][data[wl]["q4"]["phase"]!="idle"]["combined_mw"].mean() for wl in WORKLOADS if wl in data]
v8 = [data[wl]["q8"][data[wl]["q8"]["phase"]!="idle"]["combined_mw"].mean() for wl in WORKLOADS if wl in data]
ax.bar(x-w/2, v4, w, color=Q_COLORS["q4"], label="Q4_K_M", alpha=0.85, edgecolor="white")
ax.bar(x+w/2, v8, w, color=Q_COLORS["q8"], label="Q8_0",   alpha=0.85, edgecolor="white")
for i in range(len(v4)):
    pct = (v8[i]/v4[i]-1)*100
    ax.text(i, max(v4[i],v8[i])+50, f"+{pct:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=9)
ax.set_ylabel("Mean Active Combined Power (mW)"); ax.set_title("Active Power Draw", fontsize=11)
ax.legend(); ax.grid(axis="y", alpha=0.3)

# Panel B: peak combined power
ax = axes[1]
v4p = [data[wl]["q4"]["combined_mw"].max() for wl in WORKLOADS if wl in data]
v8p = [data[wl]["q8"]["combined_mw"].max() for wl in WORKLOADS if wl in data]
ax.bar(x-w/2, v4p, w, color=Q_COLORS["q4"], label="Q4_K_M", alpha=0.85, edgecolor="white")
ax.bar(x+w/2, v8p, w, color=Q_COLORS["q8"], label="Q8_0",   alpha=0.85, edgecolor="white")
for i in range(len(v4p)):
    pct = (v8p[i]/v4p[i]-1)*100
    ax.text(i, max(v4p[i],v8p[i])+200, f"+{pct:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=9)
ax.set_ylabel("Peak Combined Power (mW)"); ax.set_title("Peak Power Draw", fontsize=11)
ax.legend(); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out12 = os.path.join(res, "fig12_bandwidth_cost_quant.png")
plt.savefig(out12, dpi=200, bbox_inches="tight"); plt.close()
print(f"  Saved → {out12}")

print("\n" + "="*70)
print("COMPARATIVE ANALYSIS COMPLETE")
print("="*70)
print(f"New files in results/:")
for f in ["q4_vs_q8_summary.csv"] + [f"fig{i}_{n}.png" for i, n in [
    (7,"q4_vs_q8_bars"),(8,"phase_delta_heatmap"),(9,"freq_overlay_q4_q8"),
    (10,"regime_scatter_overlay"),(11,"decompression_penalty_delta"),(12,"bandwidth_cost_quant")]]:
    print(f"  • {f}")
