"""
M2 Unified Memory Bottleneck Analysis
Research Objectives:
  RO1: When does the memory bus saturate during multimodal inference?
  RO2: How does the GPU ALU vs Memory-Bandwidth ratio shift from vision encoding to text gen?
  RO3: Is there a measurable ALU decompression penalty from 4-bit weight dequant?
"""

import re
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─────────────────────────────────────────────
# 1.  PARSER
# ─────────────────────────────────────────────

def parse_powermetrics(path, label=""):
    """Stream-parse a powermetrics log. Returns a DataFrame."""
    print(f"  Parsing {label} ({os.path.getsize(path) // 1_048_576} MB)...")
    data = []
    entry = {}

    ts_re          = re.compile(r"Sampled system activity \((.+?)\)")
    cpu_pow_re     = re.compile(r"^CPU Power:\s*(\d+)\s*mW")
    gpu_pow_re     = re.compile(r"^GPU Power:\s*(\d+)\s*mW")
    ane_pow_re     = re.compile(r"^ANE Power:\s*(\d+)\s*mW")
    combined_re    = re.compile(r"Combined Power.*?:\s*(\d+)\s*mW")
    gpu_res_re     = re.compile(r"GPU HW active residency:\s*([\d.]+)%")
    gpu_freq_re    = re.compile(r"GPU HW active frequency:\s*(\d+)\s*MHz")
    gpu_sw_p1_re   = re.compile(r"GPU SW requested state:.*?P1\s*:\s*([\d.]+)%")
    p_clust_res_re = re.compile(r"P-Cluster HW active residency:\s*([\d.]+)%")
    e_clust_res_re = re.compile(r"E-Cluster HW active residency:\s*([\d.]+)%")
    p_clust_frq_re = re.compile(r"P-Cluster HW active frequency:\s*(\d+)\s*MHz")
    thermal_re     = re.compile(r"Current pressure level:\s*(\w+)")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()

            m = ts_re.search(line)
            if m:
                if "ts" in entry and "combined_mw" in entry:
                    data.append(entry)
                entry = {"ts": m.group(1)}
                continue

            if m := cpu_pow_re.match(line):
                entry["cpu_mw"] = int(m.group(1))
            elif m := gpu_pow_re.match(line):
                # powermetrics prints GPU Power twice; we want the standalone line
                if "combined_mw" not in entry:
                    entry["gpu_mw"] = int(m.group(1))
            elif m := ane_pow_re.match(line):
                entry["ane_mw"] = int(m.group(1))
            elif m := combined_re.search(line):
                entry["combined_mw"] = int(m.group(1))
                # also capture the gpu_mw that appears on the combined line if not yet set
                # (the "GPU Power: X mW" after GPU Usage section)
            elif m := gpu_res_re.search(line):
                if "gpu_residency" not in entry:
                    entry["gpu_residency"] = float(m.group(1))
            elif m := gpu_freq_re.search(line):
                if "gpu_freq_mhz" not in entry:
                    entry["gpu_freq_mhz"] = int(m.group(1))
            elif m := gpu_sw_p1_re.search(line):
                entry["gpu_sw_p1_pct"] = float(m.group(1))
            elif m := p_clust_res_re.search(line):
                entry["p_cluster_pct"] = float(m.group(1))
            elif m := e_clust_res_re.search(line):
                entry["e_cluster_pct"] = float(m.group(1))
            elif m := p_clust_frq_re.search(line):
                entry["p_cluster_mhz"] = int(m.group(1))
            elif m := thermal_re.search(line):
                entry["thermal"] = m.group(1)

    if "ts" in entry and "combined_mw" in entry:
        data.append(entry)

    df = pd.DataFrame(data)
    numeric_cols = ["cpu_mw", "gpu_mw", "ane_mw", "combined_mw",
                    "gpu_residency", "gpu_freq_mhz", "gpu_sw_p1_pct",
                    "p_cluster_pct", "e_cluster_pct", "p_cluster_mhz"]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["sample_idx"] = range(len(df))
    df["label"] = label
    print(f"    → {len(df):,} samples  |  Peak combined: {df['combined_mw'].max():,} mW")
    return df


# ─────────────────────────────────────────────
# 2.  PHASE DETECTION
# ─────────────────────────────────────────────

def detect_phases(df, gpu_thresh=15.0):
    """
    Label each sample as 'vision_encoding', 'text_generation', or 'idle'.
    Heuristic: vision encoding → sustained GPU residency above threshold.
    """
    # smooth residency to avoid noise-triggered flips
    smooth = df["gpu_residency"].rolling(5, min_periods=1, center=True).mean()
    df = df.copy()
    df["phase"] = "text_generation"
    df.loc[smooth >= gpu_thresh, "phase"] = "vision_encoding"
    df.loc[df["combined_mw"] < 300, "phase"] = "idle"
    return df


# ─────────────────────────────────────────────
# 3.  BANDWIDTH SATURATION METRIC
# ─────────────────────────────────────────────

def bandwidth_proxy(df):
    """
    Proxy for memory-bus utilization: combined_mw normalised to [0,1].
    The M2 8GB memory bus tops out around 68.25 GB/s; peak power ≈ peak BW pressure.
    We use a rolling 95th-pct envelope as the 'saturation ceiling'.
    """
    df = df.copy()
    window = 20
    df["bw_proxy_smooth"] = df["combined_mw"].rolling(window, min_periods=1).mean()
    df["bw_running_p95"]  = df["combined_mw"].expanding().quantile(0.95)
    df["bw_saturation"]   = df["bw_proxy_smooth"] / df["bw_running_p95"].clip(lower=1)
    return df


# ─────────────────────────────────────────────
# 4.  LOAD DATA
# ─────────────────────────────────────────────

base = os.path.dirname(os.path.abspath(__file__))
logs = os.path.join(base, "../data/processed_csvs", "logs")

files = {
    "Text-Only Baseline\n(pure text gen, no vision)":
        os.path.join(logs, "m2_text_baseline_20260419_005117.txt"),
    "Vision+Text (single image)\n(bench.zsh)":
        os.path.join(logs, "m2_trace_20260225_132943.txt"),
    "Video Inference (5 frames)\n(bench-vid.zsh)":
        os.path.join(logs, "m2_video_trace_20260419_001300.txt"),
}

print("Loading logs...")
datasets = {}
for label, path in files.items():
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = parse_powermetrics(path, label=label)
        df = detect_phases(df)
        df = bandwidth_proxy(df)
        datasets[label] = df
    else:
        print(f"  SKIP (missing/empty): {path}")

print(f"\nDatasets loaded: {list(datasets.keys())}\n")


# ─────────────────────────────────────────────
# 5.  PRINT SUMMARY STATISTICS
# ─────────────────────────────────────────────

PHASE_COLORS = {
    "vision_encoding":  "#e74c3c",
    "text_generation":  "#3498db",
    "idle":             "#95a5a6",
}

print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

for label, df in datasets.items():
    print(f"\n── {label.replace(chr(10),' | ')} ──")
    print(f"   Total samples  : {len(df):>7,}  (~{len(df)*0.5/60:.1f} min)")
    print(f"   Peak combined  : {df['combined_mw'].max():>7,} mW")
    print(f"   Mean combined  : {df['combined_mw'].mean():>7,.0f} mW")
    for phase in ["vision_encoding", "text_generation", "idle"]:
        sub = df[df["phase"] == phase]
        if len(sub) == 0:
            continue
        print(f"\n   [{phase}]  n={len(sub):,}")
        print(f"     CPU Power   : mean={sub['cpu_mw'].mean():.0f} mW  peak={sub['cpu_mw'].max():.0f} mW")
        print(f"     GPU Power   : mean={sub['gpu_mw'].mean():.0f} mW  peak={sub['gpu_mw'].max():.0f} mW")
        ane_col = sub["ane_mw"] if "ane_mw" in sub.columns else pd.Series([0]*len(sub))
        print(f"     ANE Power   : mean={sub['ane_mw'].mean():.0f} mW  peak={sub['ane_mw'].max():.0f} mW")
        print(f"     GPU Residency: mean={sub['gpu_residency'].mean():.1f}%  peak={sub['gpu_residency'].max():.1f}%")
        active_gpu = sub[sub["gpu_freq_mhz"] > 0]
        if len(active_gpu):
            print(f"     GPU Freq (when active): mean={active_gpu['gpu_freq_mhz'].mean():.0f} MHz  "
                  f"peak={active_gpu['gpu_freq_mhz'].max():.0f} MHz")
        sw_p1 = sub[sub["gpu_sw_p1_pct"] > 0]
        if len(sw_p1):
            print(f"     GPU SW P1 (max-perf req'd): mean={sw_p1['gpu_sw_p1_pct'].mean():.1f}%")

# ─────────────────────────────────────────────
# 6.  FIGURE 1: POWER TIMELINE + PHASE BANDS
#     RO1 – Bandwidth Saturation
# ─────────────────────────────────────────────

print("\nGenerating Figure 1: Power Timeline with Phase Detection...")

n_plots = len(datasets)
fig, axes = plt.subplots(n_plots, 1, figsize=(18, 5 * n_plots), sharex=False)
if n_plots == 1:
    axes = [axes]

fig.suptitle(
    "M2 SoC Power Timeline — Bandwidth Saturation Analysis (RO1)\n"
    "Red bands = Vision Encoding (compute-heavy)  |  Blue bands = Text Generation (memory-bound)",
    fontsize=13, fontweight="bold", y=1.01
)

for ax, (label, df) in zip(axes, datasets.items()):
    smooth = 6
    idx = df["sample_idx"]

    # Phase shading
    for phase, color in PHASE_COLORS.items():
        mask = df["phase"] == phase
        if not mask.any():
            continue
        starts = idx[mask & ~mask.shift(1, fill_value=False)].values
        ends   = idx[mask & ~mask.shift(-1, fill_value=False)].values
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=0.12, color=color, linewidth=0)

    # Power traces
    ax.plot(idx, df["cpu_mw"].rolling(smooth).mean(), color="#e74c3c", lw=1.0, alpha=0.85, label="CPU Power")
    ax.plot(idx, df["gpu_mw"].rolling(smooth).mean(), color="#27ae60", lw=1.5, alpha=0.90, label="GPU Power")
    if df["ane_mw"].max() > 0:
        ax.plot(idx, df["ane_mw"].rolling(smooth).mean(), color="#9b59b6", lw=1.0, alpha=0.80, label="ANE Power")
    ax.plot(idx, df["combined_mw"].rolling(smooth).mean(), color="black", lw=1.8, alpha=0.70, label="Combined Power")

    # Saturation marker: 95th pct of combined
    sat_line = df["combined_mw"].quantile(0.95)
    ax.axhline(sat_line, color="orange", ls="--", lw=1.4, label=f"95th pct ({sat_line:.0f} mW) ≈ BW ceiling")

    ax.set_title(label.replace("\n", " | "), fontsize=11)
    ax.set_ylabel("Power (mW)")
    ax.set_xlabel("Sample Index  (~0.5s per sample)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    peak_i = df["combined_mw"].idxmax()
    ax.annotate(
        f"Peak {df['combined_mw'].max():.0f} mW",
        xy=(df.loc[peak_i, "sample_idx"], df.loc[peak_i, "combined_mw"]),
        xytext=(15, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"), fontsize=9
    )

plt.tight_layout()
out1 = os.path.join(base, "../data/processed_csvs", "fig1_power_timeline_phases.png")
plt.savefig(out1, dpi=200, bbox_inches="tight")
print(f"  Saved → {out1}")
plt.close()


# ─────────────────────────────────────────────
# 7.  FIGURE 2: GPU FREQ DISTRIBUTION BY PHASE
#     RO2 + RO3 – ALU vs Memory Bandwidth, Decompression Penalty
# ─────────────────────────────────────────────

print("Generating Figure 2: GPU Frequency Distribution by Phase...")

# Merge all datasets for a combined view
all_df = pd.concat(datasets.values(), ignore_index=True)
# only consider samples where GPU is actually active
active = all_df[all_df["gpu_freq_mhz"] > 0].copy()

freq_bins = [444, 612, 808, 968, 1110, 1236, 1338, 1398, 1500]  # M2 GPU P-states

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "GPU Operating Frequency Distribution by Inference Phase\n"
    "RO2: ALU vs Memory-BW Bottleneck Shift  |  RO3: 4-bit Decompression Penalty Proxy",
    fontsize=12, fontweight="bold"
)

for ax, phase in zip(axes, ["vision_encoding", "text_generation"]):
    sub = active[active["phase"] == phase]
    if len(sub) == 0:
        ax.text(0.5, 0.5, f"No {phase} samples", ha="center", va="center")
        continue

    freqs = sub["gpu_freq_mhz"].values
    ax.hist(freqs, bins=30, color=PHASE_COLORS[phase], alpha=0.75, edgecolor="white", linewidth=0.5)

    ax.axvline(freqs.mean(), color="black", ls="--", lw=1.8, label=f"Mean: {freqs.mean():.0f} MHz")
    ax.axvline(444, color="red", ls=":", lw=1.5, label="444 MHz (min / mem-stall P-state)")

    pct_min = (freqs <= 444).mean() * 100
    ax.text(0.97, 0.95, f"{pct_min:.1f}% of active\nsamples at 444 MHz\n(memory-stall proxy)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    label_str = "Vision Encoding" if phase == "vision_encoding" else "Text Generation"
    ax.set_title(f"{label_str}  (n={len(sub):,})", fontsize=11)
    ax.set_xlabel("GPU Active Frequency (MHz)")
    ax.set_ylabel("Sample Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = os.path.join(base, "../data/processed_csvs", "fig2_gpu_freq_distribution.png")
plt.savefig(out2, dpi=200, bbox_inches="tight")
print(f"  Saved → {out2}")
plt.close()


# ─────────────────────────────────────────────
# 8.  FIGURE 3: ALU vs MEMORY-BW SCATTER
#     RO2 – Bottleneck Regime Mapping
# ─────────────────────────────────────────────

print("Generating Figure 3: ALU vs Memory-BW Scatter...")

fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 6))
if len(datasets) == 1:
    axes = [axes]

fig.suptitle(
    "GPU Bottleneck Regime Map: Residency × Frequency\n"
    "RO2 — Upper-right = ALU-bound (compute)  |  Lower-left = Memory-BW-bound",
    fontsize=12, fontweight="bold"
)

for ax, (label, df) in zip(axes, datasets.items()):
    active_df = df[df["gpu_freq_mhz"] > 0]

    for phase, color in PHASE_COLORS.items():
        sub = active_df[active_df["phase"] == phase]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["gpu_residency"], sub["gpu_freq_mhz"],
            c=color, alpha=0.35, s=8, label=phase.replace("_", " ").title()
        )

    # Quadrant annotations
    ax.axvline(30, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(800, color="gray", ls="--", lw=0.8, alpha=0.6)
    for txt, (rx, ry) in [
        ("ALU-Bound\n(High compute)", (65, 1100)),
        ("Memory-BW Bound\n(dequant / KV load)", (5, 550)),
        ("Idle / min power", (5, 200)),
        ("High BW\n(partial compute)", (65, 550)),
    ]:
        ax.text(rx, ry, txt, ha="center", va="center", fontsize=7.5,
                color="gray", style="italic")

    ax.set_title(label.replace("\n", " | "), fontsize=10)
    ax.set_xlabel("GPU HW Active Residency (%)")
    ax.set_ylabel("GPU HW Active Frequency (MHz)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 1550)
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
out3 = os.path.join(base, "../data/processed_csvs", "fig3_alu_vs_bw_scatter.png")
plt.savefig(out3, dpi=200, bbox_inches="tight")
print(f"  Saved → {out3}")
plt.close()


# ─────────────────────────────────────────────
# 9.  FIGURE 4: MEMORY BUS SATURATION CURVE
#     RO1 – Exact saturation onset per workload
# ─────────────────────────────────────────────

print("Generating Figure 4: Bandwidth Saturation Curves...")

fig, axes = plt.subplots(n_plots, 1, figsize=(18, 4 * n_plots), sharex=False)
if n_plots == 1:
    axes = [axes]

fig.suptitle(
    "Memory Bus Saturation Progression (RO1)\n"
    "Saturation Index = rolling-mean power / expanding-95th-pct power  (1.0 = bus ceiling)",
    fontsize=12, fontweight="bold"
)

for ax, (label, df) in zip(axes, datasets.items()):
    idx = df["sample_idx"]
    ax.fill_between(idx, df["bw_saturation"].clip(0, 1.3),
                    alpha=0.35, color="#e67e22", label="Saturation index")
    ax.plot(idx, df["bw_saturation"].rolling(10).mean(),
            color="#e67e22", lw=1.5)
    ax.axhline(1.0, color="red", ls="--", lw=1.5, label="Saturation ceiling (1.0)")
    ax.axhline(0.8, color="orange", ls=":", lw=1.2, label="80% saturation")

    # mark first crossing of 0.8 threshold
    cross = df[df["bw_saturation"] >= 0.8]
    if len(cross):
        first_i = cross.iloc[0]["sample_idx"]
        ax.axvline(first_i, color="purple", ls="--", lw=1.2)
        ax.text(first_i + 5, 1.15,
                f"80% sat. onset\n@ sample {int(first_i)}\n(~{first_i*0.5:.0f}s)",
                fontsize=8, color="purple")

    ax.set_title(label.replace("\n", " | "), fontsize=10)
    ax.set_ylabel("Saturation Index")
    ax.set_xlabel("Sample Index")
    ax.set_ylim(0, 1.35)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out4 = os.path.join(base, "../data/processed_csvs", "fig4_bandwidth_saturation.png")
plt.savefig(out4, dpi=200, bbox_inches="tight")
print(f"  Saved → {out4}")
plt.close()


# ─────────────────────────────────────────────
# 10. FIGURE 5: PHASE TRANSITION DEEP DIVE
#     RO2 – Metal Compute Transition
#     Focus on VIDEO trace only (has clearest phase transitions)
# ─────────────────────────────────────────────

print("Generating Figure 5: Phase Transition Deep Dive (video trace)...")

video_key = [k for k in datasets if "Video" in k or "video" in k.lower()]
vision_key = [k for k in datasets if "Vision" in k or "single" in k.lower() or "bench.zsh" in k]
focus_key  = video_key[0] if video_key else (vision_key[0] if vision_key else list(datasets.keys())[-1])
focus_df   = datasets[focus_key]

fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: full GPU residency timeline with phase coloring
ax_a = fig.add_subplot(gs[0, :])
smooth_res = focus_df["gpu_residency"].rolling(5).mean()
ax_a.plot(focus_df["sample_idx"], smooth_res, color="#2ecc71", lw=1.2, label="GPU Residency (smoothed)")
for phase, color in PHASE_COLORS.items():
    mask = focus_df["phase"] == phase
    ax_a.fill_between(focus_df["sample_idx"], smooth_res.where(mask, 0),
                      alpha=0.3, color=color, label=phase.replace("_", " ").title())
ax_a.set_ylabel("GPU Residency (%)")
ax_a.set_title(f"GPU Utilisation Over Time — {focus_key.split(chr(10))[0]}", fontsize=11)
ax_a.legend(fontsize=8, ncol=3)
ax_a.grid(True, alpha=0.3)

# Panel B: GPU frequency over time
ax_b = fig.add_subplot(gs[1, :])
ax_b.scatter(focus_df["sample_idx"], focus_df["gpu_freq_mhz"],
             c=[{"vision_encoding": "#e74c3c", "text_generation": "#3498db", "idle": "#95a5a6"}[p]
                for p in focus_df["phase"]],
             s=3, alpha=0.4)
ax_b.axhline(444, color="red", ls=":", lw=1.2, label="444 MHz (memory-stall marker)")
ax_b.set_ylabel("GPU Active Freq (MHz)")
ax_b.set_title("GPU Operating Frequency — High Freq = ALU Work, 444 MHz = Memory Stall (RO2 & RO3)", fontsize=11)
ax_b.legend(fontsize=8)
ax_b.grid(True, alpha=0.3)
ax_b.set_xlabel("Sample Index")

# Panel C: CPU vs GPU power ratio (proxy for compute pipeline ownership)
ax_c = fig.add_subplot(gs[2, 0])
valid = focus_df[focus_df["cpu_mw"] > 0].copy()
valid["gpu_cpu_ratio"] = valid["gpu_mw"] / valid["cpu_mw"].clip(lower=1)
for phase, color in PHASE_COLORS.items():
    sub = valid[valid["phase"] == phase]
    if len(sub) == 0:
        continue
    ax_c.scatter(sub["sample_idx"], sub["gpu_cpu_ratio"],
                 color=color, s=4, alpha=0.4, label=phase.replace("_", " ").title())
ax_c.axhline(1.0, color="gray", ls="--", lw=1.2, label="GPU = CPU power")
ax_c.set_ylabel("GPU/CPU Power Ratio")
ax_c.set_xlabel("Sample Index")
ax_c.set_title("GPU-to-CPU Power Ratio\n>1 = GPU-dominant (ALU), <1 = CPU-dominant (mem-bound)", fontsize=10)
ax_c.legend(fontsize=7)
ax_c.set_ylim(0, 2.5)
ax_c.grid(True, alpha=0.3)

# Panel D: Phase duration breakdown (pie chart)
ax_d = fig.add_subplot(gs[2, 1])
phase_counts = focus_df["phase"].value_counts()
colors_pie   = [PHASE_COLORS.get(p, "gray") for p in phase_counts.index]
ax_d.pie(phase_counts.values, labels=[p.replace("_", " ").title() for p in phase_counts.index],
         colors=colors_pie, autopct="%1.1f%%", startangle=90,
         textprops={"fontsize": 9})
ax_d.set_title("Time-in-Phase Breakdown\n(fraction of benchmark duration)", fontsize=10)

out5 = os.path.join(base, "../data/processed_csvs", "fig5_phase_transition_detail.png")
plt.savefig(out5, dpi=200, bbox_inches="tight")
print(f"  Saved → {out5}")
plt.close()


# ─────────────────────────────────────────────
# 11. FIGURE 6: CROSS-WORKLOAD COMPARISON BAR CHART
# ─────────────────────────────────────────────

print("Generating Figure 6: Cross-Workload Comparison...")

metrics = {
    "Peak Combined\nPower (mW)":    lambda df: df["combined_mw"].max(),
    "Mean GPU\nResidency (%)":       lambda df: df["gpu_residency"].mean(),
    "Mean GPU Freq\n(active, MHz)":  lambda df: df[df["gpu_freq_mhz"] > 0]["gpu_freq_mhz"].mean(),
    "% Time @ 444MHz\n(mem-stall)":  lambda df: (df[df["gpu_freq_mhz"] > 0]["gpu_freq_mhz"] <= 444).mean() * 100,
    "Mean ANE\nPower (mW)":          lambda df: df["ane_mw"].mean(),
}

fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
fig.suptitle("Cross-Workload Hardware Characterisation — Q4_K_M (4-bit) InternVL2.5-4B",
             fontsize=13, fontweight="bold")

short_labels = [l.split("\n")[0].split("(")[0].strip() for l in datasets.keys()]

for ax, (metric_name, fn) in zip(axes, metrics.items()):
    vals = [fn(df) for df in datasets.values()]
    bars = ax.bar(range(len(vals)), vals,
                  color=["#e74c3c", "#3498db", "#27ae60"][:len(vals)],
                  alpha=0.8, edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=8)
    ax.set_title(metric_name, fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.35)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals)*0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out6 = os.path.join(base, "../data/processed_csvs", "fig6_cross_workload_comparison.png")
plt.savefig(out6, dpi=200, bbox_inches="tight")
print(f"  Saved → {out6}")
plt.close()


# ─────────────────────────────────────────────
# 12. EXPORT ENRICHED CSVs
# ─────────────────────────────────────────────

print("\nExporting enriched CSVs...")
for label, df in datasets.items():
    slug = label.split("\n")[0].replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")[:30]
    out_csv = os.path.join(base, "../data/processed_csvs", f"enriched_{slug}.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")

print("\n✓ Analysis complete. All figures saved to results/")
print("\nKey findings to investigate:")
print("  • fig1: Where do power spikes align with vision/text phase boundaries?")
print("  • fig2: What fraction of active GPU samples are stuck at 444 MHz (memory stall)?")
print("  • fig3: Do vision_encoding samples cluster top-right (ALU) vs bottom-left (BW)?")
print("  • fig4: At what sample index does the BW saturation index first hit 0.8?")
print("  • fig5: How long is each frame's vision-encoding burst vs text-gen tail?")
print("  • fig6: Compare mean GPU frequency text-only vs vision — decompression penalty?")
