import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('m2_performance_results.csv')

# Since you have 96k samples, let's smooth the data (Window of 10 samples)
df['Power_Smooth'] = df['Power_mW'].rolling(window=10).mean()
df['GPU_Smooth'] = df['GPU_Residency_Pct'].rolling(window=10).mean()

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot Power
ax1.set_xlabel('Sample Index (Time)')
ax1.set_ylabel('Combined Power (mW)', color='tab:red')
ax1.plot(df.index, df['Power_Smooth'], color='tab:red', alpha=0.8, label='SoC Power (Smooth)')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Second axis for GPU
ax2 = ax1.twinx()
ax2.set_ylabel('GPU Residency (%)', color='tab:blue')
ax2.plot(df.index, df['GPU_Smooth'], color='tab:blue', alpha=0.6, label='GPU Usage')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Highlight the Peak
peak_val = df['Power_mW'].max()
plt.title(f'M2 Unified Bottleneck Analysis\nPeak SoC Power: {peak_val} mW', fontsize=14)
fig.tight_layout()

plt.savefig('m2_vlm_analysis_high_res.png', dpi=300)
print("High-res graph saved!")
plt.show()