import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Load the data
df = pd.read_csv('metrics.csv')

print("=== 10,000 Generation CA Experiment Analysis ===\n")

# Basic statistics
print(f"Total generations: {len(df)}")
print(f"Best score range: [{df['best_score'].min():.6f}, {df['best_score'].max():.6f}]")
print(f"Mean best score: {df['best_score'].mean():.6f}")
print(f"Std dev: {df['best_score'].std():.6f}")

# Define time windows for analysis
windows = {
    "Early (0-1000)": (0, 1000),
    "Early-Mid (1000-5000)": (1000, 5000),
    "Mid (5000-20000)": (5000, 20000),
    "Mid-Late (20000-50000)": (20000, 50000),
    "Late (50000-100000)": (50000, 100000)
}

print("\n=== Phase Analysis ===")
for phase_name, (start, end) in windows.items():
    phase_data = df.iloc[start:end]['best_score']
    print(f"\n{phase_name}:")
    print(f"  Mean: {phase_data.mean():.6f}")
    print(f"  Std dev: {phase_data.std():.6f}")
    print(f"  Min: {phase_data.min():.6f}")
    print(f"  Max: {phase_data.max():.6f}")

# Detect major transitions
print("\n=== Major Transitions ===")
# Calculate rolling statistics
window_size = 100
rolling_mean = df['best_score'].rolling(window=window_size).mean()
rolling_std = df['best_score'].rolling(window=window_size).std()

# Find significant changes in mean
mean_diff = rolling_mean.diff(periods=window_size)
threshold = 3 * mean_diff.std()
transitions = df.index[abs(mean_diff) > threshold].tolist()

print(f"Detected {len(transitions)} major transitions at generations:")
for t in transitions[:20]:  # Show first 20
    if t > window_size:
        print(f"  Generation {t}: score {df.loc[t, 'best_score']:.6f} (mean change: {mean_diff[t]:.6f})")

# Analyze cycles/periodicity
print("\n=== Periodicity Analysis ===")
# Use autocorrelation to detect cycles
from statsmodels.tsa.stattools import acf

# Sample data for autocorrelation (every 10th point for efficiency)
sampled_data = df['best_score'].iloc[::10].values
lags = range(1, min(1000, len(sampled_data)//4))
autocorr = acf(sampled_data, nlags=len(lags))

# Find peaks in autocorrelation
peaks = []
for i in range(1, len(autocorr)-1):
    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.1:
        peaks.append((i*10, autocorr[i]))  # Multiply by 10 due to sampling

print(f"Potential cycles detected at periods (generations):")
for period, corr in sorted(peaks[:10], key=lambda x: x[1], reverse=True):
    print(f"  Period ~{period} generations (correlation: {corr:.3f})")

# Long-term trends
print("\n=== Long-term Trends ===")
# Fit linear regression to different segments
segments = [(0, 10000), (10000, 50000), (50000, 100000)]
for start, end in segments:
    x = np.arange(start, end)
    y = df.iloc[start:end]['best_score'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"\nGenerations {start}-{end}:")
    print(f"  Trend: {'increasing' if slope > 0 else 'decreasing'} (slope: {slope:.2e})")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.2e}")

# Compare with 1000 generation results
print("\n=== Comparison with 1000 Generation Results ===")
early_1k = df.iloc[:1000]['best_score']
print(f"\nFirst 1000 generations:")
print(f"  Mean: {early_1k.mean():.6f}")
print(f"  Final score at gen 1000: {df.loc[999, 'best_score']:.6f}")
print(f"\nFull 100k generations:")
print(f"  Mean: {df['best_score'].mean():.6f}")
print(f"  Final score at gen 100000: {df.loc[99999, 'best_score']:.6f}")

# Stability analysis
print("\n=== Stability Analysis ===")
# Calculate coefficient of variation for different windows
cv_windows = [(0, 1000), (0, 10000), (0, 50000), (0, 100000)]
for start, end in cv_windows:
    data = df.iloc[start:end]['best_score']
    cv = data.std() / data.mean() if data.mean() != 0 else np.inf
    print(f"Coefficient of variation (0-{end}): {cv:.4f}")

# Find stable periods (low variance)
window = 1000
rolling_cv = rolling_std / rolling_mean
stable_periods = df.index[rolling_cv < 0.1].tolist()
if stable_periods:
    print(f"\nStable periods (CV < 0.1, {window}-gen window):")
    # Group consecutive indices
    groups = []
    current_group = [stable_periods[0]]
    for i in range(1, len(stable_periods)):
        if stable_periods[i] - stable_periods[i-1] <= window:
            current_group.append(stable_periods[i])
        else:
            groups.append((min(current_group), max(current_group)))
            current_group = [stable_periods[i]]
    groups.append((min(current_group), max(current_group)))
    
    for start, end in groups[:10]:  # Show first 10
        if end - start > 1000:  # Only show significant periods
            print(f"  Generations {start}-{end} (duration: {end-start})")

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Full evolution
ax1 = axes[0]
ax1.plot(df['generation'], df['best_score'], linewidth=0.5, alpha=0.7)
ax1.set_title('Full 100,000 Generation Evolution')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Best Score')
ax1.grid(True, alpha=0.3)

# Log scale view
ax2 = axes[1]
ax2.plot(df['generation'], df['best_score'], linewidth=0.5, alpha=0.7)
ax2.set_xscale('log')
ax2.set_title('Evolution (Log Scale)')
ax2.set_xlabel('Generation (log scale)')
ax2.set_ylabel('Best Score')
ax2.grid(True, alpha=0.3)

# Phase means
ax3 = axes[2]
phase_means = []
phase_labels = []
phase_positions = []
for i, (phase_name, (start, end)) in enumerate(windows.items()):
    phase_data = df.iloc[start:end]['best_score']
    phase_means.append(phase_data.mean())
    phase_labels.append(phase_name.split()[0])  # Just the phase name
    phase_positions.append((start + end) / 2)
    
ax3.bar(range(len(phase_means)), phase_means)
ax3.set_xticks(range(len(phase_means)))
ax3.set_xticklabels(phase_labels, rotation=45)
ax3.set_title('Mean Score by Phase')
ax3.set_ylabel('Mean Best Score')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('10k_analysis.png', dpi=150)
plt.close()

# Create distribution analysis
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for phase_name, (start, end) in list(windows.items())[:3]:  # First 3 phases
    phase_data = df.iloc[start:end]['best_score']
    ax.hist(phase_data, bins=50, alpha=0.5, label=phase_name, density=True)
ax.set_xlabel('Best Score')
ax.set_ylabel('Density')
ax.set_title('Score Distribution by Phase')
ax.legend()
plt.tight_layout()
plt.savefig('10k_distributions.png', dpi=150)
plt.close()

print("\n=== Analysis complete. Plots saved as 10k_analysis.png and 10k_distributions.png ===")