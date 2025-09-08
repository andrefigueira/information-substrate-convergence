import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
data = []
with open('metrics.csv', 'r') as f:
    next(f)  # Skip header
    for line in f:
        gen, score = line.strip().split(',')
        data.append((int(gen), float(score)))

generations = np.array([d[0] for d in data])
scores = np.array([d[1] for d in data])

print("=== 10,000 Generation CA Experiment Analysis ===\n")

# Basic statistics
print(f"Total generations: {len(scores)}")
print(f"Best score range: [{scores.min():.6f}, {scores.max():.6f}]")
print(f"Mean best score: {scores.mean():.6f}")
print(f"Std dev: {scores.std():.6f}")

# Define time windows for analysis
windows = {
    "Early (0-1000)": (0, 1000),
    "Early-Mid (1000-5000)": (1000, 5000),
    "Mid (5000-20000)": (5000, 20000),
    "Mid-Late (20000-50000)": (20000, 50000),
    "Late (50000-100000)": (50000, 100000)
}

print("\n=== Phase Analysis ===")
phase_stats = []
for phase_name, (start, end) in windows.items():
    phase_data = scores[start:end]
    mean_val = phase_data.mean()
    std_val = phase_data.std()
    min_val = phase_data.min()
    max_val = phase_data.max()
    phase_stats.append((phase_name, mean_val, std_val, min_val, max_val))
    print(f"\n{phase_name}:")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std dev: {std_val:.6f}")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")

# Detect major transitions using moving window
print("\n=== Major Transitions ===")
window_size = 100
# Calculate rolling mean manually
rolling_mean = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
# Calculate differences
mean_diff = np.diff(rolling_mean, n=window_size)
threshold = 3 * np.std(mean_diff[~np.isnan(mean_diff)])

significant_changes = []
for i in range(len(mean_diff)):
    if abs(mean_diff[i]) > threshold:
        actual_gen = i + window_size
        if actual_gen < len(scores):
            significant_changes.append((actual_gen, scores[actual_gen], mean_diff[i]))

print(f"Detected {len(significant_changes)} major transitions")
print("First 20 transitions:")
for gen, score, change in significant_changes[:20]:
    print(f"  Generation {gen}: score {score:.6f} (mean change: {change:.6f})")

# Analyze periodicity using autocorrelation
print("\n=== Periodicity Analysis ===")
# Sample data for efficiency
sample_rate = 10
sampled_scores = scores[::sample_rate]
# Compute autocorrelation manually
n = len(sampled_scores)
max_lag = min(1000, n//4)
autocorr = []

mean_score = sampled_scores.mean()
var_score = ((sampled_scores - mean_score) ** 2).sum()

for lag in range(1, max_lag):
    c = ((sampled_scores[:-lag] - mean_score) * (sampled_scores[lag:] - mean_score)).sum()
    autocorr.append(c / var_score)

# Find peaks in autocorrelation
peaks = []
for i in range(1, len(autocorr)-1):
    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.1:
        peaks.append(((i+1)*sample_rate, autocorr[i]))

print(f"Potential cycles detected at periods (generations):")
peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)[:10]
for period, corr in peaks_sorted:
    print(f"  Period ~{period} generations (correlation: {corr:.3f})")

# Long-term trends
print("\n=== Long-term Trends ===")
segments = [(0, 10000), (10000, 50000), (50000, 100000)]
for start, end in segments:
    x = np.arange(start, end)
    y = scores[start:end]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"\nGenerations {start}-{end}:")
    print(f"  Trend: {'increasing' if slope > 0 else 'decreasing'} (slope: {slope:.2e})")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.2e}")

# Compare with 1000 generation results
print("\n=== Comparison with 1000 Generation Results ===")
early_1k = scores[:1000]
print(f"\nFirst 1000 generations:")
print(f"  Mean: {early_1k.mean():.6f}")
print(f"  Final score at gen 1000: {scores[999]:.6f}")
print(f"\nFull 100k generations:")
print(f"  Mean: {scores.mean():.6f}")
print(f"  Final score at gen 100000: {scores[99999]:.6f}")

# Stability analysis
print("\n=== Stability Analysis ===")
cv_windows = [(0, 1000), (0, 10000), (0, 50000), (0, 100000)]
for start, end in cv_windows:
    data = scores[start:end]
    mean_val = data.mean()
    cv = data.std() / mean_val if mean_val != 0 else np.inf
    print(f"Coefficient of variation (0-{end}): {cv:.4f}")

# Find stable periods
window = 1000
rolling_std = np.array([scores[i:i+window].std() for i in range(0, len(scores)-window, 100)])
rolling_mean_stable = np.array([scores[i:i+window].mean() for i in range(0, len(scores)-window, 100)])
rolling_cv = rolling_std / (rolling_mean_stable + 1e-10)

stable_indices = np.where(rolling_cv < 0.1)[0] * 100  # Multiply by 100 due to step
print(f"\nFound {len(stable_indices)} stable points (CV < 0.1)")

# Emergence of new behaviors
print("\n=== Emergent Behaviors at Long Timescales ===")
# Compare statistics across major time divisions
divisions = [1000, 10000, 50000, 100000]
print("\nProgressive statistics:")
for i, div in enumerate(divisions):
    subset = scores[:div]
    print(f"\n0-{div} generations:")
    print(f"  Mean: {subset.mean():.6f}")
    print(f"  Std: {subset.std():.6f}")
    print(f"  Unique values (approx): {len(np.unique(np.round(subset, 4)))}")
    
    # Check for new score ranges appearing
    if i > 0:
        prev_div = divisions[i-1]
        prev_subset = scores[:prev_div]
        new_max = subset.max() - prev_subset.max()
        new_min = prev_subset.min() - subset.min()
        print(f"  New max increase: {new_max:.6f}")
        print(f"  New min decrease: {new_min:.6f}")

# Create visualizations
fig, axes = plt.subplots(4, 1, figsize=(12, 14))

# Full evolution
ax1 = axes[0]
ax1.plot(generations, scores, linewidth=0.5, alpha=0.7)
ax1.set_title('Full 100,000 Generation Evolution')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Best Score')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 0.6)

# Log scale view
ax2 = axes[1]
ax2.plot(generations[1:], scores[1:], linewidth=0.5, alpha=0.7)  # Skip 0 for log scale
ax2.set_xscale('log')
ax2.set_title('Evolution (Log Scale)')
ax2.set_xlabel('Generation (log scale)')
ax2.set_ylabel('Best Score')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 0.6)

# Phase means
ax3 = axes[2]
phase_means = [stats[1] for stats in phase_stats]
phase_labels = [stats[0].split()[0] for stats in phase_stats]
bars = ax3.bar(range(len(phase_means)), phase_means)
ax3.set_xticks(range(len(phase_means)))
ax3.set_xticklabels(phase_labels, rotation=45)
ax3.set_title('Mean Score by Phase')
ax3.set_ylabel('Mean Best Score')
ax3.grid(True, alpha=0.3)

# Add error bars for std dev
phase_stds = [stats[2] for stats in phase_stats]
ax3.errorbar(range(len(phase_means)), phase_means, yerr=phase_stds, 
             fmt='none', color='black', capsize=5)

# Rolling statistics
ax4 = axes[3]
window = 1000
step = 100
rolling_indices = range(0, len(scores)-window, step)
rolling_means = [scores[i:i+window].mean() for i in rolling_indices]
rolling_stds = [scores[i:i+window].std() for i in rolling_indices]

ax4.plot(rolling_indices, rolling_means, label='Rolling Mean', alpha=0.8)
ax4.fill_between(rolling_indices, 
                 np.array(rolling_means) - np.array(rolling_stds),
                 np.array(rolling_means) + np.array(rolling_stds),
                 alpha=0.3, label='±1 Std Dev')
ax4.set_title(f'Rolling Statistics (window={window})')
ax4.set_xlabel('Generation')
ax4.set_ylabel('Best Score')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('10k_analysis.png', dpi=150)
plt.close()

# Create distribution comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram comparison
phases_to_compare = ["Early (0-1000)", "Mid (5000-20000)", "Late (50000-100000)"]
colors = ['blue', 'green', 'red']
for i, phase_name in enumerate(phases_to_compare):
    start, end = windows[phase_name]
    phase_data = scores[start:end]
    ax1.hist(phase_data, bins=50, alpha=0.5, label=phase_name, 
             density=True, color=colors[i])

ax1.set_xlabel('Best Score')
ax1.set_ylabel('Density')
ax1.set_title('Score Distribution by Phase')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Violin plot for phase comparison
phase_data_list = []
phase_labels_short = []
for phase_name, (start, end) in windows.items():
    phase_data_list.append(scores[start:end])
    phase_labels_short.append(phase_name.split()[0])

parts = ax2.violinplot(phase_data_list, positions=range(len(phase_data_list)),
                       showmeans=True, showmedians=True)
ax2.set_xticks(range(len(phase_labels_short)))
ax2.set_xticklabels(phase_labels_short, rotation=45)
ax2.set_ylabel('Best Score')
ax2.set_title('Score Distribution Violin Plot')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('10k_distributions.png', dpi=150)
plt.close()

print("\n=== Analysis complete. Plots saved as 10k_analysis.png and 10k_distributions.png ===")

# Additional analysis: Calculate entropy-like measure
print("\n=== Information Content Analysis ===")
# Discretize scores into bins and calculate entropy
n_bins = 100
hist, bin_edges = np.histogram(scores, bins=n_bins)
probs = hist / hist.sum()
probs = probs[probs > 0]  # Remove zero probabilities
entropy = -np.sum(probs * np.log2(probs))
print(f"Overall entropy (100 bins): {entropy:.4f} bits")

# Calculate entropy for each phase
for phase_name, (start, end) in windows.items():
    phase_data = scores[start:end]
    hist, _ = np.histogram(phase_data, bins=50)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    print(f"{phase_name}: {entropy:.4f} bits")