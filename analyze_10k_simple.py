import csv
import math

# Load the data
generations = []
scores = []
with open('metrics.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        generations.append(int(row[0]))
        scores.append(float(row[1]))

print("=== 10,000 Generation CA Experiment Analysis ===\n")

# Basic statistics
def mean(data):
    return sum(data) / len(data)

def std_dev(data):
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def percentile(data, p):
    sorted_data = sorted(data)
    index = int(len(data) * p / 100)
    return sorted_data[index]

print(f"Total generations: {len(scores)}")
print(f"Best score range: [{min(scores):.6f}, {max(scores):.6f}]")
print(f"Mean best score: {mean(scores):.6f}")
print(f"Std dev: {std_dev(scores):.6f}")
print(f"Median: {percentile(scores, 50):.6f}")

# Define time windows for analysis
windows = {
    "Early (0-1000)": (0, 1000),
    "Early-Mid (1000-5000)": (1000, 5000),
    "Mid (5000-20000)": (5000, 20000),
    "Mid-Late (20000-50000)": (20000, 50000),
    "Late (50000-100000)": (50000, 100000)
}

print("\n=== Phase Analysis ===")
phase_stats = {}
for phase_name, (start, end) in windows.items():
    phase_data = scores[start:end]
    mean_val = mean(phase_data)
    std_val = std_dev(phase_data)
    min_val = min(phase_data)
    max_val = max(phase_data)
    phase_stats[phase_name] = {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }
    print(f"\n{phase_name}:")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std dev: {std_val:.6f}")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")
    print(f"  Range: {max_val - min_val:.6f}")

# Compare phases
print("\n=== Phase Comparisons ===")
phases = list(windows.keys())
for i in range(len(phases)-1):
    curr_phase = phases[i]
    next_phase = phases[i+1]
    mean_change = phase_stats[next_phase]['mean'] - phase_stats[curr_phase]['mean']
    std_change = phase_stats[next_phase]['std'] - phase_stats[curr_phase]['std']
    print(f"\n{curr_phase} → {next_phase}:")
    print(f"  Mean change: {mean_change:+.6f}")
    print(f"  Std change: {std_change:+.6f}")

# Detect major jumps
print("\n=== Major Score Jumps ===")
jumps = []
for i in range(1, len(scores)):
    jump = abs(scores[i] - scores[i-1])
    if jump > 0.1:  # Threshold for significant jump
        jumps.append((i, scores[i-1], scores[i], jump))

jumps.sort(key=lambda x: x[3], reverse=True)
print(f"Found {len(jumps)} jumps > 0.1")
print("Top 20 jumps:")
for i, (gen, prev_score, curr_score, jump_size) in enumerate(jumps[:20]):
    print(f"  {i+1}. Gen {gen}: {prev_score:.4f} → {curr_score:.4f} (jump: {jump_size:.4f})")

# Long-term trends using simple linear regression
print("\n=== Long-term Trends ===")
def simple_linear_regression(x_data, y_data):
    n = len(x_data)
    x_mean = mean(x_data)
    y_mean = mean(y_data)
    
    numerator = sum((x_data[i] - x_mean) * (y_data[i] - y_mean) for i in range(n))
    denominator = sum((x_data[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return 0, y_mean
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

segments = [(0, 10000), (10000, 50000), (50000, 100000)]
for start, end in segments:
    x = list(range(start, end))
    y = scores[start:end]
    slope, intercept = simple_linear_regression(x, y)
    print(f"\nGenerations {start}-{end}:")
    print(f"  Trend: {'increasing' if slope > 0 else 'decreasing'} (slope: {slope:.2e})")
    print(f"  Projected change over segment: {slope * (end - start):.6f}")

# Compare with 1000 generation results
print("\n=== Comparison with 1000 Generation Results ===")
early_1k = scores[:1000]
print(f"\nFirst 1000 generations:")
print(f"  Mean: {mean(early_1k):.6f}")
print(f"  Std: {std_dev(early_1k):.6f}")
print(f"  Final score at gen 1000: {scores[999]:.6f}")
print(f"\nFull 100k generations:")
print(f"  Mean: {mean(scores):.6f}")
print(f"  Std: {std_dev(scores):.6f}")
print(f"  Final score at gen 100000: {scores[99999]:.6f}")
print(f"\nDifferences:")
print(f"  Mean increase: {mean(scores) - mean(early_1k):.6f}")
print(f"  Std increase: {std_dev(scores) - std_dev(early_1k):.6f}")

# Stability analysis
print("\n=== Stability Analysis ===")
# Calculate coefficient of variation
cv_windows = [(0, 1000), (0, 10000), (0, 50000), (0, 100000)]
for start, end in cv_windows:
    data = scores[start:end]
    m = mean(data)
    cv = std_dev(data) / m if m != 0 else float('inf')
    print(f"Coefficient of variation (0-{end}): {cv:.4f}")

# Find periods of low variance
print("\n=== Low Variance Periods ===")
window_size = 1000
step = 500
low_var_periods = []
for i in range(0, len(scores) - window_size, step):
    window_data = scores[i:i+window_size]
    var = std_dev(window_data)
    if var < 0.05:  # Threshold for low variance
        low_var_periods.append((i, i+window_size, var))

print(f"Found {len(low_var_periods)} low variance periods (std < 0.05):")
for start, end, var in low_var_periods[:10]:
    print(f"  Generations {start}-{end}: std = {var:.6f}")

# Emergence at long timescales
print("\n=== Emergent Behaviors at Long Timescales ===")
# Check for new score ranges appearing over time
checkpoints = [1000, 5000, 10000, 25000, 50000, 75000, 100000]
print("\nProgressive min/max evolution:")
for i, checkpoint in enumerate(checkpoints):
    subset = scores[:checkpoint]
    print(f"\nUp to generation {checkpoint}:")
    print(f"  Min: {min(subset):.6f}")
    print(f"  Max: {max(subset):.6f}")
    print(f"  Range: {max(subset) - min(subset):.6f}")
    
    # Count approximate unique values
    rounded = [round(s, 4) for s in subset]
    unique_count = len(set(rounded))
    print(f"  Unique values (4 decimals): {unique_count}")

# Periodicity analysis - simple approach
print("\n=== Simple Periodicity Check ===")
# Check for repeating patterns at different scales
periods_to_check = [10, 50, 100, 500, 1000, 5000]
for period in periods_to_check:
    if period * 2 < len(scores):
        # Calculate correlation between values separated by 'period'
        pairs = [(scores[i], scores[i+period]) for i in range(len(scores)-period)]
        if len(pairs) > 0:
            # Simple correlation coefficient
            x_vals = [p[0] for p in pairs]
            y_vals = [p[1] for p in pairs]
            x_mean = mean(x_vals)
            y_mean = mean(y_vals)
            
            cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / len(pairs)
            x_std = std_dev(x_vals)
            y_std = std_dev(y_vals)
            
            if x_std > 0 and y_std > 0:
                corr = cov / (x_std * y_std)
                print(f"  Period {period}: correlation = {corr:.4f}")

# Information content
print("\n=== Score Distribution Analysis ===")
# Create histogram manually
n_bins = 50
bin_width = (max(scores) - min(scores)) / n_bins
bins = [0] * n_bins

for score in scores:
    bin_index = int((score - min(scores)) / bin_width)
    if bin_index >= n_bins:
        bin_index = n_bins - 1
    bins[bin_index] += 1

# Find most common score ranges
print("\nMost frequent score ranges:")
bin_info = [(i, count, min(scores) + i*bin_width, min(scores) + (i+1)*bin_width) 
            for i, count in enumerate(bins)]
bin_info.sort(key=lambda x: x[1], reverse=True)

for i, (bin_idx, count, low, high) in enumerate(bin_info[:10]):
    percentage = (count / len(scores)) * 100
    print(f"  {i+1}. Range [{low:.4f}, {high:.4f}]: {count} occurrences ({percentage:.1f}%)")

print("\n=== Summary of Key Findings ===")
print(f"\n1. The system evolves over {len(scores)} generations")
print(f"2. Score range: [{min(scores):.6f}, {max(scores):.6f}]")
print(f"3. Overall mean: {mean(scores):.6f}, std: {std_dev(scores):.6f}")
print(f"4. The system shows {len(jumps)} major transitions (jumps > 0.1)")
print(f"5. Found {len(low_var_periods)} periods of stability (low variance)")
print(f"6. Long-term trend shows {'overall increase' if scores[-1] > scores[0] else 'overall decrease'}")
print(f"7. Final score ({scores[-1]:.6f}) vs initial ({scores[0]:.6f}): "
      f"{'increased' if scores[-1] > scores[0] else 'decreased'} by {abs(scores[-1] - scores[0]):.6f}")

# Save summary statistics to file
with open('10k_analysis_summary.txt', 'w') as f:
    f.write("=== 10,000 Generation CA Experiment Summary ===\n\n")
    f.write(f"Total generations: {len(scores)}\n")
    f.write(f"Score range: [{min(scores):.6f}, {max(scores):.6f}]\n")
    f.write(f"Mean: {mean(scores):.6f}\n")
    f.write(f"Std dev: {std_dev(scores):.6f}\n")
    f.write(f"Initial score: {scores[0]:.6f}\n")
    f.write(f"Final score: {scores[-1]:.6f}\n")
    f.write(f"\nPhase Statistics:\n")
    for phase_name, stats in phase_stats.items():
        f.write(f"\n{phase_name}:\n")
        f.write(f"  Mean: {stats['mean']:.6f}\n")
        f.write(f"  Std: {stats['std']:.6f}\n")
        f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")

print("\n=== Analysis complete. Summary saved to 10k_analysis_summary.txt ===")