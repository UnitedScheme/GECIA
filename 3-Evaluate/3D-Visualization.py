"""
Data Visualization with Nature Journal Style
===========================================

This script creates publication-quality plots following Nature journal's 
style guidelines. It visualizes comparative data with smooth curves and 
standard deviation regions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import os
from scipy.interpolate import make_interp_spline

# =========================
# 1. Load and Validate Data
# =========================
print("▌Loading data...")
csv_file = 'bs01.csv'

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found in current directory!")
    exit()

df = pd.read_csv(csv_file)
print(f"✔ Data loaded successfully | Shape: {df.shape}")

# =========================
# 2. Configure Nature Style
# =========================
# Check available fonts (prioritizing Nature-recommended fonts)
available_fonts = [f.name for f in fm.fontManager.ttflist]
font_options = [
    'Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans',
    'DejaVu Sans', 'FreeSans', 'Bitstream Vera Sans'
]
selected_font = next((f for f in font_options if f in available_fonts), 'sans-serif')

# Apply Nature journal style parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [selected_font],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 10,
    'axes.labelweight': 'normal',
    'legend.fontsize': 9,
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.grid': False,
    'grid.alpha': 0,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

# =========================
# 3. Data Preparation
# =========================
# Extract data columns
x = df['state_offset'].values
y0_mean = df['0_mean'].values
y0_std = df['0_std'].values
y1_mean = df['1_mean'].values
y1_std = df['1_std'].values

# Sort data for proper smoothing
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y0_sorted = y0_mean[sort_idx]
y1_sorted = y1_mean[sort_idx]
y0_std_sorted = y0_std[sort_idx]
y1_std_sorted = y1_std[sort_idx]

# Create smooth curves using B-spline interpolation
x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
spl0 = make_interp_spline(x_sorted, y0_sorted, k=3)
y0_smooth = spl0(x_smooth)
spl1 = make_interp_spline(x_sorted, y1_sorted, k=3)
y1_smooth = spl1(x_smooth)

# =========================
# 4. Plot Configuration
# =========================
fig, ax = plt.subplots(figsize=(7, 5), dpi=1200)

# Nature-style color scheme
color_0 = '#2166AC'  # Blue
color_1 = '#B2182B'  # Red
std_alpha = 0.15     # Transparency for standard deviation regions

# Plot standard deviation regions
ax.fill_between(x_sorted, y0_sorted - y0_std_sorted, y0_sorted + y0_std_sorted,
                alpha=std_alpha, color=color_0, linewidth=0)
ax.fill_between(x_sorted, y1_sorted - y1_std_sorted, y1_sorted + y1_std_sorted,
                alpha=std_alpha, color=color_1, linewidth=0)

# Plot original data points
ax.scatter(x, y0_mean, s=8, color=color_0, alpha=0.4, edgecolors='none')
ax.scatter(x, y1_mean, s=8, color=color_1, alpha=0.4, edgecolors='none')

# Plot smoothed curves
line0, = ax.plot(x_smooth, y0_smooth, color=color_0, linewidth=1.8,
                 label='Group 0 (Mean ± STD)')
line1, = ax.plot(x_smooth, y1_smooth, color=color_1, linewidth=1.8,
                 label='Group 1 (Mean ± STD)')

# =========================
# 5. Plot Customization
# =========================
# Axis labels
ax.set_xlabel('State Offset', fontsize=10, labelpad=8)
ax.set_ylabel('Mean Value', fontsize=10, labelpad=8)

# Tick parameters
ax.tick_params(axis='both', which='major', labelsize=9, pad=3, width=0.8)
ax.tick_params(axis='both', which='minor', labelsize=8, pad=2, width=0.6)

# Axis limits with margins
x_margin = (x.max() - x.min()) * 0.02
y_margin = (max(y0_mean.max(), y1_mean.max()) - min(y0_mean.min(), y1_mean.min())) * 0.05
ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
ax.set_ylim(min(y0_mean.min(), y1_mean.min()) - y_margin,
           max(y0_mean.max(), y1_mean.max()) + y_margin)

# Legend configuration
from matplotlib.patches import Rectangle
legend_elements = [
    line0,
    line1,
    Rectangle((0, 0), 1, 1, facecolor=color_0, alpha=std_alpha, edgecolor='none'),
    Rectangle((0, 0), 1, 1, facecolor=color_1, alpha=std_alpha, edgecolor='none')
]

legend = ax.legend(legend_elements,
                  ['Group 0 mean', 'Group 1 mean', 'Group 0 ±1 STD', 'Group 1 ±1 STD'],
                  loc='upper right', frameon=True, fancybox=False,
                  framealpha=0.9, edgecolor='black', facecolor='white')
legend.get_frame().set_linewidth(0.6)

# Spine customization
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Grid configuration
ax.grid(True, alpha=0.2, linewidth=0.5, linestyle='-')

# Title
ax.set_title('Comparison of Two Groups with Standard Deviation',
             fontsize=11, pad=12, weight='normal')

plt.tight_layout(pad=2.0)

# =========================
# 6. Save High-Quality Output
# =========================
output_base = 'bis01_comparison_nature_style'
formats = ['svg', 'png', 'pdf']

for fmt in formats:
    plt.savefig(f'{output_base}.{fmt}', format=fmt, dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✔ Saved {fmt.upper()} format: {output_base}.{fmt}")

plt.show()

# =========================
# 7. Statistical Summary
# =========================
print("\n" + "="*50)
print("DATA ANALYSIS SUMMARY")
print("="*50)
print(f"X-axis range: {x.min():.2f} to {x.max():.2f}")
print(f"Group 0 - Overall mean: {y0_mean.mean():.2f} ± {y0_std.mean():.2f}")
print(f"Group 1 - Overall mean: {y1_mean.mean():.2f} ± {y1_std.mean():.2f}")

print(f"Correlation between groups: {np.corrcoef(y0_mean, y1_mean)[0,1]:.3f}")
print("="*50)
