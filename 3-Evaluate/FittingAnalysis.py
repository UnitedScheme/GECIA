"""
Curve Fitting Analysis with Multiple Methods
============================================

This script performs curve fitting on experimental data using multiple 
mathematical models and generates publication-quality visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

# =========================
# 1. Load and Validate Data
# =========================
print("▌Loading data...")
csv_file = "bs01.csv"

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found!")
    exit()

df = pd.read_csv(csv_file)
print(f"✔ Data loaded successfully | Shape: {df.shape}")

# =========================
# 2. Define Fitting Functions
# =========================
def polynomial_4th_order(x, a, b, c, d, e):
    """4th order polynomial function for curve fitting."""
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def gaussian_with_offset(x, a, mu, sigma, offset):
    """Gaussian function with vertical offset."""
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def sigmoid_function(x, L, k, x0, offset):
    """Sigmoid (logistic) function for S-shaped curves."""
    return L / (1 + np.exp(-k * (x - x0))) + offset

# =========================
# 3. Data Extraction and Preparation
# =========================
x_data = df['state_offset'].values
y0_mean = df['0_mean'].values
y0_std = df['0_std'].values
y1_mean = df['1_mean'].values
y1_std = df['1_std'].values

# Create fine grid for smooth fitted curves
x_fine = np.linspace(x_data.min(), x_data.max(), 500)

# =========================
# 4. Curve Fitting Execution
# =========================
fit_results = {}

try:
    # Polynomial fitting for Group 0
    p0_coeffs = np.polyfit(x_data, y0_mean, 4)
    y0_poly_fit = np.polyval(p0_coeffs, x_fine)
    
    # Polynomial fitting for Group 1
    p1_coeffs = np.polyfit(x_data, y1_mean, 4)
    y1_poly_fit = np.polyval(p1_coeffs, x_fine)
    
    fit_results['method'] = "4th Order Polynomial"
    fit_results['y0_fit'] = y0_poly_fit
    fit_results['y1_fit'] = y1_poly_fit
    fit_results['success'] = True
    
except Exception as e:
    print(f"Polynomial fitting failed: {e}")
    # Fallback to spline interpolation
    from scipy.interpolate import UnivariateSpline
    
    spl0 = UnivariateSpline(x_data, y0_mean, s=100)
    spl1 = UnivariateSpline(x_data, y1_mean, s=100)
    
    fit_results['method'] = "Spline Interpolation"
    fit_results['y0_fit'] = spl0(x_fine)
    fit_results['y1_fit'] = spl1(x_fine)
    fit_results['success'] = False

# =========================
# 5. Nature-Style Plot Configuration
# =========================
# Configure Nature journal style
plt.rcParams.update({
    'font.size': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 1200,
    'savefig.dpi': 600,
})

# Create figure with Nature-style proportions
fig, ax = plt.subplots(figsize=(8, 6))

# Nature-style color scheme
color_0 = '#2166AC'  # Blue
color_1 = '#B2182B'  # Red

# =========================
# 6. Data Visualization
# =========================
# Plot error bars for original data
ax.errorbar(x_data, y0_mean, yerr=y0_std, fmt='o',
           label='Group 0 (Mean ± STD)', alpha=0.7, color=color_0,
           markersize=5, capsize=3, capthick=1, linewidth=1)

ax.errorbar(x_data, y1_mean, yerr=y1_std, fmt='s',
           label='Group 1 (Mean ± STD)', alpha=0.7, color=color_1,
           markersize=5, capsize=3, capthick=1, linewidth=1)

# Plot fitted curves
ax.plot(x_fine, fit_results['y0_fit'], color=color_0, 
        label=f'Group 0 Fit ({fit_results["method"]})', linewidth=2.5)
ax.plot(x_fine, fit_results['y1_fit'], color=color_1, 
        label=f'Group 1 Fit ({fit_results["method"]})', linewidth=2.5)

# =========================
# 7. Plot Customization
# =========================
# Axis labels and title
ax.set_xlabel('State Offset', fontsize=11, labelpad=8)
ax.set_ylabel('Mean Value', fontsize=11, labelpad=8)
ax.set_title('Curve Fitting Analysis with Standard Deviation', 
             fontsize=12, pad=15, weight='normal')

# Legend configuration
ax.legend(frameon=True, fancybox=False, framealpha=0.9, 
          edgecolor='black', facecolor='white', fontsize=9)
ax.legend.get_frame().set_linewidth(0.6)

# Grid and spine customization
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set axis limits with margins
x_margin = (x_data.max() - x_data.min()) * 0.05
y_min = min(y0_mean.min(), y1_mean.min())
y_max = max(y0_mean.max(), y1_mean.max())
y_margin = (y_max - y_min) * 0.08

ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)

plt.tight_layout()

# =========================
# 8. Save High-Quality Output
# =========================
output_files = [
    'curve_fitting_analysis.png',
    'curve_fitting_analysis.pdf',
    'curve_fitting_analysis.svg'
]

for output_file in output_files:
    plt.savefig(output_file, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✔ Saved: {output_file}")

plt.show()

# =========================
# 9. Analysis Summary
# =========================
print("\n" + "="*60)
print("CURVE FITTING ANALYSIS SUMMARY")
print("="*60)
print(f"Fitting method used: {fit_results['method']}")
print(f"Number of data points: {len(x_data)}")
print(f"X-axis range: {x_data.min():.2f} to {x_data.max():.2f}")
print(f"Group 0 range: {y0_mean.min():.2f} to {y0_mean.max():.2f}")
print(f"Group 1 range: {y1_mean.min():.2f} to {y1_mean.max():.2f}")
print(f"Group 0 mean ± std: {y0_mean.mean():.2f} ± {y0_std.mean():.2f}")
print(f"Group 1 mean ± std: {y1_mean.mean():.2f} ± {y1_std.mean():.2f}")

# Calculate correlation between groups
correlation = np.corrcoef(y0_mean, y1_mean)[0, 1]
print(f"Correlation between groups: {correlation:.3f}")

if fit_results['success']:
    # Calculate fitting quality metrics for polynomial fit
    y0_pred = np.polyval(p0_coeffs, x_data)
    y1_pred = np.polyval(p1_coeffs, x_data)
    
    r2_0 = 1 - np.sum((y0_mean - y0_pred)**2) / np.sum((y0_mean - y0_mean.mean())**2)
    r2_1 = 1 - np.sum((y1_mean - y1_pred)**2) / np.sum((y1_mean - y1_mean.mean())**2)
    
    print(f"Group 0 R² score: {r2_0:.3f}")
    print(f"Group 1 R² score: {r2_1:.3f}")

print("="*60)
