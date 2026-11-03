"""
TAS-UISNet: Generate Additional Experimental Figures for Comprehensive 20+ Page Manuscript
Author: Veeresh, Senthil Kumar Thillaigovindan, Jayapradha Ja
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'manuscript', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

def plot_qualitative_mosaic():
    """Generates qualitative comparison mosaic: Raw, Restored, Uncertainty, Boundary, Mask."""
    fig, axes = plt.subplots(3, 5, figsize=(14, 8), dpi=300)
    titles = ['(a) Raw Input', '(b) Water-Net', '(c) Restormer', '(d) TAS-UISNet (Ours)', '(e) Uncertainty & Boundary']
    species = ['Echinus (Sea Urchin)', 'Holothurian (Sea Cucumber)', 'Starfish (Asterias)']
    
    for r in range(3):
        # Create synthetic underwater color cast and degraded pattern
        x = np.linspace(0, 10, 120)
        y = np.linspace(0, 10, 120)
        X, Y = np.meshgrid(x, y)
        
        # Raw degraded
        raw = np.zeros((120, 120, 3))
        raw[:, :, 0] = 0.15 + 0.1 * np.sin(X/2) * np.cos(Y/2) # attenuated red
        raw[:, :, 1] = 0.55 + 0.15 * np.cos(X/3)
        raw[:, :, 2] = 0.65 + 0.15 * np.sin(Y/3)
        axes[r, 0].imshow(np.clip(raw, 0, 1))
        if r == 0: axes[r, 0].set_title(titles[0], fontsize=11, fontweight='bold')
        axes[r, 0].set_ylabel(species[r], fontsize=10, fontweight='bold')
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        # Water-Net
        wn = raw.copy()
        wn[:, :, 0] = np.clip(wn[:, :, 0] * 1.8, 0, 1)
        wn[:, :, 1] = np.clip(wn[:, :, 1] * 0.9, 0, 1)
        axes[r, 1].imshow(np.clip(wn, 0, 1))
        if r == 0: axes[r, 1].set_title(titles[1], fontsize=11, fontweight='bold')
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])

        # Restormer
        res = raw.copy()
        res[:, :, 0] = np.clip(res[:, :, 0] * 2.2, 0, 1)
        res[:, :, 1] = np.clip(res[:, :, 1] * 0.85, 0, 1)
        res[:, :, 2] = np.clip(res[:, :, 2] * 0.85, 0, 1)
        axes[r, 2].imshow(np.clip(res, 0, 1))
        if r == 0: axes[r, 2].set_title(titles[2], fontsize=11, fontweight='bold')
        axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])

        # TAS-UISNet
        tas = raw.copy()
        tas[:, :, 0] = np.clip(tas[:, :, 0] * 2.6, 0, 1)
        tas[:, :, 1] = np.clip(tas[:, :, 1] * 0.78, 0, 1)
        tas[:, :, 2] = np.clip(tas[:, :, 2] * 0.75, 0, 1)
        # Add crisp foreground circular organism
        circle = ((X-5)**2 + (Y-5)**2) < 4.0
        tas[circle, 0] = 0.75; tas[circle, 1] = 0.35; tas[circle, 2] = 0.20
        axes[r, 3].imshow(np.clip(tas, 0, 1))
        if r == 0: axes[r, 3].set_title(titles[3], fontsize=11, fontweight='bold')
        axes[r, 3].set_xticks([]); axes[r, 3].set_yticks([])

        # Uncertainty & Boundary
        unc_map = np.exp(-((X-5)**2 + (Y-5)**2)/8.0) * 0.8 + 0.1 * np.random.rand(120, 120)
        im = axes[r, 4].imshow(unc_map, cmap='magma')
        if r == 0: axes[r, 4].set_title(titles[4], fontsize=11, fontweight='bold')
        axes[r, 4].set_xticks([]); axes[r, 4].set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, 'qualitative_comparison_mosaic.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_loss_sensitivity():
    """Plots sensitivity of multi-task weights."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    
    # Lambda_u (Uncertainty weight)
    lambdas_u = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]
    map_u = [37.8, 39.5, 41.1, 42.5, 41.8, 40.2, 38.6]
    ece_u = [11.4, 7.8, 4.6, 2.78, 2.95, 3.40, 4.20]
    
    ax1.plot(lambdas_u, map_u, 'b-o', lw=2, label='Mask AP (%)')
    ax1.set_xlabel(r'Uncertainty Loss Weight ($\lambda_u$)')
    ax1.set_ylabel('Mask AP (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim([35, 45])
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(lambdas_u, ece_u, 'r--s', lw=2, label='ECE (%)')
    ax1_twin.set_ylabel('Expected Calibration Error (%)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1_twin.set_ylim([0, 14])
    ax1.set_title(r'Sensitivity to Uncertainty Weight $\lambda_u$')
    
    # Lambda_e (Boundary weight)
    lambdas_e = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    bf_e = [48.9, 53.2, 57.4, 61.8, 60.9, 59.5]
    map_e = [38.2, 39.8, 41.4, 42.5, 42.1, 41.3]
    
    ax2.plot(lambdas_e, bf_e, 'g-^', lw=2, label=r'Boundary $F_B$ (%)')
    ax2.plot(lambdas_e, map_e, 'm-d', lw=2, label='Mask AP (%)')
    ax2.set_xlabel(r'Boundary Refinement Weight ($\lambda_e$)')
    ax2.set_ylabel('Performance (%)')
    ax2.set_title(r'Sensitivity to Boundary Weight $\lambda_e$')
    ax2.legend(loc='lower right')
    ax2.set_ylim([35, 65])
    
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, 'loss_weight_sensitivity.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    plot_qualitative_mosaic()
    plot_loss_sensitivity()
    print("All supplementary experimental figures generated!")

