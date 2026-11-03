"""
TAS-UISNet: Render Clean High-Resolution Diagrams for PDF Integration
Converts conceptual architecture, modules, and protocols into publication-quality figures.
Author: Veeresh, Senthil Kumar Thillaigovindan, Jayapradha Ja
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'manuscript', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'serif'

def draw_overall_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.axis('off')
    
    # Bounding boxes
    def add_box(xy, w, h, text, color, textcolor='white', bold=True):
        rect = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08", 
                                      facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(xy[0] + w/2, xy[1] + h/2, text, color=textcolor, fontsize=8.5, 
                ha='center', va='center', weight=weight, multialignment='center')

    # Draw components
    add_box((0.02, 0.4), 0.12, 0.2, "Degraded Image\n(480x640x3)", '#0b2545')
    add_box((0.18, 0.25), 0.16, 0.5, "Shared Multi-Scale\nPyramid Encoder\n{P2, P3, P4, P5}\nwith CBAM Attn", '#134074')
    add_box((0.38, 0.65), 0.18, 0.22, "Task-Conditioned\nRestoration Decoder\n(ICSM + FSFI)", '#006666')
    add_box((0.38, 0.35), 0.18, 0.18, "Uncertainty Module\n& Dynamic Gating\n(\u03a9 Gating)", '#8b0000')
    add_box((0.38, 0.08), 0.18, 0.18, "Coarse Semantic\nPrior Extractor\n(Non-Circular)", '#4b0082')
    add_box((0.62, 0.18), 0.16, 0.35, "Boundary Refinement\nModule (BRM)\n(Differential Edge)", '#2e8b57')
    add_box((0.82, 0.18), 0.15, 0.35, "Instance Segmentation\nHead (RoIAlign,\nBoxes, Masks)", '#4682b4')

    # Output boxes
    add_box((0.62, 0.65), 0.16, 0.22, "Enhanced Image\n(PSNR: 24.68 dB\nUIQM: 3.342)", '#20b2aa')
    add_box((0.82, 0.65), 0.15, 0.22, "Uncertainty Map\n(ECE: 2.78%)", '#cd5c5c')
    add_box((0.82, 0.02), 0.15, 0.12, "Instance Masks\n(Mask AP: 42.5%)", '#3cb371')

    # Arrows
    arrow_args = dict(arrowstyle="->", lw=1.5, color='#222222')
    ax.annotate("", xy=(0.18, 0.5), xytext=(0.14, 0.5), arrowprops=arrow_args)
    ax.annotate("", xy=(0.38, 0.75), xytext=(0.34, 0.65), arrowprops=arrow_args)
    ax.annotate("", xy=(0.38, 0.17), xytext=(0.34, 0.35), arrowprops=arrow_args)
    ax.annotate("", xy=(0.38, 0.44), xytext=(0.34, 0.5), arrowprops=arrow_args)
    ax.annotate("", xy=(0.62, 0.75), xytext=(0.56, 0.75), arrowprops=arrow_args)
    ax.annotate("", xy=(0.82, 0.75), xytext=(0.78, 0.75), arrowprops=arrow_args)
    ax.annotate("", xy=(0.62, 0.35), xytext=(0.56, 0.35), arrowprops=arrow_args)
    ax.annotate("", xy=(0.82, 0.35), xytext=(0.78, 0.35), arrowprops=arrow_args)
    ax.annotate("", xy=(0.89, 0.14), xytext=(0.89, 0.18), arrowprops=arrow_args)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, 'fig1_overall_architecture_render.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def draw_icsm_mechanism():
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    ax.axis('off')
    
    def add_box(xy, w, h, text, color, textcolor='white'):
        rect = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor='#333333')
        ax.add_patch(rect)
        ax.text(xy[0] + w/2, xy[1] + h/2, text, color=textcolor, fontsize=8.5, ha='center', va='center', weight='bold')

    add_box((0.05, 0.6), 0.2, 0.25, "Input Feature\nF \u2208 R^(C x H x W)", '#134074')
    add_box((0.05, 0.15), 0.2, 0.25, "Gated Prior\nS \u2208 R^(1 x H x W)", '#8b0000')

    add_box((0.35, 0.7), 0.18, 0.2, "Scale Head \u03b3(S)\nConv 3x3 -> Conv 1x1", '#006666')
    add_box((0.35, 0.15), 0.18, 0.2, "Shift Head \u03b2(S)\nConv 3x3 -> Conv 1x1", '#006666')

    add_box((0.6, 0.65), 0.12, 0.18, "(1 + \u03b3) \u2299 F\nModulation", '#e65100')
    add_box((0.78, 0.4), 0.18, 0.25, "Modulated Feature\nF_mod = (1+\u03b3)\u2299F + \u03b2", '#2e8b57')

    arrow_args = dict(arrowstyle="->", lw=1.5, color='#222222')
    ax.annotate("", xy=(0.35, 0.8), xytext=(0.25, 0.27), arrowprops=arrow_args)
    ax.annotate("", xy=(0.35, 0.25), xytext=(0.25, 0.27), arrowprops=arrow_args)
    ax.annotate("", xy=(0.6, 0.74), xytext=(0.25, 0.72), arrowprops=arrow_args)
    ax.annotate("", xy=(0.6, 0.74), xytext=(0.53, 0.8), arrowprops=arrow_args)
    ax.annotate("", xy=(0.78, 0.52), xytext=(0.72, 0.74), arrowprops=arrow_args)
    ax.annotate("", xy=(0.78, 0.52), xytext=(0.53, 0.25), arrowprops=arrow_args)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, 'fig2_icsm_mechanism_render.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def draw_orin_workflow():
    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    ax.axis('off')
    
    def add_box(xy, w, h, text, color):
        rect = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor='#222222')
        ax.add_patch(rect)
        ax.text(xy[0] + w/2, xy[1] + h/2, text, color='white', fontsize=8.5, ha='center', va='center', weight='bold')

    add_box((0.03, 0.3), 0.18, 0.4, "PyTorch Model\n(FP32 Baseline)", '#0b2545')
    add_box((0.26, 0.3), 0.18, 0.4, "ONNX Graph\nOptimization (v14)", '#134074')
    add_box((0.49, 0.3), 0.22, 0.4, "TensorRT Engine\nCalibration (INT8 / FP16)", '#d97706')
    add_box((0.76, 0.25), 0.21, 0.5, "NVIDIA Jetson Orin\nExecution\n57.1 FPS @ 15W", '#059669')

    arrow_args = dict(arrowstyle="->", lw=2, color='#333333')
    ax.annotate("", xy=(0.26, 0.5), xytext=(0.21, 0.5), arrowprops=arrow_args)
    ax.annotate("", xy=(0.49, 0.5), xytext=(0.44, 0.5), arrowprops=arrow_args)
    ax.annotate("", xy=(0.76, 0.5), xytext=(0.71, 0.5), arrowprops=arrow_args)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, 'fig3_orin_workflow_render.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    draw_overall_architecture()
    draw_icsm_mechanism()
    draw_orin_workflow()
    print("Conceptual architecture diagrams successfully rendered!")

