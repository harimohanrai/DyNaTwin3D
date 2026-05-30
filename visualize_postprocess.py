"""
visualize_postprocess.py — Publication-Quality Visualization
=============================================================
Generates all figures for the post-processing analysis paper.

Figures generated:
  Fig 1: Slice overlays — T1ce + GT + Baseline + Adaptive for selected cases
  Fig 2: Summary bar chart — all models × strategies for NCR Dice
  Fig 3: Box plots — NCR distribution before/after per model
  Fig 4: Per-case improvement scatter — baseline vs adaptive NCR
  Fig 5: ET trade-off analysis — NCR gain vs ET loss
  Fig 6: NCR/TC ratio analysis — when does post-processing trigger

Does NOT modify any existing code.
All outputs → outputs/postprocess_figures/

Usage:
  cd ~/Hari/3D_DynaTwin
  TF_CPP_MIN_LOG_LEVEL=3 python -W ignore visualize_postprocess.py
"""

import os, sys, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

PROJECT_ROOT = '/home/ubuntu/Hari/3D_DynaTwin'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dynatwin.losses import get_custom_objects
from dynatwin.data_pipeline import load_volume, clear_vol_cache
from dynatwin.evaluate_fixed import (
    sliding_window_inference, morphological_postprocess,
)
import dynatwin.evaluate_fixed as _ef

# ── Global plot style ─────────────────────────────────────────────
mpl.rcParams.update({
    'font.weight':        'bold',
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
    'axes.titlesize':     13,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'legend.framealpha':  0.85,
    'lines.linewidth':    2.2,
    'figure.dpi':         150,
})

# ── Config ────────────────────────────────────────────────────────
_WEIGHTS_DIR = '/home/ubuntu/Hari/3D_DynaTwin/outputs'
FIG_DIR = os.path.join(_WEIGHTS_DIR, 'postprocess_figures')
os.makedirs(FIG_DIR, exist_ok=True)

PER_CASE_CSV = os.path.join(
    _WEIGHTS_DIR, 'postprocess_analysis', 'postprocess_per_case.csv')

T1CE_CHANNEL = 1
LABEL_NCR = 1; LABEL_ED = 2; LABEL_ET = 3

# Color map for overlays
CLASS_COLORS = {
    0: [0, 0, 0, 0],         # background — transparent
    1: [255, 0, 0, 180],     # NCR — red
    2: [0, 255, 0, 120],     # ED — green
    3: [255, 255, 0, 180],   # ET — yellow
}

MODEL_CONFIGS = {
    'M1_ResUNet3D': os.path.join(
        _WEIGHTS_DIR, 'M1_ResUNet3D_fold0_best.keras'),
    'M2_ResAttUNet3D': os.path.join(
        _WEIGHTS_DIR, 'M2_ResAttUNet3D_fold1_best.keras'),
    'M3_ASPP_AttDS3D': os.path.join(
        _WEIGHTS_DIR, 'M3_ASPP_AttDS3D_fold0_best.keras'),
    'M3Plus': os.path.join(
        _WEIGHTS_DIR, 'M3Plus_fold0_best.keras'),
}

MODEL_COLORS = {
    'M1_ResUNet3D':    '#4C72B0',
    'M2_ResAttUNet3D': '#DD8452',
    'M3_ASPP_AttDS3D': '#55A868',
    'M3Plus':          '#C44E52',
}

MODEL_SHORT = {
    'M1_ResUNet3D':    'M1-ResUNet',
    'M2_ResAttUNet3D': 'M2-ResAttUNet',
    'M3_ASPP_AttDS3D': 'M3-ASPP',
    'M3Plus':          'M3Plus',
}

# Cases to visualize (from results)
# Rescued cases, good cases, hurt cases
VIZ_CASES = {
    'rescued': [
        'BraTS20_Training_112',  # 0.053 → 0.911
        'BraTS20_Training_051',  # 0.000 → 0.820
        'BraTS20_Training_073',  # 0.009 → 0.686
    ],
    'stable': [
        'BraTS20_Training_041',  # 0.806 → 0.839
        'BraTS20_Training_164',  # 0.936 → 0.936
    ],
    'hurt': [
        'BraTS20_Training_183',  # may show degradation
    ],
}


def _save(name, dpi=300):
    path_png = os.path.join(FIG_DIR, f'{name}.png')
    path_pdf = os.path.join(FIG_DIR, f'{name}.pdf')
    plt.savefig(path_png, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [saved] {name}.png / .pdf')


# ── Post-processing function (adaptive) ──────────────────────────

def pp_adaptive(pred_seg, t1ce):
    corrected = pred_seg.copy()
    tc_mask = np.isin(pred_seg, [LABEL_NCR, LABEL_ET])
    if tc_mask.sum() == 0:
        return corrected
    et_mask = (pred_seg == LABEL_ET)
    if et_mask.sum() > 10:
        threshold = np.percentile(t1ce[et_mask], 25)
    else:
        threshold = np.median(t1ce[tc_mask])
    corrected[tc_mask] = np.where(
        t1ce[tc_mask] < threshold, LABEL_NCR, LABEL_ET)
    return corrected


# ── Overlay helper ────────────────────────────────────────────────

def make_overlay(t1ce_slice, seg_slice, alpha=0.6):
    """Create RGBA overlay of segmentation on T1ce."""
    # Normalize T1ce to [0, 255]
    vmin, vmax = np.percentile(t1ce_slice[t1ce_slice > 0], [1, 99]) \
        if (t1ce_slice > 0).any() else (0, 1)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1
    t1ce_norm = np.clip((t1ce_slice - vmin) / (vmax - vmin), 0, 1)

    # Base image in RGBA
    img = np.stack([t1ce_norm * 255] * 3 + [np.ones_like(t1ce_norm) * 255],
                   axis=-1).astype(np.uint8)

    # Overlay segmentation
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        mask = (seg_slice == cls)
        if mask.any():
            for c in range(3):
                img[mask, c] = np.clip(
                    img[mask, c] * (1 - alpha) + color[c] * alpha,
                    0, 255).astype(np.uint8)

    return img


def find_best_slice(seg_3d, target_label=1):
    """Find axial slice with most voxels of target label."""
    counts = [(seg_3d[:, :, z] == target_label).sum()
              for z in range(seg_3d.shape[2])]
    if max(counts) == 0:
        # Fallback: find slice with most tumor
        counts = [(seg_3d[:, :, z] > 0).sum()
                  for z in range(seg_3d.shape[2])]
    return int(np.argmax(counts))


# ══════════════════════════════════════════════════════════════════
# FIG 1 — Slice Overlays
# ══════════════════════════════════════════════════════════════════

def fig1_slice_overlays():
    print('\n[Fig 1] Slice overlays for selected cases ...')

    # Use M1 for visualization (simplest, representative)
    model_key = 'M1_ResUNet3D'
    ckpt_path = MODEL_CONFIGS[model_key]

    custom_objs = get_custom_objects()
    model = tf.keras.models.load_model(
        ckpt_path, custom_objects=custom_objs,
        compile=False, safe_mode=False)
    _ps = model.input_shape[1:4]
    _ef._GAUSS = _ef._gaussian_kernel(_ps)

    all_cases = (VIZ_CASES['rescued'] + VIZ_CASES['stable'] +
                 VIZ_CASES['hurt'])
    n_cases = len(all_cases)

    fig, axes = plt.subplots(n_cases, 4, figsize=(20, 5 * n_cases))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['T1ce', 'Ground Truth', 'Baseline Prediction',
                  'Adaptive Post-Processing']

    for row, case_id in enumerate(all_cases):
        print(f'  Processing {case_id} ...', end=' ', flush=True)

        try:
            X, Y = load_volume(case_id)
        except Exception as e:
            print(f'ERROR: {e}')
            for ax in axes[row]:
                ax.text(0.5, 0.5, f'Load failed\n{e}', ha='center',
                        va='center', transform=ax.transAxes)
                ax.axis('off')
            continue

        t1ce = X[..., T1CE_CHANNEL]

        # Inference
        seg_pred = sliding_window_inference(model, X, patch_size=_ps)
        seg_pred = morphological_postprocess(seg_pred)

        # Post-processing
        seg_pp = pp_adaptive(seg_pred, t1ce)

        # Find best slice (most NCR in ground truth)
        z = find_best_slice(Y, target_label=1)

        # Compute Dice for annotation
        ncr_pred = (seg_pred == 1).astype(np.uint8)
        ncr_pp = (seg_pp == 1).astype(np.uint8)
        ncr_gt = (Y == 1).astype(np.uint8)

        def _dice(p, g):
            i = (p * g).sum()
            d = p.sum() + g.sum()
            return (2.0 * i / d) if d > 0 else float('nan')

        dice_base = _dice(ncr_pred, ncr_gt)
        dice_pp = _dice(ncr_pp, ncr_gt)

        # Category label
        if case_id in VIZ_CASES['rescued']:
            cat = 'RESCUED'
            cat_color = '#27ae60'
        elif case_id in VIZ_CASES['stable']:
            cat = 'STABLE'
            cat_color = '#3498db'
        else:
            cat = 'HURT'
            cat_color = '#e74c3c'

        # Column 0: T1ce
        axes[row, 0].imshow(t1ce[:, :, z].T, cmap='gray', origin='lower')
        axes[row, 0].set_ylabel(f'{case_id}\n[{cat}]',
                                fontsize=11, fontweight='bold',
                                color=cat_color)

        # Column 1: Ground truth overlay
        axes[row, 1].imshow(make_overlay(t1ce[:, :, z].T, Y[:, :, z].T))

        # Column 2: Baseline prediction
        axes[row, 2].imshow(make_overlay(t1ce[:, :, z].T, seg_pred[:, :, z].T))
        axes[row, 2].text(0.02, 0.98,
                          f'NCR Dice: {dice_base:.3f}',
                          transform=axes[row, 2].transAxes,
                          fontsize=10, fontweight='bold',
                          color='white', va='top',
                          bbox=dict(boxstyle='round', facecolor='black',
                                    alpha=0.7))

        # Column 3: Adaptive post-processing
        axes[row, 3].imshow(make_overlay(t1ce[:, :, z].T, seg_pp[:, :, z].T))
        delta = dice_pp - dice_base
        color = '#27ae60' if delta > 0.01 else ('#e74c3c' if delta < -0.01 else '#f39c12')
        axes[row, 3].text(0.02, 0.98,
                          f'NCR Dice: {dice_pp:.3f} ({delta:+.3f})',
                          transform=axes[row, 3].transAxes,
                          fontsize=10, fontweight='bold',
                          color='white', va='top',
                          bbox=dict(boxstyle='round', facecolor=color,
                                    alpha=0.8))

        for ax in axes[row]:
            ax.axis('off')

        print(f'slice {z}  NCR: {dice_base:.3f} → {dice_pp:.3f}')

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold',
                               pad=15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Necrotic (NCR)'),
        Patch(facecolor='yellow', alpha=0.7, label='Enhancing (ET)'),
        Patch(facecolor='green', alpha=0.5, label='Edema (ED)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Post-Processing Effect on Necrotic Core Segmentation\n'
                 '(M1-ResUNet, Adaptive Strategy)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save('fig1_slice_overlays')

    del model; gc.collect()
    tf.keras.backend.clear_session()
    clear_vol_cache()


# ══════════════════════════════════════════════════════════════════
# FIG 2 — Summary Bar Chart
# ══════════════════════════════════════════════════════════════════

def fig2_summary_bars():
    print('\n[Fig 2] Summary bar charts ...')
    df = pd.read_csv(PER_CASE_CSV)

    summary = df.groupby(['model', 'strategy']).agg(
        NCR_mean=('NCR', 'mean'),
        NCR_std=('NCR', 'std'),
    ).reset_index()

    models = list(MODEL_SHORT.keys())
    strategies = ['none', 'mean', 'median', 'otsu', 'adaptive']
    strat_colors = {
        'none': '#95a5a6', 'mean': '#3498db', 'median': '#2ecc71',
        'otsu': '#e67e22', 'adaptive': '#e74c3c',
    }
    strat_labels = {
        'none': 'Baseline', 'mean': 'Mean', 'median': 'Median',
        'otsu': 'Otsu', 'adaptive': 'Adaptive',
    }

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, strat in enumerate(strategies):
        vals = []
        errs = []
        for m in models:
            row = summary[(summary['model'] == m) &
                          (summary['strategy'] == strat)]
            if len(row) > 0:
                vals.append(float(row['NCR_mean'].values[0]))
                errs.append(float(row['NCR_std'].values[0]))
            else:
                vals.append(0)
                errs.append(0)

        bars = ax.bar(x + i * width, vals, width,
                      yerr=errs, capsize=3,
                      label=strat_labels[strat],
                      color=strat_colors[strat],
                      alpha=0.85, edgecolor='white', linewidth=0.5)

        # Value labels on top
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold', rotation=45)

    ax.set_xlabel('Model', fontweight='bold', fontsize=13)
    ax.set_ylabel('NCR Dice Score', fontweight='bold', fontsize=13)
    ax.set_title('Necrotic Core Dice: Baseline vs Post-Processing Strategies\n'
                 '(37 Holdout Cases, BraTS 2020)',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, 0.85)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='_')
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    _save('fig2_ncr_bar_chart')


# ══════════════════════════════════════════════════════════════════
# FIG 3 — Box Plots
# ══════════════════════════════════════════════════════════════════

def fig3_box_plots():
    print('\n[Fig 3] Box plots NCR before/after ...')
    df = pd.read_csv(PER_CASE_CSV)

    models = list(MODEL_SHORT.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6),
                             sharey=True)

    for idx, model_key in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model_key]

        baseline = model_df[model_df['strategy'] == 'none']['NCR'].dropna().values
        adaptive = model_df[model_df['strategy'] == 'adaptive']['NCR'].dropna().values

        bp = ax.boxplot([baseline, adaptive],
                        labels=['Baseline', 'Adaptive'],
                        patch_artist=True,
                        widths=0.5,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=8))

        bp['boxes'][0].set_facecolor('#95a5a6')
        bp['boxes'][1].set_facecolor('#e74c3c')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)

        # Overlay individual points
        for i, (data, xpos) in enumerate([(baseline, 1), (adaptive, 2)]):
            jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(data))
            ax.scatter(np.full_like(data, xpos) + jitter, data,
                       alpha=0.4, s=20, color='black', zorder=3)

        ax.set_title(MODEL_SHORT[model_key], fontweight='bold', fontsize=13)
        ax.set_ylabel('NCR Dice' if idx == 0 else '', fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

        # Annotate mean improvement
        delta = adaptive.mean() - baseline.mean()
        ax.text(1.5, -0.05,
                f'Mean Δ: {delta:+.3f}',
                ha='center', fontsize=11, fontweight='bold',
                color='#27ae60' if delta > 0 else '#e74c3c',
                transform=ax.get_xaxis_transform())

    fig.suptitle('NCR Dice Distribution: Baseline vs Adaptive Post-Processing',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save('fig3_ncr_boxplots')


# ══════════════════════════════════════════════════════════════════
# FIG 4 — Per-case Scatter
# ══════════════════════════════════════════════════════════════════

def fig4_scatter():
    print('\n[Fig 4] Per-case scatter baseline vs adaptive ...')
    df = pd.read_csv(PER_CASE_CSV)

    models = list(MODEL_SHORT.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5),
                             sharey=True, sharex=True)

    for idx, model_key in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model_key]

        base_df = model_df[model_df['strategy'] == 'none'][
            ['case', 'NCR']].set_index('case')
        adap_df = model_df[model_df['strategy'] == 'adaptive'][
            ['case', 'NCR']].set_index('case')

        merged = base_df.join(adap_df, lsuffix='_base', rsuffix='_adap').dropna()

        improved = merged['NCR_adap'] > merged['NCR_base'] + 0.01
        degraded = merged['NCR_adap'] < merged['NCR_base'] - 0.01
        stable = ~improved & ~degraded

        ax.scatter(merged.loc[improved, 'NCR_base'],
                   merged.loc[improved, 'NCR_adap'],
                   c='#27ae60', s=50, alpha=0.7, label=f'Improved ({improved.sum()})',
                   edgecolors='white', linewidth=0.5)
        ax.scatter(merged.loc[degraded, 'NCR_base'],
                   merged.loc[degraded, 'NCR_adap'],
                   c='#e74c3c', s=50, alpha=0.7, label=f'Degraded ({degraded.sum()})',
                   edgecolors='white', linewidth=0.5)
        ax.scatter(merged.loc[stable, 'NCR_base'],
                   merged.loc[stable, 'NCR_adap'],
                   c='#3498db', s=50, alpha=0.7, label=f'Stable ({stable.sum()})',
                   edgecolors='white', linewidth=0.5)

        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

        ax.set_xlabel('Baseline NCR Dice' if idx == len(models) // 2 else '',
                      fontweight='bold')
        ax.set_ylabel('Adaptive NCR Dice' if idx == 0 else '',
                      fontweight='bold')
        ax.set_title(MODEL_SHORT[model_key], fontweight='bold', fontsize=13)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.15)

    fig.suptitle('Per-Case NCR Dice: Baseline vs Adaptive\n'
                 'Points above diagonal = improved by post-processing',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save('fig4_per_case_scatter')


# ══════════════════════════════════════════════════════════════════
# FIG 5 — NCR Gain vs ET Trade-off
# ══════════════════════════════════════════════════════════════════

def fig5_tradeoff():
    print('\n[Fig 5] NCR gain vs ET trade-off ...')
    df = pd.read_csv(PER_CASE_CSV)

    summary = df.groupby(['model', 'strategy']).agg(
        NCR_mean=('NCR', 'mean'),
        ET_mean=('ET', 'mean'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))

    for model_key in MODEL_SHORT:
        model_df = summary[summary['model'] == model_key]
        baseline = model_df[model_df['strategy'] == 'none']
        bl_ncr = float(baseline['NCR_mean'].values[0])
        bl_et = float(baseline['ET_mean'].values[0])

        for _, row in model_df.iterrows():
            strat = row['strategy']
            if strat == 'none':
                continue
            ncr_gain = row['NCR_mean'] - bl_ncr
            et_drop = row['ET_mean'] - bl_et

            marker = {'mean': 's', 'median': '^', 'otsu': 'D',
                      'adaptive': 'o'}.get(strat, 'o')
            size = 120 if strat == 'adaptive' else 80

            ax.scatter(ncr_gain, et_drop,
                       c=MODEL_COLORS[model_key],
                       marker=marker, s=size, alpha=0.8,
                       edgecolors='black', linewidth=0.5,
                       zorder=3)
            ax.annotate(f'{MODEL_SHORT[model_key]}\n{strat}',
                        (ncr_gain, et_drop),
                        fontsize=7, ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Ideal region
    ax.fill_between([0, 0.2], 0, 0.1, alpha=0.05, color='green')
    ax.text(0.1, 0.02, 'Ideal\n(NCR↑, ET stable)',
            ha='center', fontsize=9, color='green', alpha=0.5)

    ax.set_xlabel('NCR Dice Gain', fontweight='bold', fontsize=13)
    ax.set_ylabel('ET Dice Change', fontweight='bold', fontsize=13)
    ax.set_title('Trade-off: NCR Improvement vs ET Degradation\n'
                 'per Model × Strategy',
                 fontweight='bold', fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    _save('fig5_ncr_et_tradeoff')


# ══════════════════════════════════════════════════════════════════
# FIG 6 — All Metrics Comparison (WT, TC, ET, NCR, ED)
# ══════════════════════════════════════════════════════════════════

def fig6_all_metrics():
    print('\n[Fig 6] All metrics comparison ...')
    df = pd.read_csv(PER_CASE_CSV)

    metrics = ['WT', 'TC', 'ET', 'NCR', 'ED']
    strategies = ['none', 'adaptive']
    strat_labels = {'none': 'Baseline', 'adaptive': 'Adaptive'}
    strat_colors = {'none': '#95a5a6', 'adaptive': '#e74c3c'}

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 6),
                             sharey=True)

    for col, metric in enumerate(metrics):
        ax = axes[col]

        all_data = []
        all_labels = []
        all_colors = []

        for model_key in MODEL_SHORT:
            for strat in strategies:
                vals = df[(df['model'] == model_key) &
                          (df['strategy'] == strat)][metric].dropna().values
                all_data.append(vals)
                all_labels.append(f'{MODEL_SHORT[model_key]}\n{strat_labels[strat]}')
                all_colors.append(strat_colors[strat])

        positions = []
        pos = 0
        for i in range(len(MODEL_SHORT)):
            positions.extend([pos, pos + 0.5])
            pos += 1.5

        bp = ax.boxplot(all_data, positions=positions,
                        patch_artist=True, widths=0.4,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=5))

        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(metric, fontweight='bold', fontsize=14)
        ax.set_xticks([(positions[i] + positions[i+1]) / 2
                       for i in range(0, len(positions), 2)])
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_SHORT],
                           fontsize=8, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)

    # Common legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', alpha=0.7, label='Baseline'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Adaptive'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('All Segmentation Metrics: Baseline vs Adaptive Post-Processing',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save('fig6_all_metrics')


# ══════════════════════════════════════════════════════════════════
# FIG 7 — Wilcoxon Statistical Test
# ══════════════════════════════════════════════════════════════════

def fig7_statistical_tests():
    print('\n[Fig 7] Statistical significance tests ...')
    from scipy.stats import wilcoxon

    df = pd.read_csv(PER_CASE_CSV)

    results = []
    for model_key in MODEL_SHORT:
        model_df = df[df['model'] == model_key]
        base = model_df[model_df['strategy'] == 'none'].set_index('case')
        adap = model_df[model_df['strategy'] == 'adaptive'].set_index('case')

        common = base.index.intersection(adap.index)
        b_ncr = base.loc[common, 'NCR'].values
        a_ncr = adap.loc[common, 'NCR'].values

        # Remove NaN pairs
        mask = ~(np.isnan(b_ncr) | np.isnan(a_ncr))
        b_ncr, a_ncr = b_ncr[mask], a_ncr[mask]

        if len(b_ncr) >= 10:
            stat, p = wilcoxon(b_ncr, a_ncr, alternative='two-sided')
            # Bootstrap CI for delta
            rng = np.random.default_rng(42)
            deltas = a_ncr - b_ncr
            boot_means = [rng.choice(deltas, len(deltas), replace=True).mean()
                          for _ in range(2000)]
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)
        else:
            stat, p = np.nan, np.nan
            ci_lo, ci_hi = np.nan, np.nan

        results.append({
            'model': MODEL_SHORT[model_key],
            'n': len(b_ncr),
            'baseline_mean': b_ncr.mean(),
            'adaptive_mean': a_ncr.mean(),
            'delta_mean': (a_ncr - b_ncr).mean(),
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'wilcoxon_W': stat,
            'p_value': p,
            'significant': p < 0.05 if not np.isnan(p) else False,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(FIG_DIR, 'statistical_tests.csv'), index=False)

    # Print
    print(f"\n  {'Model':<16} {'N':>4} {'Baseline':>10} {'Adaptive':>10} "
          f"{'Delta':>8} {'95% CI':>16} {'W':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'─'*90}")
    for _, r in results_df.iterrows():
        sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01
              else ('*' if r['p_value'] < 0.05 else 'ns'))
        print(f"  {r['model']:<16} {r['n']:>4} {r['baseline_mean']:>10.4f} "
              f"{r['adaptive_mean']:>10.4f} {r['delta_mean']:>+8.4f} "
              f"[{r['ci_lo']:>+.4f}, {r['ci_hi']:>+.4f}] "
              f"{r['wilcoxon_W']:>10.1f} {r['p_value']:>10.6f} {sig:>5}")

    # Plot: Bootstrap CI for delta
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, r in results_df.iterrows():
        color = MODEL_COLORS[list(MODEL_SHORT.keys())[i]]
        ax.errorbar(i, r['delta_mean'],
                    yerr=[[r['delta_mean'] - r['ci_lo']],
                          [r['ci_hi'] - r['delta_mean']]],
                    fmt='o', color=color, capsize=12, markersize=14,
                    markeredgewidth=2, markeredgecolor='black',
                    linewidth=2.5)
        sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01
              else ('*' if r['p_value'] < 0.05 else 'ns'))
        ax.text(i, r['ci_hi'] + 0.01,
                f'{r["delta_mean"]:+.4f}\np={r["p_value"]:.4f} {sig}',
                ha='center', fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['model'], fontweight='bold')
    ax.set_ylabel('NCR Dice Change (Adaptive − Baseline)', fontweight='bold')
    ax.set_title('Statistical Significance: Adaptive Post-Processing Effect\n'
                 'Bootstrap 95% CI + Wilcoxon Signed-Rank Test',
                 fontweight='bold', fontsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    _save('fig7_statistical_tests')


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print('╔══════════════════════════════════════════════════════╗')
    print('║  Post-Processing Visualization — Publication Figures ║')
    print('╚══════════════════════════════════════════════════════╝')
    print(f'  Output → {FIG_DIR}\n')

    # Fig 1 requires model inference — run first
    fig1_slice_overlays()

    # Remaining figures use CSV only — fast
    fig2_summary_bars()
    fig3_box_plots()
    fig4_scatter()
    fig5_tradeoff()
    fig6_all_metrics()
    fig7_statistical_tests()

    files = sorted(os.listdir(FIG_DIR))
    print(f'\n  Done. {len(files)} files in {FIG_DIR}:')
    for f in files:
        print(f'    {f}')


if __name__ == '__main__':
    main()
