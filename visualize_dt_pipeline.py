"""
visualize_dt_pipeline.py — Digital Twin Pipeline Visualization
===============================================================
Runs the full DT pipeline on a demo holdout case for each model
and generates publication figures.

  G1 — PINN calibration convergence (D, rho over epochs)
  G2 — UQ uncertainty bands (core, edema, enhancing)
  G3 — Tumor progression 90-day (volume over time)
  G4 — RT simulation (pre/post LQ model)
  G5 — Survival estimation comparison
  G6 — 3D tumor density MIP (maximum intensity projection)

Compute-heavy: ~10-15 min per model (PINN calibration + MC UQ).

Usage:
  cd ~/Hari/3D_DynaTwin
  TF_CPP_MIN_LOG_LEVEL=3 python -W ignore visualize_dt_pipeline.py
"""

import os, sys, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker

PROJECT_ROOT = '/home/ubuntu/Hari/3D_DynaTwin'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

mpl.rcParams.update({
    'font.weight': 'bold', 'axes.titleweight': 'bold',
    'axes.labelweight': 'bold', 'axes.titlesize': 13,
    'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10,
    'legend.framealpha': 0.85, 'lines.linewidth': 2.2,
    'figure.dpi': 150,
})

OUT = '/home/ubuntu/Hari/3D_DynaTwin/outputs'
FIG_DIR = os.path.join(OUT, 'publication_figures')
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    'M1_ResUNet3D': os.path.join(OUT, 'M1_ResUNet3D_fold0_best.keras'),
    'M2_ResAttUNet3D': os.path.join(OUT, 'M2_ResAttUNet3D_fold1_best.keras'),
    'M3_ASPP_AttDS3D': os.path.join(OUT, 'M3_ASPP_AttDS3D_fold0_best.keras'),
    'M3Plus': os.path.join(OUT, 'M3Plus_fold0_best.keras'),
}
SHORT = {
    'M1_ResUNet3D': 'M1-ResUNet', 'M2_ResAttUNet3D': 'M2-AttUNet',
    'M3_ASPP_AttDS3D': 'M3-ASPP', 'M3Plus': 'M3Plus',
}
COLORS = {
    'M1_ResUNet3D': '#4C72B0', 'M2_ResAttUNet3D': '#DD8452',
    'M3_ASPP_AttDS3D': '#55A868', 'M3Plus': '#C44E52',
}
CLASS_CFG = {
    'core':      {'color': '#C1121F', 'fill': '#E63946', 'title': 'Core Tumour'},
    'edema':     {'color': '#1B7F4F', 'fill': '#52B788', 'title': 'Oedema'},
    'enhancing': {'color': '#1A4FBF', 'fill': '#4895EF', 'title': 'Enhancing Tumour'},
}

# Use first holdout case as demo
DEMO_CASE = 'BraTS20_Training_041'


def _save(name, dpi=300):
    plt.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=dpi,
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIG_DIR, f'{name}.pdf'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [saved] {name}')


# ══════════════════════════════════════════════════════════════════
# Run DT pipeline per model
# ══════════════════════════════════════════════════════════════════

def run_dt_pipeline(model_key, ckpt_path):
    """Run full DT pipeline and return all results."""
    from dynatwin.losses import get_custom_objects
    from dynatwin.data_pipeline import load_volume, clear_vol_cache
    from dynatwin.evaluate_fixed import sliding_window_inference, morphological_postprocess
    import dynatwin.evaluate_fixed as _ef
    from dynatwin.digital_twin import (
        predictions_to_density, PatientDigitalTwin,
        PINNCalibrator, UncertaintyAnalyzer, SurvivalPredictor,
    )
    from dynatwin.config import strategy

    print(f'\n  Loading {model_key} ...')
    tf.keras.backend.clear_session()
    gc.collect()

    with strategy.scope():
        model = tf.keras.models.load_model(
            ckpt_path, custom_objects=get_custom_objects(),
            compile=False, safe_mode=False)

    _ps = model.input_shape[1:4]
    _ef._GAUSS = _ef._gaussian_kernel(_ps)
    print(f'  Params: {model.count_params():,}, Patch: {_ps}')

    # Inference
    print(f'  Inference on {DEMO_CASE} ...')
    X, Y = load_volume(DEMO_CASE)
    seg_pred = sliding_window_inference(model, X, patch_size=_ps)
    seg_pred = morphological_postprocess(seg_pred)

    # Convert to density
    c0 = predictions_to_density(seg_pred)
    print(f'  Density: shape={c0.shape}, range=[{c0.min():.3f}, {c0.max():.3f}]')

    # Digital Twin
    twin = PatientDigitalTwin(c0)

    # PINN Calibration with logging
    print(f'  PINN calibration (500 epochs) ...')
    pinn = PINNCalibrator(twin)

    # Capture D, rho at each epoch
    d_history = []
    rho_history = []
    loss_history = []

    # Run calibration in chunks to capture history
    original_calibrate = pinn.calibrate
    pr = pinn.calibrate(epochs=500, lr=1e-3, beta=0.8)

    # Get final params
    D_final = pr['D']
    rho_final = pr['rho']
    print(f'  PINN result: D={D_final:.5f}, rho={rho_final:.5f}')

    # Tumor progression
    print(f'  Tumor progression (90 days) ...')
    twin.D_range = (0.85 * D_final, 1.15 * D_final)
    twin.rho_range = (0.85 * rho_final, 1.15 * rho_final)
    scenarios = twin.predict_progression(90, 5)

    # UQ
    print(f'  Monte Carlo UQ (20 samples) ...')
    uq = UncertaintyAnalyzer()
    bands = uq.monte_carlo_uncertainty(twin, n=20, days=90)

    # Survival
    print(f'  Survival estimation ...')
    sp = SurvivalPredictor(twin, pr)
    surv = sp.estimate_survival(scenarios[0])

    del model
    gc.collect()
    clear_vol_cache()

    return {
        'seg_pred': seg_pred, 'c0': c0, 'X': X, 'Y': Y,
        'pinn_params': pr, 'scenarios': scenarios,
        'bands': bands, 'surv': surv,
        'D': D_final, 'rho': rho_final,
    }


# ══════════════════════════════════════════════════════════════════
# LQ Radiotherapy model
# ══════════════════════════════════════════════════════════════════

def lq_rt(c0, dose=60.0, fx=30, alpha=0.3, beta=0.03):
    d = dose / fx
    sf = np.exp(-alpha * d - beta * d**2)
    return np.clip(c0 * sf**fx, 0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# G1 — PINN Parameters Comparison
# ══════════════════════════════════════════════════════════════════

def fig_g1(results):
    print('\n[G1] PINN parameters ...')
    models = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # D values
    ax = axes[0]
    for i, mk in enumerate(models):
        D = results[mk]['D']
        ax.bar(i, D, color=COLORS[mk], alpha=0.8, edgecolor='black')
        ax.text(i, D + 0.01, f'{D:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([SHORT[m] for m in models], fontweight='bold')
    ax.set_ylabel('D (mm²/day)', fontweight='bold')
    ax.set_title('Diffusion Coefficient', fontweight='bold')
    ax.axhspan(0.05, 0.5, alpha=0.08, color='green')
    ax.text(0.02, 0.95, 'Literature\nrange', transform=ax.transAxes,
            fontsize=8, color='green', alpha=0.5, va='top')
    ax.spines[['top', 'right']].set_visible(False)

    # rho values
    ax = axes[1]
    for i, mk in enumerate(models):
        rho = results[mk]['rho']
        ax.bar(i, rho, color=COLORS[mk], alpha=0.8, edgecolor='black')
        ax.text(i, rho + 0.002, f'{rho:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([SHORT[m] for m in models], fontweight='bold')
    ax.set_ylabel('ρ (1/day)', fontweight='bold')
    ax.set_title('Proliferation Rate', fontweight='bold')
    ax.axhspan(0.05, 0.5, alpha=0.08, color='green')
    ax.spines[['top', 'right']].set_visible(False)

    # Infiltration index
    ax = axes[2]
    for i, mk in enumerate(models):
        ratio = np.sqrt(results[mk]['D'] / (results[mk]['rho'] + 1e-8))
        ax.bar(i, ratio, color=COLORS[mk], alpha=0.8, edgecolor='black')
        ax.text(i, ratio + 0.05, f'{ratio:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([SHORT[m] for m in models], fontweight='bold')
    ax.set_ylabel('√(D/ρ) (mm)', fontweight='bold')
    ax.set_title('Infiltration Index', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'PINN-Calibrated Growth Parameters — {DEMO_CASE}\n'
                 'Fisher-KPP Reaction-Diffusion Model (500 epochs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save('G1_pinn_parameters')


# ══════════════════════════════════════════════════════════════════
# G2 — UQ Uncertainty Bands
# ══════════════════════════════════════════════════════════════════

def fig_g2(results):
    print('\n[G2] UQ uncertainty bands ...')
    models = list(results.keys())

    for cls, cfg in CLASS_CFG.items():
        fig, axes = plt.subplots(1, len(models),
                                 figsize=(6 * len(models), 5), squeeze=False)
        fig.suptitle(f'UQ 95% CI — {cfg["title"]}',
                     fontsize=14, fontweight='bold')

        for col, mk in enumerate(models):
            b = results[mk]['bands'][cls]
            ax = axes[0][col]
            ax.fill_between(b['days'], b['lower'], b['upper'],
                            color=cfg['fill'], alpha=0.25, label='95% CI')
            ax.plot(b['days'], b['median'], color=cfg['color'],
                    lw=2.5, marker='o', markersize=5, label='Median')
            ax.set_title(SHORT[mk], fontweight='bold')
            ax.set_xlabel('Days', fontweight='bold')
            ax.set_ylabel('Volume (voxels)', fontweight='bold')
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
            ax.legend(fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', ls='--', alpha=0.3)

        plt.tight_layout()
        _save(f'G2_uq_{cls}')


# ══════════════════════════════════════════════════════════════════
# G3 — Tumor Progression
# ══════════════════════════════════════════════════════════════════

def fig_g3(results):
    print('\n[G3] Tumor progression ...')
    models = list(results.keys())

    fig, axes = plt.subplots(1, len(models),
                             figsize=(6 * len(models), 5), squeeze=False)
    fig.suptitle(f'90-Day Tumour Progression — Digital Twin\n{DEMO_CASE}',
                 fontsize=14, fontweight='bold')

    cls_colors = {'core': '#C1121F', 'edema': '#1B7F4F', 'enhancing': '#1A4FBF'}
    cls_labels = {'core': 'Core', 'edema': 'Edema', 'enhancing': 'Enhancing'}

    for col, mk in enumerate(models):
        ax = axes[0][col]
        sc = results[mk]['scenarios'][0]

        for cls in ['core', 'edema', 'enhancing']:
            vols = [int(np.sum(sc[cls][t] > 0)) for t in range(len(sc['days']))]
            ax.plot(sc['days'], vols, color=cls_colors[cls],
                    lw=2.2, marker='o', markersize=4, label=cls_labels[cls])

        ax.set_title(SHORT[mk], fontweight='bold')
        ax.set_xlabel('Days', fontweight='bold')
        ax.set_ylabel('Volume (voxels)', fontweight='bold')
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', ls='--', alpha=0.3)

    plt.tight_layout()
    _save('G3_tumor_progression')


# ══════════════════════════════════════════════════════════════════
# G4 — RT Simulation
# ══════════════════════════════════════════════════════════════════

def fig_g4(results):
    print('\n[G4] RT simulation ...')
    models = list(results.keys())

    fig, axes = plt.subplots(len(models), 3,
                             figsize=(15, 4 * len(models)), squeeze=False)
    fig.suptitle('Radiotherapy Simulation — LQ Model\n'
                 '60 Gy / 30 fx (α=0.3, β=0.03)',
                 fontsize=14, fontweight='bold')

    for row, mk in enumerate(models):
        c0 = results[mk]['c0']
        c0_rt = lq_rt(c0)
        diff = c0.max(0) - c0_rt.max(0)

        # Pre-RT MIP
        im0 = axes[row][0].imshow(c0.max(0).T, cmap='hot', vmin=0, vmax=1, origin='lower')
        axes[row][0].set_title(f'{SHORT[mk]} — Pre-RT', fontweight='bold')
        axes[row][0].axis('off')

        # Post-RT MIP
        axes[row][1].imshow(c0_rt.max(0).T, cmap='hot', vmin=0, vmax=1, origin='lower')
        axes[row][1].set_title('Post-RT', fontweight='bold')
        axes[row][1].axis('off')

        # Difference
        im2 = axes[row][2].imshow(diff.T, cmap='RdYlGn',
                                   vmin=0, vmax=max(diff.max(), 0.01), origin='lower')
        axes[row][2].set_title('Density Reduction\n(pre − post)', fontweight='bold')
        axes[row][2].axis('off')
        plt.colorbar(im2, ax=axes[row][2], fraction=0.046, pad=0.04)

        # Pre/post volume text
        pre_vol = int(np.sum(c0 > 0.1))
        post_vol = int(np.sum(c0_rt > 0.1))
        reduction = (1 - post_vol / max(pre_vol, 1)) * 100
        axes[row][0].text(0.02, 0.02, f'Vol: {pre_vol:,}',
                          transform=axes[row][0].transAxes, fontsize=9,
                          color='white', fontweight='bold',
                          bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
        axes[row][1].text(0.02, 0.02, f'Vol: {post_vol:,}\n({reduction:.0f}% reduction)',
                          transform=axes[row][1].transAxes, fontsize=9,
                          color='white', fontweight='bold',
                          bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))

    plt.tight_layout()
    _save('G4_rt_simulation')


# ══════════════════════════════════════════════════════════════════
# G5 — Survival Comparison
# ══════════════════════════════════════════════════════════════════

def fig_g5(results):
    print('\n[G5] Survival estimation ...')
    models = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calibrated OS
    ax = axes[0]
    calib_os = []
    for i, mk in enumerate(models):
        s = results[mk]['surv']
        os_val = s.get('calibrated_median_os_days', np.nan)
        calib_os.append(os_val)
        ax.bar(i, os_val, color=COLORS[mk], alpha=0.8, edgecolor='black')
        ax.text(i, os_val + 5, f'{os_val:.0f}d', ha='center',
                fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([SHORT[m] for m in models], fontweight='bold')
    ax.set_ylabel('Calibrated OS (days)', fontweight='bold')
    ax.set_title('PINN-Calibrated Median Overall Survival', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    # Median survival from progression
    ax = axes[1]
    for i, mk in enumerate(models):
        s = results[mk]['surv']
        ms = s.get('median_survival', np.nan)
        ax.bar(i, ms, color=COLORS[mk], alpha=0.8, edgecolor='black')
        ax.text(i, ms + 5, f'{ms:.0f}d', ha='center',
                fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([SHORT[m] for m in models], fontweight='bold')
    ax.set_ylabel('Predicted Survival (days)', fontweight='bold')
    ax.set_title('Progression-Based Survival Estimate', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    fig.suptitle(f'Survival Estimation — {DEMO_CASE}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save('G5_survival')


# ══════════════════════════════════════════════════════════════════
# G6 — 3D Density MIP
# ══════════════════════════════════════════════════════════════════

def fig_g6(results):
    print('\n[G6] 3D density MIP ...')
    models = list(results.keys())

    fig, axes = plt.subplots(len(models), 3,
                             figsize=(14, 4 * len(models)), squeeze=False)
    fig.suptitle(f'Tumour Density — Maximum Intensity Projection\n{DEMO_CASE}',
                 fontsize=14, fontweight='bold')

    proj_labels = ['Axial (top-down)', 'Coronal (front)', 'Sagittal (side)']

    for row, mk in enumerate(models):
        c0 = results[mk]['c0']

        # Three MIP projections
        mips = [
            c0.max(axis=2).T,   # axial (z projection)
            c0.max(axis=1).T,   # coronal (y projection)
            c0.max(axis=0).T,   # sagittal (x projection)
        ]

        for col, (mip, label) in enumerate(zip(mips, proj_labels)):
            im = axes[row][col].imshow(mip, cmap='hot', vmin=0, vmax=1, origin='lower')
            axes[row][col].axis('off')
            if col == 0:
                axes[row][col].set_ylabel(SHORT[mk], fontsize=12,
                                           fontweight='bold', labelpad=10)
                axes[row][col].axis('on')
                axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
            if row == 0:
                axes[row][col].set_title(label, fontweight='bold')

        # Volume annotation
        vol = int(np.sum(c0 > 0.1))
        axes[row][0].text(0.02, 0.98, f'Vol>0.1: {vol:,}',
                          transform=axes[row][0].transAxes, fontsize=9,
                          color='white', fontweight='bold', va='top',
                          bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Tumour Density')

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    _save('G6_density_mip')


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print('╔══════════════════════════════════════════════════════════╗')
    print('║  DyNaTwin — Digital Twin Pipeline Visualization         ║')
    print('╚══════════════════════════════════════════════════════════╝')
    print(f'  Demo case: {DEMO_CASE}')
    print(f'  Output → {FIG_DIR}\n')

    # Run DT pipeline for each model
    results = {}
    for mk, ckpt in MODELS.items():
        if not os.path.exists(ckpt):
            print(f'  {mk}: checkpoint not found — skipping')
            continue
        try:
            results[mk] = run_dt_pipeline(mk, ckpt)
        except Exception as e:
            print(f'  {mk}: PIPELINE ERROR — {e}')
            import traceback; traceback.print_exc()

    if not results:
        print('\n  No results — check checkpoints.')
        return

    # Generate figures
    fig_g1(results)
    fig_g2(results)
    fig_g3(results)
    fig_g4(results)
    fig_g5(results)
    fig_g6(results)

    files = sorted([f for f in os.listdir(FIG_DIR) if f.startswith('G')])
    print(f'\n  Done. DT figures:')
    for f in files:
        print(f'    {f}')


if __name__ == '__main__':
    main()
