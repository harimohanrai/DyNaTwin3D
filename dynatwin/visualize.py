# -*- coding: utf-8 -*-
"""dynatwin/visualize.py  — 3D Edition"""

import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure

from dynatwin.config import OUTPUT_DIR


def _find_key(h, *candidates):
    for c in candidates:
        if c in h: return c
    return None


def plot_training_history(history, model_name, fold=None,
                          save_dir=OUTPUT_DIR):
    h      = history.history if hasattr(history,'history') else history
    epochs = range(1, len(h['loss'])+1)
    suffix = f'_fold{fold}' if fold is not None else ''

    fig, axes = plt.subplots(2, 3, figsize=(18,10))
    fig.suptitle(f"{model_name}{suffix} — 3D Training History", fontsize=13)

    panels = [
        (axes[0,0], ('seg_any_tumour_dice',), ('val_seg_any_tumour_dice',), 'FG Dice'),
        (axes[0,1], ('seg_dice_coef',),        ('val_seg_dice_coef',),        'Mean Dice'),
        (axes[0,2], ('loss',),                 ('val_loss',),                 'Total Loss'),
        (axes[1,0], ('seg_dice_coef_necrotic',),('val_seg_dice_coef_necrotic',),'Dice Necrotic'),
        (axes[1,1], ('seg_dice_coef_edema',),   ('val_seg_dice_coef_edema',),   'Dice Edema'),
        (axes[1,2], ('seg_dice_coef_enhancing',),('val_seg_dice_coef_enhancing',),'Dice Enh'),
    ]
    for ax, tr_c, val_c, ylabel in panels:
        tk = _find_key(h, *tr_c)
        vk = _find_key(h, *val_c)
        if tk and vk:
            ax.plot(epochs, h[tk],  label='Train', lw=1.8)
            ax.plot(epochs, h[vk],  label='Val',   lw=1.8, linestyle='--')
        ax.set_title(ylabel, fontweight='bold'); ax.set_xlabel('Epoch')
        ax.legend(); ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}{suffix}_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[visualize] Saved → {path}")
    return path


def plot_survival_curve(surv: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(surv['time_points'], surv['survival_probability'],
            lw=2.5, color='steelblue', label='S(t)')
    ax.fill_between(surv['time_points'], surv['survival_probability'],
                    alpha=0.12, color='steelblue')
    ax.axvline(surv['median_survival'], color='red', linestyle='--',
               label=f"Median = {surv['median_survival']} d")
    ax.set_ylim(0,1.05); ax.set_xlabel('Days')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f"PINN+CSV Survival  "
                 f"(calib. median OS = {surv['calibrated_median_os_days']:.0f} d)")
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[visualize] Saved → {save_path}")


def plot_fold_summary(fold_scores: list, model_name: str,
                      metric='dice_whole_tumour', save_dir=OUTPUT_DIR):
    """Box-plot of per-fold validation scores."""
    fig, ax = plt.subplots(figsize=(7,5))
    ax.boxplot(fold_scores, labels=[f'Fold {i+1}' for i in range(len(fold_scores))])
    ax.set_title(f"{model_name} — {metric} per fold")
    ax.set_ylabel(metric); ax.spines[['top','right']].set_visible(False)
    path = os.path.join(save_dir, f'{model_name}_fold_summary.png')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[visualize] Saved → {path}")
    return path


def plot_metric_comparison(results_dict: dict,
                           metric='dice_whole_tumour',
                           save_dir=OUTPUT_DIR):
    """Side-by-side violin plot comparing models on holdout set."""
    names = list(results_dict.keys())
    data  = [[r.get(metric, float('nan')) for r in results_dict[n]]
             for n in names]
    fig, ax = plt.subplots(figsize=(9,5))
    parts = ax.violinplot(data, positions=range(len(names)), showmedians=True)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_title(f"Holdout {metric} — M1 vs M2 vs M3")
    ax.set_ylabel(metric); ax.spines[['top','right']].set_visible(False)
    path = os.path.join(save_dir, f'comparison_{metric}.png')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[visualize] Saved → {path}")
    return path


def create_3d_mesh(volume, threshold=0.5, step_size=2):
    try:
        v=np.asarray(volume,np.float32)
        mn,mx=v.min(),v.max()
        if mx-mn<0.01: return None,None
        t=float(np.clip(threshold,mn+0.01,mx-0.01))
        verts,faces,_,_=measure.marching_cubes(
            v,level=t,step_size=step_size,allow_degenerate=False)
        return verts,faces
    except: return None,None


def plot_multiclass_3d(seg, title='3D Tumour'):
    seg=np.asarray(seg,np.uint8); fig=go.Figure(); fig.update_layout(title=title)
    for cv,col,nm,op in [(1,'#FF4444','Necrotic',0.70),
                          (2,'#44FF88','Edema',0.45),
                          (3,'#4488FF','Enhancing',0.65)]:
        v=(seg==cv).astype(np.float32)
        verts,faces=create_3d_mesh(v,0.5)
        if verts is None: continue
        fig.add_trace(go.Mesh3d(x=verts[:,2],y=verts[:,1],z=verts[:,0],
            i=faces[:,0],j=faces[:,1],k=faces[:,2],
            opacity=op,color=col,name=nm,flatshading=True))
    return fig

print("[visualize] 3D visualisation utilities ready.")
