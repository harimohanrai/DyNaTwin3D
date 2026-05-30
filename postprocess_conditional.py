"""
postprocess_conditional.py — Conditional T1ce Post-Processing
==============================================================
Adds Strategy 5: Conditional Otsu
  - Only apply Otsu when model's NCR volume is suspiciously low
    relative to TC volume (NCR/TC ratio below threshold)
  - Leave good predictions untouched → preserves ET on easy cases
  - Rescues failed NCR on hard cases

Runs on all 4 models: M1, M2, M3, M3Plus
Does NOT modify any existing code.
All outputs → outputs/postprocess_analysis/

Usage:
  cd ~/Hari/3D_DynaTwin
  python postprocess_conditional.py
  python postprocess_conditional.py --models M1_ResUNet3D
"""

import os, sys, argparse, gc
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dynatwin.config import NUM_CLASSES
from dynatwin.losses import get_custom_objects
from dynatwin.data_pipeline import load_volume, clear_vol_cache

# Import and patch evaluate_fixed
import dynatwin.evaluate_fixed as _ef
from dynatwin.evaluate_fixed import sliding_window_inference, morphological_postprocess

# ── Config ────────────────────────────────────────────────────────
T1CE_CHANNEL = 1

_WEIGHTS_DIR = '/home/ubuntu/Hari/3D_DynaTwin/outputs'
PP_OUTPUT_DIR = os.path.join(_WEIGHTS_DIR, 'postprocess_analysis')
os.makedirs(PP_OUTPUT_DIR, exist_ok=True)

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

LABEL_BG = 0; LABEL_NCR = 1; LABEL_ED = 2; LABEL_ET = 3

# ── Holdout IDs ───────────────────────────────────────────────────
# Using same split logic as your training code
# If this doesn't give 37 cases, replace with hardcoded list

def get_holdout_ids():
    from dynatwin.data_pipeline import load_survival_df, get_stratified_split
    try:
        surv_df = load_survival_df()
        splits = get_stratified_split(surv_df)
        # splits should contain holdout_ids
        if hasattr(splits, '__len__') and len(splits) == 2:
            holdout_ids, _ = splits
            return holdout_ids
    except Exception:
        pass

    # Fallback: manual split matching your training
    from dynatwin.config import TRAIN_DATASET_PATH
    all_dirs = sorted([
        d for d in os.listdir(TRAIN_DATASET_PATH)
        if os.path.isdir(os.path.join(TRAIN_DATASET_PATH, d))
        and d.startswith('BraTS20_Training_')
    ])
    np.random.seed(42)
    indices = np.random.permutation(len(all_dirs))
    n_holdout = max(1, int(len(all_dirs) * 0.10))
    return [all_dirs[i] for i in indices[:n_holdout]]


# ── Dice ──────────────────────────────────────────────────────────

def dice_score(pred, gt, smooth=1e-6):
    p = pred.flatten().astype(np.float64)
    g = gt.flatten().astype(np.float64)
    if p.sum() == 0 and g.sum() == 0:
        return np.nan
    return float((2.0 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth))


def compute_all_dice(pred_seg, gt_seg):
    return {
        'WT':  dice_score((pred_seg > 0).astype(np.uint8),
                          (gt_seg > 0).astype(np.uint8)),
        'TC':  dice_score(np.isin(pred_seg, [1, 3]).astype(np.uint8),
                          np.isin(gt_seg, [1, 3]).astype(np.uint8)),
        'ET':  dice_score((pred_seg == 3).astype(np.uint8),
                          (gt_seg == 3).astype(np.uint8)),
        'NCR': dice_score((pred_seg == 1).astype(np.uint8),
                          (gt_seg == 1).astype(np.uint8)),
        'ED':  dice_score((pred_seg == 2).astype(np.uint8),
                          (gt_seg == 2).astype(np.uint8)),
    }


# ── Post-processing strategies ────────────────────────────────────

def _otsu_threshold(values):
    """Compute Otsu threshold on a 1D array. Returns threshold value."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-6:
        return (vmin + vmax) / 2.0

    normed = ((values - vmin) / (vmax - vmin) * 255).astype(np.int32)
    normed = np.clip(normed, 0, 255)
    hist = np.bincount(normed, minlength=256).astype(np.float64)
    hist /= hist.sum()

    best_t, best_var = 0, 0.0
    w0, s0 = 0.0, 0.0
    total_mean = np.sum(np.arange(256) * hist)

    for t in range(256):
        w0 += hist[t]
        s0 += t * hist[t]
        if w0 == 0 or w0 == 1:
            continue
        w1 = 1.0 - w0
        m0 = s0 / w0
        m1 = (total_mean - s0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var = var
            best_t = t

    return vmin + (best_t / 255.0) * (vmax - vmin)


def pp_none(pred_seg, t1ce):
    return pred_seg


def pp_otsu(pred_seg, t1ce):
    corrected = pred_seg.copy()
    tc_mask = np.isin(pred_seg, [LABEL_NCR, LABEL_ET])
    if tc_mask.sum() < 10:
        return corrected
    tc_vals = t1ce[tc_mask]
    thresh = _otsu_threshold(tc_vals)
    corrected[tc_mask] = np.where(tc_vals < thresh, LABEL_NCR, LABEL_ET)
    return corrected


def pp_conditional_otsu(pred_seg, t1ce, ncr_ratio_threshold=0.10):
    """
    Strategy 5: Conditional Otsu.

    Only apply Otsu when the model's NCR prediction is suspicious:
      NCR_volume / TC_volume < threshold

    If NCR/TC ratio is healthy (above threshold), trust the model.
    If ratio is suspiciously low (model assigned almost everything
    to ET), override with Otsu splitting.

    Default threshold 0.10 = if less than 10% of tumour core is
    predicted as necrotic, we suspect the model failed on NCR.
    """
    tc_mask = np.isin(pred_seg, [LABEL_NCR, LABEL_ET])
    ncr_mask = (pred_seg == LABEL_NCR)

    n_tc = tc_mask.sum()
    n_ncr = ncr_mask.sum()

    if n_tc == 0:
        return pred_seg

    ncr_ratio = n_ncr / n_tc

    if ncr_ratio >= ncr_ratio_threshold:
        # Model's NCR looks reasonable — trust it
        return pred_seg
    else:
        # NCR suspiciously low — apply Otsu correction
        return pp_otsu(pred_seg, t1ce)


def pp_conditional_otsu_15(pred_seg, t1ce):
    """Conditional Otsu with 15% threshold."""
    return pp_conditional_otsu(pred_seg, t1ce, ncr_ratio_threshold=0.15)


def pp_conditional_otsu_20(pred_seg, t1ce):
    """Conditional Otsu with 20% threshold."""
    return pp_conditional_otsu(pred_seg, t1ce, ncr_ratio_threshold=0.20)


def pp_conditional_otsu_25(pred_seg, t1ce):
    """Conditional Otsu with 25% threshold."""
    return pp_conditional_otsu(pred_seg, t1ce, ncr_ratio_threshold=0.25)


PP_STRATEGIES = {
    'none':             pp_none,
    'otsu':             pp_otsu,
    'cond_otsu_10':     pp_conditional_otsu,
    'cond_otsu_15':     pp_conditional_otsu_15,
    'cond_otsu_20':     pp_conditional_otsu_20,
    'cond_otsu_25':     pp_conditional_otsu_25,
}


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_model(model_key, ckpt_path, holdout_ids):
    print(f"\n{'─'*60}")
    print(f"  Model: {model_key}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'─'*60}")

    if not os.path.exists(ckpt_path):
        print(f"  ERROR: checkpoint not found — skipping")
        print(f"  Searched: {ckpt_path}")
        # Try to find it
        import glob
        pattern = os.path.join(_WEIGHTS_DIR, f'{model_key}*best.keras')
        found = glob.glob(pattern)
        if found:
            print(f"  Found alternatives: {found}")
        return []

    custom_objs = get_custom_objects()
    model = tf.keras.models.load_model(
        ckpt_path, custom_objects=custom_objs,
        compile=False, safe_mode=False)
    print(f"  Loaded: {model.count_params():,} params")

    # Patch PATCH_SIZE and Gaussian kernel for this model
    _ps = model.input_shape[1:4]
    _ef._GAUSS = _ef._gaussian_kernel(_ps)
    print(f"  Patch size: {_ps}")

    results = []
    for idx, case_id in enumerate(holdout_ids):
        print(f"  {idx+1}/{len(holdout_ids)}  {case_id}  ", end='', flush=True)

        try:
            X, Y = load_volume(case_id)
        except Exception as e:
            print(f"LOAD ERROR: {e}")
            continue

        # Ground truth
        if Y.ndim > 3 and Y.shape[-1] > 1:
            gt_seg = np.argmax(Y, axis=-1)
        else:
            gt_seg = Y.squeeze() if Y.ndim > 3 else Y

        t1ce = X[..., T1CE_CHANNEL]

        # Inference
        pred_soft = sliding_window_inference(model, X, patch_size=_ps)
        pred_seg_base = morphological_postprocess(pred_soft)
        if pred_seg_base.ndim > 3 and pred_seg_base.shape[-1] > 1:
            pred_seg_base = np.argmax(pred_seg_base, axis=-1)

        # All strategies
        for strat_name, strat_fn in PP_STRATEGIES.items():
            pred_final = strat_fn(pred_seg_base.copy(), t1ce)
            dice = compute_all_dice(pred_final, gt_seg)
            results.append({
                'model': model_key, 'case': case_id,
                'strategy': strat_name,
                'WT': dice['WT'], 'TC': dice['TC'],
                'ET': dice['ET'], 'NCR': dice['NCR'], 'ED': dice['ED'],
            })

        # Print baseline vs best conditional
        n_strats = len(PP_STRATEGIES)
        base_ncr = results[-n_strats]['NCR']  # 'none'
        cond_results = {r['strategy']: r['NCR'] for r in results[-n_strats:]
                        if r['strategy'].startswith('cond_')}
        if cond_results:
            best_cond = max(cond_results, key=lambda k:
                           cond_results[k] if not np.isnan(cond_results[k]) else -1)
            best_ncr = cond_results[best_cond]
            b = f'{base_ncr:.3f}' if not np.isnan(base_ncr) else 'nan'
            a = f'{best_ncr:.3f}' if not np.isnan(best_ncr) else 'nan'
            d = ''
            if not np.isnan(base_ncr) and not np.isnan(best_ncr):
                delta = best_ncr - base_ncr
                d = f' ({"+" if delta >= 0 else ""}{delta:.3f})'
            print(f"NCR: {b} → {a} [{best_cond}]{d}")
        else:
            print()

    del model; gc.collect()
    tf.keras.backend.clear_session()
    clear_vol_cache()
    return results


def summarise_and_save(all_results):
    df = pd.DataFrame(all_results)

    detail_path = os.path.join(PP_OUTPUT_DIR, 'conditional_per_case.csv')
    df.to_csv(detail_path, index=False)
    print(f"\n  Per-case → {detail_path}")

    summary = df.groupby(['model', 'strategy']).agg(
        WT_mean=('WT', 'mean'), TC_mean=('TC', 'mean'),
        ET_mean=('ET', 'mean'), NCR_mean=('NCR', 'mean'),
        ED_mean=('ED', 'mean'), NCR_std=('NCR', 'std'),
        n_cases=('case', 'count'),
    ).round(4).reset_index()

    summary_path = os.path.join(PP_OUTPUT_DIR, 'conditional_summary.csv')
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print(f"  CONDITIONAL POST-PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print(f"\n  Summary → {summary_path}")

    # Best strategy per model
    print(f"\n{'─'*70}")
    print(f"  BEST STRATEGY PER MODEL")
    print(f"{'─'*70}")

    for model_key in df['model'].unique():
        model_df = summary[summary['model'] == model_key]
        baseline = model_df[model_df['strategy'] == 'none']
        bl_ncr = float(baseline['NCR_mean'].values[0])
        bl_et = float(baseline['ET_mean'].values[0])
        bl_wt = float(baseline['WT_mean'].values[0])
        bl_tc = float(baseline['TC_mean'].values[0])

        print(f"\n  {model_key}:")
        print(f"    {'Strategy':<16} {'NCR':>7} {'ET':>7} {'WT':>7} {'TC':>7}  {'NCR gain':>9} {'ET drop':>8}")
        print(f"    {'-'*70}")

        for _, row in model_df.iterrows():
            strat = row['strategy']
            ncr = row['NCR_mean']
            et = row['ET_mean']
            wt = row['WT_mean']
            tc = row['TC_mean']
            ncr_d = ncr - bl_ncr
            et_d = et - bl_et
            marker = ' ◀ BEST' if strat.startswith('cond_') and ncr >= model_df[model_df['strategy'].str.startswith('cond_')]['NCR_mean'].max() - 0.001 else ''
            print(f"    {strat:<16} {ncr:>7.4f} {et:>7.4f} {wt:>7.4f} {tc:>7.4f}  {ncr_d:>+9.4f} {et_d:>+8.4f}{marker}")


def main():
    parser = argparse.ArgumentParser(
        description='Conditional Otsu post-processing on all models')
    parser.add_argument('--models', nargs='+',
                        default=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  CONDITIONAL POST-PROCESSING ANALYSIS")
    print(f"  Models: {args.models}")
    print(f"  Strategies: {list(PP_STRATEGIES.keys())}")
    print(f"  Output: {PP_OUTPUT_DIR}")
    print(f"{'='*70}")

    holdout_ids = get_holdout_ids()
    print(f"  Holdout: {len(holdout_ids)} cases")

    all_results = []
    for model_key in args.models:
        if model_key not in MODEL_CONFIGS:
            print(f"  WARNING: {model_key} not found — skipping")
            continue
        results = evaluate_model(model_key, MODEL_CONFIGS[model_key],
                                 holdout_ids)
        all_results.extend(results)

    if all_results:
        summarise_and_save(all_results)
    else:
        print("\n  No results.")

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
