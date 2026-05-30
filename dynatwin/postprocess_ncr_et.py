"""
postprocess_ncr_et.py — Standalone T1ce Intensity-Based Post-Processing
========================================================================
PURPOSE:
  Test whether a simple T1ce intensity threshold within predicted
  tumour core can improve NCR vs ET discrimination.

WHAT IT DOES:
  1. Loads each trained model (M1, M2, M3)
  2. Runs inference on holdout cases
  3. Applies T1ce-based post-processing to split NCR/ET within TC
  4. Computes Dice BEFORE and AFTER post-processing
  5. Saves comparison CSV

DOES NOT MODIFY ANY EXISTING CODE.
  - Imports from dynatwin modules are READ-ONLY
  - All outputs go to a new subfolder: outputs/postprocess_analysis/
  - Original model checkpoints, configs, logs are untouched

USAGE:
  cd /home/ubuntu/Hari/3D_DynaTwin
  source venv_3d/bin/activate
  python postprocess_ncr_et.py

  Or for a single model:
  python postprocess_ncr_et.py --models M1_ResUNet3D

REQUIREMENTS:
  - Trained model checkpoints in outputs/
  - BraTS dataset accessible via dynatwin.data_pipeline.load_volume
"""

import os, sys, argparse, gc, time
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Ensure dynatwin is importable ────────────────────────────────
# Add project root to path if running from outside
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == 'outputs':
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dynatwin.config import (
    PATCH_SIZE, NUM_CLASSES, OUTPUT_DIR, CSV_PATH,
    N_FOLDS, TRAIN_DATASET_PATH,
)
from dynatwin.losses import get_custom_objects
from dynatwin.data_pipeline import load_volume, clear_vol_cache
from dynatwin.evaluate_fixed import (
    sliding_window_inference, morphological_postprocess,
)


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

# BraTS 2020 modality order: T1, T1ce, T2, FLAIR
# T1ce is channel index 1 (0-indexed)
T1CE_CHANNEL = 1

# Output directory for post-processing analysis
PP_OUTPUT_DIR = os.path.join('/home/ubuntu/Hari/3D_DynaTwin/outputs', 'postprocess_analysis')
os.makedirs(PP_OUTPUT_DIR, exist_ok=True)

# Models and their best checkpoint paths
# Each entry: (model_key, checkpoint_path)
# Adjust fold index if your best fold differs
_WEIGHTS_DIR = "/home/ubuntu/Hari/3D_DynaTwin/outputs"

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

# BraTS label convention
LABEL_BG  = 0
LABEL_NCR = 1
LABEL_ED  = 2
LABEL_ET  = 3


# ══════════════════════════════════════════════════════════════════
# Holdout case IDs — same split used in training
# ══════════════════════════════════════════════════════════════════

def get_holdout_ids():
    from dynatwin.data_pipeline import load_survival_df, get_stratified_split
    surv_df = load_survival_df()
    holdout_ids, _ = get_stratified_split(surv_df)
    print(f"  Holdout cases: {len(holdout_ids)}")
    return holdout_ids


# ══════════════════════════════════════════════════════════════════
# Dice computation (standalone — no dependency on losses.py metrics)
# ══════════════════════════════════════════════════════════════════

def dice_score(pred_mask, gt_mask, smooth=1e-6):
    """
    Compute Dice score between two binary masks.
    Returns float Dice, or np.nan if both masks are empty.
    """
    pred_flat = pred_mask.flatten().astype(np.float64)
    gt_flat   = gt_mask.flatten().astype(np.float64)

    # If both empty — label doesn't exist in this case
    if pred_flat.sum() == 0 and gt_flat.sum() == 0:
        return np.nan

    intersection = (pred_flat * gt_flat).sum()
    return float(
        (2.0 * intersection + smooth)
        / (pred_flat.sum() + gt_flat.sum() + smooth)
    )


def compute_all_dice(pred_seg, gt_seg):
    """
    Compute Dice for all BraTS regions from argmax label maps.

    Args:
        pred_seg: 3D array of predicted class labels (0,1,2,3)
        gt_seg:   3D array of ground truth class labels (0,1,2,3)

    Returns:
        dict with WT, TC, ET, NCR, ED Dice scores
    """
    # Whole Tumour: labels 1+2+3
    wt_pred = (pred_seg > 0).astype(np.uint8)
    wt_gt   = (gt_seg   > 0).astype(np.uint8)

    # Tumour Core: labels 1+3
    tc_pred = np.isin(pred_seg, [1, 3]).astype(np.uint8)
    tc_gt   = np.isin(gt_seg,   [1, 3]).astype(np.uint8)

    # Enhancing Tumour: label 3
    et_pred = (pred_seg == 3).astype(np.uint8)
    et_gt   = (gt_seg   == 3).astype(np.uint8)

    # Necrotic Core: label 1
    ncr_pred = (pred_seg == 1).astype(np.uint8)
    ncr_gt   = (gt_seg   == 1).astype(np.uint8)

    # Edema: label 2
    ed_pred = (pred_seg == 2).astype(np.uint8)
    ed_gt   = (gt_seg   == 2).astype(np.uint8)

    return {
        'WT':  dice_score(wt_pred,  wt_gt),
        'TC':  dice_score(tc_pred,  tc_gt),
        'ET':  dice_score(et_pred,  et_gt),
        'NCR': dice_score(ncr_pred, ncr_gt),
        'ED':  dice_score(ed_pred,  ed_gt),
    }


# ══════════════════════════════════════════════════════════════════
# T1ce-based post-processing strategies
# ══════════════════════════════════════════════════════════════════

def postprocess_t1ce_mean(pred_seg, t1ce_volume):
    """
    Strategy 1: Mean-based T1ce split within tumour core.

    Within predicted TC (labels 1 and 3):
      - Compute mean T1ce intensity of all TC voxels
      - Below mean → NCR (hypointense = dead tissue)
      - Above mean → ET  (hyperintense = enhancing)

    Non-TC voxels (background, edema) are unchanged.

    Args:
        pred_seg:     3D argmax prediction (labels 0-3)
        t1ce_volume:  3D T1ce MRI volume (same spatial dims)

    Returns:
        corrected_seg: 3D array with corrected NCR/ET labels
    """
    corrected = pred_seg.copy()

    # Find tumour core voxels (NCR=1 or ET=3)
    tc_mask = np.isin(pred_seg, [LABEL_NCR, LABEL_ET])
    n_tc = tc_mask.sum()

    if n_tc == 0:
        return corrected  # no tumour core — nothing to do

    # T1ce intensities within tumour core
    tc_intensities = t1ce_volume[tc_mask]
    threshold = np.mean(tc_intensities)

    # Reclassify: below threshold → NCR, above → ET
    tc_voxels_t1ce = t1ce_volume[tc_mask]
    new_labels = np.where(
        tc_voxels_t1ce < threshold, LABEL_NCR, LABEL_ET
    )
    corrected[tc_mask] = new_labels

    return corrected


def postprocess_t1ce_median(pred_seg, t1ce_volume):
    """
    Strategy 2: Median-based T1ce split.
    More robust to outliers than mean.
    """
    corrected = pred_seg.copy()
    tc_mask = np.isin(pred_seg, [LABEL_NCR, LABEL_ET])

    if tc_mask.sum() == 0:
        return corrected

    tc_intensities = t1ce_volume[tc_mask]
    threshold = np.median(tc_intensities)

    new_labels = np.where(
        tc_intensities < threshold, LABEL_NCR, LABEL_ET
    )
    corrected[tc_mask] = new_labels
    return corrected


def postprocess_t1ce_otsu(pred_seg, t1ce_volume):
    """
    Strategy 3: Otsu's threshold within tumour core.
    Finds the optimal bimodal split between NCR and ET intensities.
    No external dependency — implements Otsu from scratch.
    """
    corrected = pred_seg.copy()
    tc_mask = np.isin(pred_seg, [LABEL_NCR, LABEL_ET])

    if tc_mask.sum() < 10:  # too few voxels for reliable threshold
        return corrected

    tc_intensities = t1ce_volume[tc_mask]

    # Otsu's method
    # Quantise intensities into 256 bins for efficiency
    hist_min = tc_intensities.min()
    hist_max = tc_intensities.max()

    if hist_max - hist_min < 1e-6:
        return corrected  # uniform intensity — can't split

    # Normalise to [0, 255]
    normalised = ((tc_intensities - hist_min)
                  / (hist_max - hist_min) * 255).astype(np.int32)
    normalised = np.clip(normalised, 0, 255)

    # Build histogram
    hist = np.bincount(normalised, minlength=256).astype(np.float64)
    hist /= hist.sum()

    best_thresh = 0
    best_var = 0.0

    cum_sum_0 = 0.0
    cum_weight_0 = 0.0

    total_mean = np.sum(np.arange(256) * hist)

    for t in range(256):
        cum_weight_0 += hist[t]
        cum_sum_0 += t * hist[t]

        if cum_weight_0 == 0 or cum_weight_0 == 1:
            continue

        cum_weight_1 = 1.0 - cum_weight_0
        mean_0 = cum_sum_0 / cum_weight_0
        mean_1 = (total_mean - cum_sum_0) / cum_weight_1

        between_var = (cum_weight_0 * cum_weight_1
                       * (mean_0 - mean_1) ** 2)

        if between_var > best_var:
            best_var = between_var
            best_thresh = t

    # Convert threshold back to original intensity scale
    threshold = hist_min + (best_thresh / 255.0) * (hist_max - hist_min)

    new_labels = np.where(
        tc_intensities < threshold, LABEL_NCR, LABEL_ET
    )
    corrected[tc_mask] = new_labels
    return corrected


def postprocess_t1ce_adaptive(pred_seg, t1ce_volume):
    """
    Strategy 4: Adaptive — use GT ET intensity profile as guide.

    Within predicted TC:
      - If model predicted ANY ET voxels, use the lower quartile
        of predicted-ET T1ce intensities as the split threshold.
        (Rationale: the dimmest "enhancing" voxels are likely
        misclassified NCR)
      - If model predicted NO ET, fall back to median split.

    This leverages the model's own partial knowledge — it gets
    ET mostly right, it just over-assigns it.
    """
    corrected = pred_seg.copy()
    tc_mask = np.isin(pred_seg, [LABEL_NCR, LABEL_ET])

    if tc_mask.sum() == 0:
        return corrected

    # Check if model predicted any ET at all
    et_mask = (pred_seg == LABEL_ET)
    n_et = et_mask.sum()

    if n_et > 10:
        # Use lower quartile of predicted-ET T1ce as threshold
        et_intensities = t1ce_volume[et_mask]
        threshold = np.percentile(et_intensities, 25)
    else:
        # Fallback: median of all TC
        tc_intensities = t1ce_volume[tc_mask]
        threshold = np.median(tc_intensities)

    tc_intensities = t1ce_volume[tc_mask]
    new_labels = np.where(
        tc_intensities < threshold, LABEL_NCR, LABEL_ET
    )
    corrected[tc_mask] = new_labels
    return corrected


# All strategies to test
def postprocess_cond_voxel(pred_seg, t1ce, min_voxels=100):
    ncr_count = (pred_seg == LABEL_NCR).sum()
    if ncr_count >= min_voxels:
        return pred_seg
    return postprocess_t1ce_adaptive(pred_seg, t1ce)

def postprocess_cond_v200(pred_seg, t1ce):
    return postprocess_cond_voxel(pred_seg, t1ce, 200)

def postprocess_cond_v500(pred_seg, t1ce):
    return postprocess_cond_voxel(pred_seg, t1ce, 500)

PP_STRATEGIES = {
    'none':       None,
    'adaptive':   postprocess_t1ce_adaptive,
    'cond_v100':  postprocess_cond_voxel,
    'cond_v200':  postprocess_cond_v200,
    'cond_v500':  postprocess_cond_v500,
}


# ══════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════

def evaluate_model_with_postprocessing(model_key, ckpt_path,
                                        holdout_ids):
    """
    Run inference + all post-processing strategies on one model.

    Returns:
        List of dicts, one per (case, strategy) combination.
    """
    print(f"\n{'─'*60}")
    print(f"  Model: {model_key}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'─'*60}")

    if not os.path.exists(ckpt_path):
        print(f"  ERROR: checkpoint not found — skipping")
        return []

    # Load model
    custom_objs = get_custom_objects()
    model = tf.keras.models.load_model(
        ckpt_path, custom_objects=custom_objs,
        compile=False, safe_mode=False,
    )
    print(f"  Loaded: {model.count_params():,} params")

    results = []

    for idx, case_id in enumerate(holdout_ids):
        print(f"  {idx+1}/{len(holdout_ids)}  {case_id}  ", end='')

        try:
            X, Y = load_volume(case_id)
        except Exception as e:
            print(f"LOAD ERROR: {e}")
            continue

        # Ground truth: one-hot → argmax
        if Y.ndim > 3 and Y.shape[-1] > 1: gt_seg = np.argmax(Y, axis=-1)
        else: gt_seg = Y.squeeze() if Y.ndim > 3 else Y

        # T1ce volume for post-processing
        t1ce = X[..., T1CE_CHANNEL]  # 3D volume

        # Model inference (same as your evaluate_fixed.py)
        _ps = model.input_shape[1:4]; import dynatwin.evaluate_fixed as _ef; _ef._GAUSS = _ef._gaussian_kernel(_ps); pred_soft = sliding_window_inference(model, X, patch_size=_ps)
        pred_seg_base = morphological_postprocess(pred_soft)
        # pred_seg_base is argmax label map (0-3)

        # If morphological_postprocess returns soft probs, convert
        if pred_seg_base.ndim > 3 and pred_seg_base.shape[-1] > 1:
            pred_seg_base = np.argmax(pred_seg_base, axis=-1)

        # Evaluate each strategy
        for strat_name, strat_fn in PP_STRATEGIES.items():
            if strat_fn is None:
                pred_final = pred_seg_base
            else:
                pred_final = strat_fn(pred_seg_base.copy(), t1ce)

            dice = compute_all_dice(pred_final, gt_seg)

            row = {
                'model':    model_key,
                'case':     case_id,
                'strategy': strat_name,
                'WT':       dice['WT'],
                'TC':       dice['TC'],
                'ET':       dice['ET'],
                'NCR':      dice['NCR'],
                'ED':       dice['ED'],
            }
            results.append(row)

        # Print baseline vs best post-processed NCR for this case
        baseline_ncr = results[-5]['NCR']  # 'none' is first strategy
        pp_ncrs = {r['strategy']: r['NCR']
                   for r in results[-5:] if r['strategy'] != 'none'}
        best_strat = max(pp_ncrs, key=lambda k: pp_ncrs[k]
                         if not np.isnan(pp_ncrs[k]) else -1)
        best_ncr = pp_ncrs[best_strat]

        b_str = f'{baseline_ncr:.3f}' if not np.isnan(baseline_ncr) else 'nan'
        a_str = f'{best_ncr:.3f}' if not np.isnan(best_ncr) else 'nan'
        delta = ''
        if not np.isnan(baseline_ncr) and not np.isnan(best_ncr):
            d = best_ncr - baseline_ncr
            delta = f' ({"+" if d >= 0 else ""}{d:.3f})'

        print(f"NCR: {b_str} → {a_str} [{best_strat}]{delta}")

    del model
    gc.collect()
    tf.keras.backend.clear_session()
    clear_vol_cache()

    return results


def summarise_and_save(all_results):
    """
    Build summary tables and save everything.
    """
    df = pd.DataFrame(all_results)

    # ── Per-case detail CSV ───────────────────────────────────────
    detail_path = os.path.join(PP_OUTPUT_DIR, 'postprocess_per_case.csv')
    df.to_csv(detail_path, index=False)
    print(f"\n  Per-case results saved → {detail_path}")

    # ── Summary: mean Dice per (model, strategy) ─────────────────
    summary = df.groupby(['model', 'strategy']).agg(
        WT_mean=('WT', 'mean'),
        TC_mean=('TC', 'mean'),
        ET_mean=('ET', 'mean'),
        NCR_mean=('NCR', 'mean'),
        ED_mean=('ED', 'mean'),
        NCR_std=('NCR', 'std'),
        n_cases=('case', 'count'),
    ).round(4).reset_index()

    summary_path = os.path.join(PP_OUTPUT_DIR, 'postprocess_summary.csv')
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print(f"  POST-PROCESSING COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print(f"\n  Summary saved → {summary_path}")

    # ── Best strategy per model ──────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  BEST STRATEGY PER MODEL (by NCR Dice)")
    print(f"{'─'*70}")

    for model_key in df['model'].unique():
        model_df = summary[summary['model'] == model_key]
        baseline = model_df[model_df['strategy'] == 'none']
        best_row = model_df.loc[model_df['NCR_mean'].idxmax()]

        bl_ncr = float(baseline['NCR_mean'].values[0])
        best_ncr = float(best_row['NCR_mean'])
        best_strat = best_row['strategy']
        delta = best_ncr - bl_ncr

        print(f"  {model_key}:")
        print(f"    Baseline NCR:       {bl_ncr:.4f}")
        print(f"    Best post-proc NCR: {best_ncr:.4f}  "
              f"[{best_strat}]  ({'+' if delta >= 0 else ''}{delta:.4f})")

        # Also check: did post-processing hurt other metrics?
        bl_wt = float(baseline['WT_mean'].values[0])
        bl_tc = float(baseline['TC_mean'].values[0])
        bl_et = float(baseline['ET_mean'].values[0])
        best_full = model_df[model_df['strategy'] == best_strat]
        pp_wt = float(best_full['WT_mean'].values[0])
        pp_tc = float(best_full['TC_mean'].values[0])
        pp_et = float(best_full['ET_mean'].values[0])

        print(f"    WT: {bl_wt:.4f} → {pp_wt:.4f}  "
              f"({'OK' if pp_wt >= bl_wt - 0.01 else 'DEGRADED'})")
        print(f"    TC: {bl_tc:.4f} → {pp_tc:.4f}  "
              f"({'OK' if pp_tc >= bl_tc - 0.01 else 'DEGRADED'})")
        print(f"    ET: {bl_et:.4f} → {pp_et:.4f}  "
              f"({'OK' if pp_et >= bl_et - 0.01 else 'DEGRADED'})")
        print()

    # ── Cases where post-processing helped most ──────────────────
    print(f"{'─'*70}")
    print(f"  TOP 10 CASES WITH BIGGEST NCR IMPROVEMENT")
    print(f"{'─'*70}")

    for model_key in df['model'].unique():
        model_df = df[df['model'] == model_key]
        # Find best strategy from summary
        model_summary = summary[summary['model'] == model_key]
        best_strat = model_summary.loc[
            model_summary['NCR_mean'].idxmax(), 'strategy']

        baseline_df = model_df[model_df['strategy'] == 'none'][
            ['case', 'NCR']].rename(columns={'NCR': 'NCR_before'})
        pp_df = model_df[model_df['strategy'] == best_strat][
            ['case', 'NCR']].rename(columns={'NCR': 'NCR_after'})

        merged = baseline_df.merge(pp_df, on='case')
        merged['delta'] = merged['NCR_after'] - merged['NCR_before']
        merged = merged.sort_values('delta', ascending=False).head(10)

        print(f"\n  {model_key} [{best_strat}]:")
        for _, row in merged.iterrows():
            b = f"{row['NCR_before']:.3f}" if not np.isnan(
                row['NCR_before']) else '  nan'
            a = f"{row['NCR_after']:.3f}" if not np.isnan(
                row['NCR_after']) else '  nan'
            d = f"{row['delta']:+.3f}" if not np.isnan(
                row['delta']) else '  nan'
            print(f"    {row['case']}  {b} → {a}  ({d})")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Test T1ce intensity-based NCR/ET post-processing')
    parser.add_argument(
        '--models', nargs='+',
        default=list(MODEL_CONFIGS.keys()),
        help='Model keys to evaluate (default: all)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  T1ce POST-PROCESSING ANALYSIS")
    print(f"  Models: {args.models}")
    print(f"  Strategies: {list(PP_STRATEGIES.keys())}")
    print(f"  Output: {PP_OUTPUT_DIR}")
    print(f"{'='*70}")

    holdout_ids = get_holdout_ids()
    print(f"  Holdout IDs: {holdout_ids[:5]}... ({len(holdout_ids)} total)")

    all_results = []

    for model_key in args.models:
        if model_key not in MODEL_CONFIGS:
            print(f"  WARNING: {model_key} not in MODEL_CONFIGS — skipping")
            continue
        ckpt_path = MODEL_CONFIGS[model_key]
        results = evaluate_model_with_postprocessing(
            model_key, ckpt_path, holdout_ids)
        all_results.extend(results)

    if all_results:
        summarise_and_save(all_results)
    else:
        print("\n  No results — check model paths and holdout IDs.")

    print(f"\n  Done. All outputs in {PP_OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
