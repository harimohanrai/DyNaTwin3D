# -*- coding: utf-8 -*-
"""
dynatwin/re_evaluate_v2.py
===========================
Re-evaluate all three saved models on the holdout set.

Key difference from re_evaluate.py:
  - Checkpoint selected by val_seg_dice_coef (mean of Nec+Edema+ET Dice)
    instead of val_seg_any_tumour_dice.
    This picks the checkpoint that was best at subregion segmentation,
    not just the whole tumour blob — which is why all three models
    looked identical in the original evaluation.

  - All results saved to outputs_v2/ — outputs/ is never touched.

  - Live per-case printing matches the format:
      [M3_ASPP_AttDS3D] 1/37  BraTS20_Training_041
      WT=0.925  TC=0.913  ET=0.783  Nec=0.790

  - After each model, prints a case-level diff table vs original results.

Usage:
    cd ~/Hari/3D_DynaTwin
    source venv_3d/bin/activate
    python -m dynatwin.re_evaluate_v2
"""

import os
import gc
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

from dynatwin.config        import OUTPUT_DIR, strategy
from dynatwin.data_pipeline import load_survival_df, get_stratified_split
from dynatwin.losses        import get_custom_objects
from dynatwin.evaluate_fixed      import evaluate_set
from dynatwin.statistics    import summarise_results

# ── New output folder — completely separate from original outputs/ ──
OUTPUT_DIR_V2 = os.path.join(os.path.dirname(OUTPUT_DIR), 'outputs_v2')
os.makedirs(OUTPUT_DIR_V2, exist_ok=True)
print(f"[re_evaluate_v2] Saving all results to: {OUTPUT_DIR_V2}")
print(f"[re_evaluate_v2] Original results in:   {OUTPUT_DIR}  (untouched)")

MODELS = [
    'M1_ResUNet3D',
    'M2_ResAttUNet3D',
    'M3_ASPP_AttDS3D',
]


# ══════════════════════════════════════════════════════════════════
# Checkpoint selector — uses subregion Dice, not binary tumour Dice
# ══════════════════════════════════════════════════════════════════

def _find_best_checkpoint_by_subregion(model_key):
    """
    Reads fold logs from outputs/ and selects the fold checkpoint
    with the highest val_seg_dice_coef (mean of Nec+Edema+ET Dice).

    Falls back to val_seg_any_tumour_dice if val_seg_dice_coef column
    is not present in the log (older training run).

    Returns (ckpt_path, best_fold_idx, best_score).
    """
    best_score = -1.0
    best_ckpt  = None
    best_fold  = -1

    for fold_idx in range(10):
        ckpt_path = os.path.join(OUTPUT_DIR,
                                 f'{model_key}_fold{fold_idx}_best.keras')
        log_path  = os.path.join(OUTPUT_DIR,
                                 f'{model_key}_fold{fold_idx}.log')

        if not os.path.exists(ckpt_path):
            break

        fold_score   = -1.0
        metric_used  = 'none'

        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    rows = list(csv.DictReader(f))

                if rows and 'val_seg_dice_coef' in rows[0]:
                    scores = [float(r['val_seg_dice_coef'])
                              for r in rows
                              if r.get('val_seg_dice_coef', '').strip()]
                    metric_used = 'val_seg_dice_coef'
                else:
                    scores = [float(r['val_seg_any_tumour_dice'])
                              for r in rows
                              if r.get('val_seg_any_tumour_dice', '').strip()]
                    metric_used = 'val_seg_any_tumour_dice (fallback)'

                fold_score = max(scores) if scores else -1.0

            except Exception as e:
                print(f"  [{model_key}] fold{fold_idx} log read error: {e}")

        print(f"  [{model_key}] fold{fold_idx}  "
              f"metric={metric_used}  score={fold_score:.4f}")

        if fold_score > best_score:
            best_score = fold_score
            best_ckpt  = ckpt_path
            best_fold  = fold_idx

    if best_ckpt is None:
        print(f"  [{model_key}] ERROR: no checkpoint found in {OUTPUT_DIR}")
    else:
        print(f"  [{model_key}] --> Selected fold{best_fold}  "
              f"score={best_score:.4f}  ckpt={best_ckpt}")

    return best_ckpt, best_fold, best_score


# ══════════════════════════════════════════════════════════════════
# Per-case diff printer
# ══════════════════════════════════════════════════════════════════

def _print_diff_table(model_key, results_v2):
    """
    Loads original per-case CSV from outputs/ and prints a side-by-side
    diff for every holdout case.  Positive = V2 is better.
    """
    # Try both naming conventions the original script may have used
    candidates = [
        os.path.join(OUTPUT_DIR, f'{model_key}_reeval_per_case.csv'),
        os.path.join(OUTPUT_DIR, f'{model_key}_holdout_per_case.csv'),
    ]
    orig_path = None
    for c in candidates:
        if os.path.exists(c):
            orig_path = c
            break

    if orig_path is None:
        print(f"\n  [diff] No original per-case CSV found for {model_key}")
        print(f"  [diff] Looked for:")
        for c in candidates:
            print(f"           {c}")
        print(f"  [diff] Skipping diff table.")
        return

    orig_df = pd.read_csv(orig_path).set_index('case_id')

    sep = '  ' + '─' * 70
    print(f"\n{sep}")
    print(f"  CASE-LEVEL DIFF — {model_key}")
    print(f"  V2 checkpoint (val_seg_dice_coef)  vs  "
          f"Original (val_seg_any_tumour_dice)")
    print(f"  Positive values = V2 is better")
    print(sep)
    print(f"  {'Case ID':<32} {'WT':>7} {'TC':>7} {'ET':>7} {'Nec':>7}  "
          f"{'Edema':>7}")
    print(sep)

    wt_diffs   = []
    tc_diffs   = []
    et_diffs   = []
    nec_diffs  = []
    edema_diffs= []

    for r in results_v2:
        cid = r['case_id']
        if cid not in orig_df.index:
            print(f"  {cid:<32}  (not in original results — skipping)")
            continue

        o = orig_df.loc[cid]

        def _d(new_key, old_key):
            nv = r.get(new_key, float('nan'))
            ov = o.get(old_key, float('nan'))
            if pd.isna(nv) or pd.isna(ov):
                return float('nan')
            return float(nv) - float(ov)

        wt_d   = _d('dice_whole_tumour', 'dice_whole_tumour')
        tc_d   = _d('dice_tumour_core',  'dice_tumour_core')
        et_d   = _d('dice_enhancing',    'dice_enhancing')
        nec_d  = _d('dice_necrotic',     'dice_necrotic')
        edema_d= _d('dice_edema',        'dice_edema')

        def _fmt(v):
            if pd.isna(v):
                return '    nan'
            return f'{v:>+7.3f}'

        print(f"  {cid:<32} {_fmt(wt_d)} {_fmt(tc_d)} "
              f"{_fmt(et_d)} {_fmt(nec_d)}  {_fmt(edema_d)}")

        wt_diffs.append(wt_d)
        tc_diffs.append(tc_d)
        et_diffs.append(et_d)
        nec_diffs.append(nec_d)
        edema_diffs.append(edema_d)

    print(sep)

    def _mean_diff(lst):
        v = [x for x in lst if not pd.isna(x)]
        return float(np.mean(v)) if v else float('nan')

    def _fmt(v):
        if pd.isna(v):
            return '    nan'
        return f'{v:>+7.3f}'

    print(f"  {'MEAN DIFF':<32} "
          f"{_fmt(_mean_diff(wt_diffs))} "
          f"{_fmt(_mean_diff(tc_diffs))} "
          f"{_fmt(_mean_diff(et_diffs))} "
          f"{_fmt(_mean_diff(nec_diffs))}  "
          f"{_fmt(_mean_diff(edema_diffs))}")
    print(sep)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def re_evaluate_all():
    surv_df     = load_survival_df()
    holdout_ids, _ = get_stratified_split(surv_df)

    print(f"\n{'='*70}")
    print(f"  RE-EVALUATION V2")
    print(f"  Checkpoint metric : val_seg_dice_coef (Nec + Edema + ET mean)")
    print(f"  Holdout cases     : {len(holdout_ids)}")
    print(f"  Output folder     : {OUTPUT_DIR_V2}")
    print(f"{'='*70}")

    summary_rows = []
    custom_objs  = get_custom_objects()

    for model_key in MODELS:

        print(f"\n{'─'*70}")
        print(f"  MODEL: {model_key}")
        print(f"{'─'*70}")

        ckpt, best_fold, best_score = _find_best_checkpoint_by_subregion(
            model_key)

        if ckpt is None:
            print(f"  Skipping {model_key} — no checkpoint found.")
            continue

        print(f"\n  Loading model from: {ckpt}")
        tf.keras.backend.clear_session()

        with strategy.scope():
            model = tf.keras.models.load_model(
                ckpt,
                custom_objects=custom_objs,
                compile=False,
                safe_mode=False,
            )

        print(f"  Parameters: {model.count_params():,}")
        print(f"\n  Running TTA sliding-window inference ...\n")

        # evaluate_set already prints live per-case lines in the format:
        #   [M3_ASPP_AttDS3D] 1/37  BraTS20_Training_041
        #   WT=0.925  TC=0.913  ET=0.783  Nec=0.790
        results = evaluate_set(
            model, holdout_ids,
            use_tta=True,
            desc=model_key,
        )

        # ── Save per-case CSV ────────────────────────────────────
        per_case_df   = pd.DataFrame(results)
        per_case_path = os.path.join(OUTPUT_DIR_V2,
                                     f'{model_key}_v2_per_case.csv')
        per_case_df.to_csv(per_case_path, index=False)
        print(f"\n  Per-case results saved → {per_case_path}")

        # ── Save summary CSV ─────────────────────────────────────
        summary_df   = summarise_results(results)
        summary_path = os.path.join(OUTPUT_DIR_V2,
                                    f'{model_key}_v2_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  {model_key} — V2 Summary:")
        print(summary_df.to_string(index=False))

        # ── Print diff vs original ───────────────────────────────
        _print_diff_table(model_key, results)

        # ── Collect for final comparison table ───────────────────
        def _mean(k):
            return float(np.nanmean(
                [r.get(k, float('nan')) for r in results]))

        summary_rows.append({
            'Model':        model_key,
            'Best_Fold':    best_fold,
            'Ckpt_Score':   round(best_score, 4),
            'WT_Dice':      round(_mean('dice_whole_tumour'), 4),
            'TC_Dice':      round(_mean('dice_tumour_core'),  4),
            'ET_Dice':      round(_mean('dice_enhancing'),    4),
            'Nec_Dice':     round(_mean('dice_necrotic'),     4),
            'Edema_Dice':   round(_mean('dice_edema'),        4),
            'WT_HD95':      round(_mean('hd95_whole_tumour'), 2),
            'TC_HD95':      round(_mean('hd95_tumour_core'),  2),
        })

        del model
        gc.collect()

    # ── Final comparison table ───────────────────────────────────
    comparison_df   = pd.DataFrame(summary_rows)
    comparison_path = os.path.join(OUTPUT_DIR_V2, 'reeval_v2_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)

    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — V2  (checkpoint by val_seg_dice_coef)")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))

    # ── Side-by-side summary diff vs original ────────────────────
    orig_comparison = os.path.join(OUTPUT_DIR, 'reeval_comparison.csv')
    if os.path.exists(orig_comparison):
        orig_df = pd.read_csv(orig_comparison)
        print(f"\n{'='*70}")
        print(f"  SUMMARY DIFF — V2 vs Original  (positive = V2 better)")
        print(f"{'='*70}")
        print(f"  {'Model':<22} {'Metric':<12} {'Original':>10} "
              f"{'V2':>10} {'Diff':>10}")
        print(f"  {'─'*66}")
        for _, row in comparison_df.iterrows():
            orig_row = orig_df[orig_df['Model'] == row['Model']]
            if orig_row.empty:
                continue
            o = orig_row.iloc[0]
            for col in ['WT_Dice', 'TC_Dice', 'ET_Dice',
                        'Nec_Dice', 'Edema_Dice']:
                diff = row[col] - o[col]
                sign = '+' if diff >= 0 else ''
                print(f"  {row['Model']:<22} {col:<12} "
                      f"{o[col]:>10.4f} {row[col]:>10.4f} "
                      f"{sign}{diff:>9.4f}")
            print()

    print(f"\n  All V2 results saved to: {OUTPUT_DIR_V2}")
    print(f"  Original results intact: {OUTPUT_DIR}")


if __name__ == '__main__':
    re_evaluate_all()
