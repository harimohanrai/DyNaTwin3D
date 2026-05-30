# -*- coding: utf-8 -*-
"""
dynatwin/re_evaluate.py
========================
Re-evaluate all three saved models on the holdout set using the
corrected evaluate.py. Run this after M3 finishes training.

Usage:
    cd ~/Hari/3D_DynaTwin
    source venv_3d/bin/activate
    OUTPUT_DIR=~/Hari/3D_DynaTwin/outputs python -m dynatwin.re_evaluate

What it does:
  1. Loads holdout IDs using the same seed/split as training
  2. For each model finds the best fold checkpoint
  3. Runs corrected sliding-window + TTA + morphological inference
  4. Saves per-case and summary CSVs
  5. Prints final comparison table
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from dynatwin.config        import OUTPUT_DIR, CSV_PATH, strategy
from dynatwin.data_pipeline import load_survival_df, get_stratified_split
from dynatwin.losses        import get_custom_objects
from dynatwin.evaluate_fixed      import evaluate_set
from dynatwin.statistics    import summarise_results


# ── Model registry ──────────────────────────────────────────────────
MODELS = [
    'M1_ResUNet3D',
    'M2_ResAttUNet3D',
    'M3_ASPP_AttDS3D',
]


def _find_best_checkpoint(model_key: str) -> str:
    """
    Find the best fold checkpoint for a model by reading fold logs
    and picking the fold with highest val_seg_any_tumour_dice.
    """
    import csv
    best_score = -1.0
    best_ckpt  = None

    for fold_idx in range(10):   # try up to 10 folds
        ckpt_path = os.path.join(OUTPUT_DIR,
                                 f'{model_key}_fold{fold_idx}_best.keras')
        log_path  = os.path.join(OUTPUT_DIR,
                                 f'{model_key}_fold{fold_idx}.log')

        if not os.path.exists(ckpt_path):
            break

        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    rows = list(csv.DictReader(f))
                scores = [float(r['val_seg_any_tumour_dice'])
                          for r in rows
                          if 'val_seg_any_tumour_dice' in r]
                fold_best = max(scores) if scores else -1.0
            except Exception:
                fold_best = -1.0
        else:
            fold_best = -1.0

        if fold_best > best_score:
            best_score = fold_best
            best_ckpt  = ckpt_path

    return best_ckpt


def re_evaluate_all():
    surv_df     = load_survival_df()
    holdout_ids, _ = get_stratified_split(surv_df)

    print(f"\n{'='*70}")
    print(f"  RE-EVALUATION with corrected evaluate.py")
    print(f"  Holdout: {len(holdout_ids)} cases")
    print(f"{'='*70}")

    summary_rows = []
    custom_objs  = get_custom_objects()

    for model_key in MODELS:
        ckpt = _find_best_checkpoint(model_key)
        if ckpt is None:
            print(f"\n  [{model_key}] No checkpoint found — skipping.")
            continue

        print(f"\n  [{model_key}] Loading: {ckpt}")
        tf.keras.backend.clear_session()

        with strategy.scope():
            model = tf.keras.models.load_model(
                ckpt, custom_objects=custom_objs,
                compile=False, safe_mode=False)

        print(f"  [{model_key}] Evaluating {len(holdout_ids)} cases ...")
        results = evaluate_set(model, holdout_ids,
                               use_tta=True, desc=model_key)

        # Save per-case CSV
        per_case_df = pd.DataFrame(results)
        per_case_path = os.path.join(OUTPUT_DIR,
                                     f'{model_key}_reeval_per_case.csv')
        per_case_df.to_csv(per_case_path, index=False)

        # Summary
        summary_df = summarise_results(results)
        summary_path = os.path.join(OUTPUT_DIR,
                                    f'{model_key}_reeval_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  {model_key} re-evaluation summary:")
        print(summary_df.to_string(index=False))

        def _mean(k):
            return float(np.nanmean(
                [r.get(k, float('nan')) for r in results]))

        summary_rows.append({
            'Model':    model_key,
            'WT_Dice':  round(_mean('dice_whole_tumour'), 4),
            'TC_Dice':  round(_mean('dice_tumour_core'),  4),
            'ET_Dice':  round(_mean('dice_enhancing'),    4),
            'Nec_Dice': round(_mean('dice_necrotic'),     4),
            'Edema_Dice': round(_mean('dice_edema'),      4),
            'WT_HD95':  round(_mean('hd95_whole_tumour'), 2),
            'TC_HD95':  round(_mean('hd95_tumour_core'),  2),
        })

        del model
        import gc; gc.collect()

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — corrected evaluate.py")
    print(f"{'='*70}")
    comparison_df = pd.DataFrame(summary_rows)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(
        os.path.join(OUTPUT_DIR, 'reeval_comparison.csv'), index=False)
    print(f"\n  Saved → {OUTPUT_DIR}/reeval_comparison.csv")


if __name__ == '__main__':
    re_evaluate_all()
