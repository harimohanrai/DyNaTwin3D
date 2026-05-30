"""
dynatwin/main.py  — 3D Edition
================================
Run:  python -m dynatwin.main
Automatically skips models already present in ablation_results.csv
so training can resume safely after a crash without rerunning
completed models.

NOTE: M1_ResUNet3D, M2_ResAttUNet3D, M3_ASPP_AttDS3D already completed.
      Only M3Plus is queued for training.
"""
import os
import pandas as pd
from dynatwin.config        import CSV_PATH, OUTPUT_DIR
from dynatwin.data_pipeline import load_survival_df, get_stratified_split
from dynatwin.models        import (build_unet3d_m1, build_unet3d_m2,
                                    build_unet3d_m3, build_unet3d_m3plus)
from dynatwin.train         import run_one_model
from dynatwin.statistics    import compare_models
from dynatwin.visualize     import plot_metric_comparison


def train_all():
    surv_df = load_survival_df()
    holdout_ids, folds = get_stratified_split(surv_df)
    print(f"\n{'='*70}")
    print(f"  Holdout : {len(holdout_ids)} cases")
    print(f"  Folds   : {len(folds)} x "
          f"(train={len(folds[0][0])}, val={len(folds[0][1])})")
    print(f"{'='*70}")

    # ── Skip models already completed in a previous run ───────────
    completed = []
    if os.path.exists(CSV_PATH):
        completed = pd.read_csv(CSV_PATH)['Model'].tolist()
        if completed:
            print(f"\n  Already completed (will skip): {completed}")

    all_results = {}

    # M1, M2, M3 already completed — only M3Plus is queued.
    # To re-enable the others, uncomment their entries below:
    for build_fn, key in [
        # (build_unet3d_m1, 'M1_ResUNet3D'),
        # (build_unet3d_m2, 'M2_ResAttUNet3D'),
        # (build_unet3d_m3, 'M3_ASPP_AttDS3D'),
        (build_unet3d_m3plus, 'M3Plus'),
    ]:
        if key in completed:
            print(f"\n  [{key}] found in CSV — skipping.")
            continue

        print(f"\n  [{key}] starting...")
        _, results = run_one_model(
            build_fn    = build_fn,
            model_key   = key,
            folds       = folds,
            holdout_ids = holdout_ids,
            surv_df     = surv_df,
            use_tta     = True,
        )
        all_results[key] = results

    # ── Final comparison across all models ────────────────────────
    print(f"\n{'='*70}\n  FINAL RESULTS\n{'='*70}")
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH).sort_values('WT_Dice', ascending=False)
        print(df.to_string(index=False))

    if all_results:
        for metric in ['dice_whole_tumour', 'dice_tumour_core', 'dice_enhancing']:
            cmp = compare_models(all_results, metric=metric)
            print(f"\n{metric}:\n{cmp.to_string(index=False)}")
            cmp.to_csv(
                os.path.join(OUTPUT_DIR, f'comparison_{metric}.csv'),
                index=False)
            plot_metric_comparison(all_results, metric=metric)
    else:
        print("\n  No new models trained this session — "
              "all models were already in CSV.")

    print(f"\n  All outputs → {OUTPUT_DIR}")


if __name__ == '__main__':
    train_all()
