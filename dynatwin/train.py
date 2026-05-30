# -*- coding: utf-8 -*-
"""
dynatwin/train.py  — 3D Edition
==================================
One model runs completely (train → validate → test → visualize → stats)
before the next model starts.

Fixes applied vs previous version:
  Fix 1  _load_and_compile: added is_m3plus guard — aux2/aux3 heads
         were being added on resume for M3Plus incorrectly.

  Fix 2  Resume and fresh-start paths now both pass is_m3plus to
         _build_and_compile and _load_and_compile.

  Fix 3  Original digital_twin import restored alongside v2 import.
         M1/M2/M3 Digital Twin block was crashing with NameError
         because the original import was commented out.

  Fix 4  evaluate_fixed used everywhere instead of evaluate —
         both in the top-level import and inside the Digital Twin
         sliding_window_inference call.

  Fix 5  Monitor key unified — both checkpoint saving and fold best
         score selection use val_seg_any_tumour_dice consistently.

  Fix 6  digital_twin_v2 import wrapped in try/except — if the file
         does not exist yet the entire train.py import still succeeds.
"""

import os, gc, math, time, csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, Callback,
)

from dynatwin.config import (
    PATCH_SIZE, NUM_CLASSES, EPOCHS, BATCH_SIZE, PATCHES_PER_CASE,
    N_FOLDS, OUTPUT_DIR, CSV_PATH, TRAIN_DATASET_PATH, strategy,
)
from dynatwin.losses import (combined_seg_loss, grade_cls_loss,
                              masked_survival_loss, SEG_METRICS,
                              get_custom_objects)

# Fix 4: use evaluate_fixed everywhere
from dynatwin.evaluate_fixed import evaluate_set

from dynatwin.visualize import (plot_training_history, plot_survival_curve,
                                 plot_fold_summary)

# Fix 3: restore original digital_twin import
from dynatwin.digital_twin import (predictions_to_density, PatientDigitalTwin,
                                    PINNCalibrator, SurvivalPredictor)

# Fix 6: wrap v2 import — graceful fallback if file not yet created
try:
    from dynatwin.digital_twin_v2 import (
        predictions_to_density as predictions_to_density_v2,
        PatientDigitalTwin     as PatientDigitalTwin_v2,
        PINNCalibrator         as PINNCalibrator_v2,
        SurvivalPredictor      as SurvivalPredictor_v2,
    )
    _HAS_DT_V2 = True
except ImportError:
    _HAS_DT_V2 = False
    print("[train] WARNING: digital_twin_v2 not found — "
          "M3Plus PINN will fall back to original digital_twin")

from dynatwin.data_pipeline import (make_dataset, load_volume,
                                     survival_info, clear_vol_cache)
from dynatwin.statistics import summarise_results, compare_models


# ══════════════════════════════════════════════════════════════════
# LR schedule
# ══════════════════════════════════════════════════════════════════

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-7):
        super().__init__()
        self.base_lr      = float(base_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        s = tf.cast(step, tf.float32)
        w = self.base_lr * (s / self.warmup_steps)
        c = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * (s - self.warmup_steps) /
                         (self.total_steps - self.warmup_steps)))
        return tf.where(s < self.warmup_steps, w, c)

    def get_config(self):
        return dict(base_lr=self.base_lr, warmup_steps=self.warmup_steps,
                    total_steps=self.total_steps, min_lr=self.min_lr)


def make_optimizer(steps_per_epoch, epochs=EPOCHS,
                   base_lr=3e-4, warmup_epochs=5, weight_decay=2e-4):
    total  = steps_per_epoch * epochs
    warmup = steps_per_epoch * warmup_epochs
    return tf.keras.optimizers.AdamW(
        learning_rate=WarmupCosineDecay(base_lr, warmup, total),
        weight_decay=weight_decay, clipnorm=1.0)


# ══════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════

class OverfitMonitor(Callback):
    def __init__(self, threshold=0.12, patience=5, start_epoch=10):
        super().__init__()
        self.threshold   = threshold
        self.patience    = patience
        self.start_epoch = start_epoch
        self._count      = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        logs = logs or {}
        tr  = logs.get('seg_any_tumour_dice',    0.0)
        val = logs.get('val_seg_any_tumour_dice', 0.0)
        gap = tr - val
        if gap > self.threshold:
            self._count += 1
            print(f'  [OverfitMonitor] epoch={epoch+1} '
                  f'Dice gap={gap:.3f} ({self._count}/{self.patience})')
        else:
            self._count = 0


class LRLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        v  = float(lr(self.model.optimizer.iterations)) \
             if isinstance(lr,
                tf.keras.optimizers.schedules.LearningRateSchedule) \
             else float(lr)
        if logs:
            logs['learning_rate'] = v


# ══════════════════════════════════════════════════════════════════
# Resume helper
# ══════════════════════════════════════════════════════════════════

def _get_resume_epoch(log_path: str) -> int:
    """
    Read the CSVLogger log and return the last completed epoch + 1.
    Returns 0 if log does not exist or is empty.
    """
    if not os.path.exists(log_path):
        return 0
    try:
        with open(log_path, 'r') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return 0
        last_epoch = int(float(rows[-1]['epoch']))
        return last_epoch + 1
    except Exception as e:
        print(f"  [resume] Could not read log {log_path}: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════
# Loss dict builder  (shared between build and load)
# ══════════════════════════════════════════════════════════════════

def _make_loss_dict(is_m3: bool, is_m3plus: bool):
    """
    Fix 1: centralised loss dict — is_m3plus guard ensures aux heads
    are never added for M3Plus which has no aux outputs.
    """
    loss_dict    = {'seg':  combined_seg_loss,
                    'cls':  grade_cls_loss,
                    'surv': masked_survival_loss}
    loss_weights = {'seg': 1.0, 'cls': 0.3, 'surv': 0.2}

    if is_m3 and not is_m3plus:   # Fix 1
        loss_dict.update({'aux2': combined_seg_loss,
                          'aux3': combined_seg_loss})
        loss_weights.update({'aux2': 0.2, 'aux3': 0.1})

    return loss_dict, loss_weights


# ══════════════════════════════════════════════════════════════════
# Build + compile
# ══════════════════════════════════════════════════════════════════

def _build_and_compile(build_fn, train_ids,
                       is_m3=False, is_m3plus=False):   # Fix 2
    with strategy.scope():
        model = build_fn()
        est   = int(len(train_ids) * PATCHES_PER_CASE) // BATCH_SIZE
        opt   = make_optimizer(est)
        loss_dict, loss_weights = _make_loss_dict(is_m3, is_m3plus)
        seg_metric_objs = [
            tf.keras.metrics.MeanMetricWrapper(fn, name=fn.__name__)
            for fn in SEG_METRICS
        ]
        model.compile(
            optimizer=opt,
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics={'seg': seg_metric_objs},
        )
    return model, est


def _load_and_compile(ckpt_path, build_fn, train_ids,
                      is_m3=False, is_m3plus=False):    # Fix 1 + Fix 2
    custom_objs = get_custom_objects()
    with strategy.scope():
        model = tf.keras.models.load_model(
            ckpt_path, custom_objects=custom_objs,
            compile=False, safe_mode=False)
        est = int(len(train_ids) * PATCHES_PER_CASE) // BATCH_SIZE
        opt = make_optimizer(est)
        loss_dict, loss_weights = _make_loss_dict(is_m3, is_m3plus)
        seg_metric_objs = [
            tf.keras.metrics.MeanMetricWrapper(fn, name=fn.__name__)
            for fn in SEG_METRICS
        ]
        model.compile(
            optimizer=opt,
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics={'seg': seg_metric_objs},
        )
    return model, est


# ══════════════════════════════════════════════════════════════════
# Main per-model pipeline
# ══════════════════════════════════════════════════════════════════

def run_one_model(build_fn, model_key: str,
                  folds: list, holdout_ids: list, surv_df,
                  use_tta: bool = True) -> pd.DataFrame:

    is_m3     = model_key == 'M3_ASPP_AttDS3D'
    is_m3plus = model_key == 'M3Plus'

    print(f"\n{'='*70}")
    print(f"  STARTING:  {model_key}  ({N_FOLDS}-fold CV + holdout eval)")
    print(f"{'='*70}")

    fold_val_scores = []
    best_fold_ckpt  = None
    best_fold_score = -1.0
    monitor_key     = 'val_seg_any_tumour_dice'   # Fix 5: unified
    monitor_mode    = 'max'

    # ── Cross-validation ─────────────────────────────────────────
    for fold_idx, (train_ids, val_ids) in enumerate(folds):

        ckpt_path = os.path.join(OUTPUT_DIR,
                                 f'{model_key}_fold{fold_idx}_best.keras')
        log_path  = os.path.join(OUTPUT_DIR,
                                 f'{model_key}_fold{fold_idx}.log')

        resume_epoch = _get_resume_epoch(log_path)

        if resume_epoch >= EPOCHS:
            print(f"\n  ── Fold {fold_idx+1}/{N_FOLDS} already complete "
                  f"(epoch {resume_epoch}/{EPOCHS}) — skipping.")
            try:
                with open(log_path, 'r') as f:
                    rows = list(csv.DictReader(f))
                scores = [float(r[monitor_key]) for r in rows
                          if monitor_key in r]
                fold_best = max(scores) if scores else -1.0
            except Exception:
                fold_best = -1.0
            fold_val_scores.append(fold_best)
            if fold_best > best_fold_score:
                best_fold_score = fold_best
                best_fold_ckpt  = ckpt_path
            continue

        print(f"\n  ── Fold {fold_idx+1}/{N_FOLDS}  "
              f"(train={len(train_ids)}, val={len(val_ids)}) ──")

        tf.keras.backend.clear_session(); gc.collect(); time.sleep(1)

        if resume_epoch > 0 and os.path.exists(ckpt_path):
            print(f"  Resuming from epoch {resume_epoch} "
                  f"(checkpoint: {ckpt_path})")
            model, est_steps = _load_and_compile(
                ckpt_path, build_fn, train_ids,
                is_m3, is_m3plus)               # Fix 2
        else:
            resume_epoch = 0
            model, est_steps = _build_and_compile(
                build_fn, train_ids,
                is_m3, is_m3plus)               # Fix 2

        print(f"  {model_key}: {model.count_params():,} params  "
              f"est_steps/epoch={est_steps}  "
              f"initial_epoch={resume_epoch}")

        callbacks = [
            ModelCheckpoint(ckpt_path, monitor=monitor_key,
                            mode=monitor_mode,
                            save_best_only=True, verbose=1),
            EarlyStopping(monitor=monitor_key, mode=monitor_mode,
                          patience=15, restore_best_weights=True,
                          verbose=1),
            LRLogger(),
            CSVLogger(log_path, append=(resume_epoch > 0)),
            OverfitMonitor(start_epoch=10),
        ]

        train_ds  = make_dataset(train_ids, surv_df,
                                 shuffle=True, augment=True)
        val_ds    = make_dataset(val_ids, surv_df,
                                 shuffle=False, augment=False,
                                 patches_per_case=2)
        val_steps = (len(val_ids) * 2) // BATCH_SIZE

        # M3 only: downsample seg target for aux heads
        if is_m3:
            def _add_aux(x, y):
                seg_full = y['seg']
                seg_half = tf.nn.avg_pool3d(seg_full, ksize=2,
                                            strides=2, padding='SAME')
                seg_qtr  = tf.nn.avg_pool3d(seg_half, ksize=2,
                                            strides=2, padding='SAME')
                y['aux2'] = seg_half
                y['aux3'] = seg_qtr
                return x, y
            train_ds = train_ds.map(_add_aux,
                                    num_parallel_calls=tf.data.AUTOTUNE)
            val_ds   = val_ds.map(_add_aux,
                                  num_parallel_calls=tf.data.AUTOTUNE)

        history = model.fit(
            train_ds.repeat(),
            initial_epoch=resume_epoch,
            epochs=EPOCHS,
            steps_per_epoch=est_steps,
            validation_data=val_ds.repeat(),
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1,
        )

        plot_training_history(history, model_key, fold=fold_idx)

        # Fix 5: read fold best from unified monitor key
        try:
            with open(log_path, 'r') as f:
                all_rows = list(csv.DictReader(f))
            all_scores = [float(r[monitor_key]) for r in all_rows
                          if monitor_key in r]
            fold_best = max(all_scores) if all_scores else -1.0
        except Exception:
            h = history.history
            fold_best = max(h.get(monitor_key, [-1.0]))

        fold_val_scores.append(fold_best)
        print(f"  Fold {fold_idx+1} best {monitor_key}: {fold_best:.4f}")

        if fold_best > best_fold_score:
            best_fold_score = fold_best
            best_fold_ckpt  = ckpt_path

        del model, train_ds, val_ds
        gc.collect()
        clear_vol_cache()

    cv_mean = float(np.mean(fold_val_scores))
    cv_std  = float(np.std(fold_val_scores))
    print(f"\n  CV Summary  {model_key}:")
    print(f"  Folds {monitor_key} : "
          f"{[round(s, 4) for s in fold_val_scores]}")
    print(f"  Mean : {cv_mean:.4f}  Std : {cv_std:.4f}")

    # ── Load best fold model ──────────────────────────────────────
    print(f"\n  Loading best checkpoint: {best_fold_ckpt}")
    tf.keras.backend.clear_session(); gc.collect()

    with strategy.scope():
        best_model = tf.keras.models.load_model(
            best_fold_ckpt, custom_objects=get_custom_objects(),
            compile=False, safe_mode=False)

    # ── Holdout evaluation ────────────────────────────────────────
    print(f"\n  Evaluating holdout ({len(holdout_ids)} cases) "
          f"use_tta={use_tta} ...")
    holdout_results = evaluate_set(best_model, holdout_ids,
                                   use_tta=use_tta, desc='holdout')

    summary_df = summarise_results(holdout_results)
    print(f"\n  {model_key} holdout summary:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, f'{model_key}_holdout_summary.csv'),
        index=False)

    def _mean(k):
        return float(np.nanmean(
            [r.get(k, float('nan')) for r in holdout_results]))

    dice_wt = _mean('dice_whole_tumour')
    dice_tc = _mean('dice_tumour_core')
    dice_et = _mean('dice_enhancing')
    hd95_wt = _mean('hd95_whole_tumour')

    print(f"\n  WT Dice:{dice_wt:.4f}  TC Dice:{dice_tc:.4f}  "
          f"ET Dice:{dice_et:.4f}  WT HD95:{hd95_wt:.1f}")

    # ── Digital Twin + PINN ───────────────────────────────────────
    demo_id = holdout_ids[0]
    try:
        X, Y = load_volume(demo_id)

        # Fix 4: use evaluate_fixed for sliding window
        from dynatwin.evaluate_fixed import (sliding_window_inference,
                                              morphological_postprocess)
        seg_pred = sliding_window_inference(best_model, X)
        seg_pred = morphological_postprocess(seg_pred)
        sinfo    = survival_info(demo_id, surv_df)

        if is_m3plus and _HAS_DT_V2:
            # M3Plus: use v2 digital twin with model survival prediction
            X_patch     = tf.expand_dims(
                tf.convert_to_tensor(
                    X[:PATCH_SIZE[0], :PATCH_SIZE[1], :PATCH_SIZE[2]],
                    dtype=tf.float32), axis=0)
            model_out   = best_model(X_patch, training=False)
            pred_log_os = float(model_out['surv'][0, 0])
            pred_os_days= float(np.exp(np.clip(pred_log_os, 0, 8)))
            c0   = predictions_to_density_v2(seg_pred)
            twin = PatientDigitalTwin_v2(c0)
            pinn = PINNCalibrator_v2(twin)
            pr   = pinn.calibrate(500, 1e-3, 0.8)
            twin.D_range   = (0.85*pr['D'],   1.15*pr['D'])
            twin.rho_range = (0.85*pr['rho'], 1.15*pr['rho'])
            scenarios = twin.predict_progression(90, 5)
            sp   = SurvivalPredictor_v2(twin, pr, sinfo,
                                        model_pred_os=pred_os_days)
        else:
            # Fix 3: M1/M2/M3 use original digital_twin
            c0   = predictions_to_density(seg_pred)
            twin = PatientDigitalTwin(c0)
            pinn = PINNCalibrator(twin)
            pr   = pinn.calibrate(500, 1e-3, 0.8)
            twin.D_range   = (0.85*pr['D'],   1.15*pr['D'])
            twin.rho_range = (0.85*pr['rho'], 1.15*pr['rho'])
            scenarios = twin.predict_progression(90, 5)
            sp   = SurvivalPredictor(twin, pr, sinfo)

        surv = sp.estimate_survival(scenarios[0])
        plot_survival_curve(
            surv, os.path.join(OUTPUT_DIR, f'{model_key}_survival.png'))
        pinn_D    = pr['D']
        pinn_rho  = pr['rho']
        calib_os  = surv['calibrated_median_os_days']
        pred_surv = surv['median_survival']

    except Exception as e:
        print(f"  [twin] WARNING: {e}")
        pinn_D = pinn_rho = float('nan')
        calib_os = pred_surv = float('nan')

    # ── Ablation CSV ──────────────────────────────────────────────
    row = {
        'Model':          model_key,
        'Params_M':       round(best_model.count_params() / 1e6, 2),
        'CV_Mean_FGDice': round(cv_mean, 4),
        'CV_Std_FGDice':  round(cv_std,  4),
        'WT_Dice':        round(dice_wt, 4),
        'TC_Dice':        round(dice_tc, 4),
        'ET_Dice':        round(dice_et, 4),
        'WT_HD95':        round(hd95_wt, 2),
        'PINN_D':         round(float(pinn_D),   6),
        'PINN_rho':       round(float(pinn_rho), 6),
        'Calib_OS':       calib_os,
        'Pred_Surv':      pred_surv,
    }
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = df[df['Model'] != model_key]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, index=False)
    print(f"\n  Results saved → {CSV_PATH}")

    del best_model; gc.collect()
    return df, holdout_results


print("[train] 3D run_one_model pipeline ready.")
