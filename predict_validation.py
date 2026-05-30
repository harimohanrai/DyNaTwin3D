"""
predict_validation.py — BraTS 2020 Validation Inference
========================================================
Runs inference on BraTS 2020 validation set (no labels).
Saves predictions as NIfTI files in BraTS submission format.

For each model:
  - Baseline prediction (no post-processing)
  - cond_v500 post-processed prediction

Output format:
  outputs/validation_predictions/
    M1_ResUNet3D/
      BraTS20_Validation_001.nii.gz
      ...
    M1_ResUNet3D_pp/
      BraTS20_Validation_001.nii.gz
      ...

NIfTI files use original label convention: 0, 1, 2, 4 (ET=4, not 3)
for compatibility with BraTS online evaluation portal.

Usage:
  cd ~/Hari/3D_DynaTwin
  TF_CPP_MIN_LOG_LEVEL=3 python -W ignore predict_validation.py
  # Single model:
  TF_CPP_MIN_LOG_LEVEL=3 python -W ignore predict_validation.py --models M1_ResUNet3D
"""

import os, sys, argparse, gc, time
import numpy as np
import nibabel as nib
import tensorflow as tf

PROJECT_ROOT = '/home/ubuntu/Hari/3D_DynaTwin'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dynatwin.losses import get_custom_objects
from dynatwin.evaluate_fixed import sliding_window_inference, morphological_postprocess
import dynatwin.evaluate_fixed as _ef

# ── Config ────────────────────────────────────────────────────────

VAL_DIR = '/home/ubuntu/Hari/DynaTwin/Brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
_WEIGHTS_DIR = '/home/ubuntu/Hari/3D_DynaTwin/outputs'
PRED_DIR = os.path.join(_WEIGHTS_DIR, 'validation_predictions')
os.makedirs(PRED_DIR, exist_ok=True)

T1CE_CHANNEL = 1
LABEL_NCR = 1; LABEL_ED = 2; LABEL_ET = 3

MODEL_CONFIGS = {
    'M1_ResUNet3D': os.path.join(_WEIGHTS_DIR, 'M1_ResUNet3D_fold0_best.keras'),
    'M2_ResAttUNet3D': os.path.join(_WEIGHTS_DIR, 'M2_ResAttUNet3D_fold1_best.keras'),
    'M3_ASPP_AttDS3D': os.path.join(_WEIGHTS_DIR, 'M3_ASPP_AttDS3D_fold0_best.keras'),
    'M3Plus': os.path.join(_WEIGHTS_DIR, 'M3Plus_fold0_best.keras'),
}


# ── Normalization (matches training pipeline exactly) ─────────────

def normalize_volume(X):
    X = X.astype(np.float32)
    for c in range(X.shape[-1]):
        ch = X[..., c]
        nz = ch[np.abs(ch) > 1e-6]
        if len(nz) == 0:
            continue
        p1, p99 = np.percentile(nz, 1), np.percentile(nz, 99)
        X[..., c] = np.clip((ch - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    return X


# ── Load validation volume ────────────────────────────────────────

def load_val_volume(case_id):
    cp = os.path.join(VAL_DIR, case_id)
    mods = ['flair', 't1ce', 't1', 't2']
    vols = [nib.load(os.path.join(cp, f'{case_id}_{m}.nii'))
            .get_fdata(dtype=np.float32) for m in mods]
    X = np.stack(vols, axis=-1)
    X = normalize_volume(X)

    # Keep reference NIfTI for header/affine
    ref_nii = nib.load(os.path.join(cp, f'{case_id}_flair.nii'))
    return X, ref_nii


# ── cond_v500 post-processing ─────────────────────────────────────

def cond_v500(pred_seg, t1ce):
    if (pred_seg == LABEL_NCR).sum() >= 500:
        return pred_seg
    corrected = pred_seg.copy()
    tc_mask = np.isin(pred_seg, [LABEL_NCR, LABEL_ET])
    if tc_mask.sum() == 0:
        return corrected
    et_mask = (pred_seg == LABEL_ET)
    if et_mask.sum() > 10:
        threshold = np.percentile(t1ce[et_mask], 25)
    else:
        threshold = np.median(t1ce[tc_mask])
    corrected[tc_mask] = np.where(t1ce[tc_mask] < threshold, LABEL_NCR, LABEL_ET)
    return corrected


# ── Remap labels for BraTS submission ─────────────────────────────

def remap_to_brats(seg):
    """Convert internal labels (0,1,2,3) to BraTS convention (0,1,2,4)."""
    out = seg.copy().astype(np.uint8)
    out[seg == 3] = 4  # ET: 3 → 4
    return out


# ── Save prediction as NIfTI ──────────────────────────────────────

def save_nifti(seg, ref_nii, output_path):
    seg_brats = remap_to_brats(seg)
    nii = nib.Nifti1Image(seg_brats, ref_nii.affine, ref_nii.header)
    nib.save(nii, output_path)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='BraTS 2020 validation set inference')
    parser.add_argument('--models', nargs='+',
                        default=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    # Get validation case IDs
    val_cases = sorted([
        d for d in os.listdir(VAL_DIR)
        if os.path.isdir(os.path.join(VAL_DIR, d))
        and d.startswith('BraTS20_Validation_')
    ])
    print(f"\n{'='*70}")
    print(f"  BraTS 2020 VALIDATION SET INFERENCE")
    print(f"  Cases: {len(val_cases)}")
    print(f"  Models: {args.models}")
    print(f"  Output: {PRED_DIR}")
    print(f"{'='*70}")

    custom_objs = get_custom_objects()

    for model_key in args.models:
        ckpt = MODEL_CONFIGS.get(model_key)
        if not ckpt or not os.path.exists(ckpt):
            print(f"\n  [{model_key}] Checkpoint not found — skipping")
            continue

        print(f"\n{'─'*70}")
        print(f"  Model: {model_key}")
        print(f"  Checkpoint: {ckpt}")

        # Create output dirs
        base_dir = os.path.join(PRED_DIR, model_key)
        pp_dir = os.path.join(PRED_DIR, f'{model_key}_pp')
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(pp_dir, exist_ok=True)

        # Load model
        tf.keras.backend.clear_session()
        gc.collect()

        from dynatwin.config import strategy
        with strategy.scope():
            model = tf.keras.models.load_model(
                ckpt, custom_objects=custom_objs,
                compile=False, safe_mode=False)

        _ps = model.input_shape[1:4]
        _ef._GAUSS = _ef._gaussian_kernel(_ps)
        print(f"  Params: {model.count_params():,}, Patch: {_ps}")
        print(f"  Baseline → {base_dir}")
        print(f"  cond_v500 → {pp_dir}")
        print(f"{'─'*70}")

        t_start = time.time()
        for idx, case_id in enumerate(val_cases):
            t_case = time.time()

            # Load
            X, ref_nii = load_val_volume(case_id)
            t1ce = X[..., T1CE_CHANNEL]

            # Inference
            seg_pred = sliding_window_inference(model, X, patch_size=_ps)
            seg_pred = morphological_postprocess(seg_pred)

            # Post-processing
            seg_pp = cond_v500(seg_pred, t1ce)

            # Save
            base_path = os.path.join(base_dir, f'{case_id}.nii.gz')
            pp_path = os.path.join(pp_dir, f'{case_id}.nii.gz')
            save_nifti(seg_pred, ref_nii, base_path)
            save_nifti(seg_pp, ref_nii, pp_path)

            # Stats
            ncr_base = int((seg_pred == LABEL_NCR).sum())
            ncr_pp = int((seg_pp == LABEL_NCR).sum())
            et_base = int((seg_pred == LABEL_ET).sum())
            tc_base = int(np.isin(seg_pred, [LABEL_NCR, LABEL_ET]).sum())
            pp_applied = 'YES' if ncr_base < 500 else 'no'

            elapsed = time.time() - t_case
            print(f"  {idx+1:3d}/{len(val_cases)}  {case_id}  "
                  f"TC={tc_base:>6,}  NCR={ncr_base:>5,}→{ncr_pp:>5,}  "
                  f"ET={et_base:>6,}  PP={pp_applied}  "
                  f"({elapsed:.1f}s)")

        total = time.time() - t_start
        print(f"\n  [{model_key}] Done. {len(val_cases)} cases in {total:.0f}s "
              f"({total/len(val_cases):.1f}s/case)")

        del model
        gc.collect()
        tf.keras.backend.clear_session()

    # Summary
    print(f"\n{'='*70}")
    print(f"  PREDICTIONS SAVED")
    print(f"{'='*70}")
    for model_key in args.models:
        base_dir = os.path.join(PRED_DIR, model_key)
        pp_dir = os.path.join(PRED_DIR, f'{model_key}_pp')
        if os.path.exists(base_dir):
            n = len([f for f in os.listdir(base_dir) if f.endswith('.nii.gz')])
            print(f"  {model_key}: {n} baseline + {n} post-processed")
    print(f"\n  Output: {PRED_DIR}")
    print(f"  Format: NIfTI (.nii.gz), labels 0/1/2/4 (BraTS convention)")
    print(f"  Ready for BraTS online evaluation portal submission.")


if __name__ == '__main__':
    main()
