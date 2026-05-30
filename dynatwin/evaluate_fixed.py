# -*- coding: utf-8 -*-
"""
dynatwin/evaluate_fixed.py  — 3D Edition (corrected)
================================================
Fixes applied vs previous version:

  Fix 1  Gaussian sigma_scale 0.125 → 0.25
         With sigma=12 voxels, patches at the boundary got near-zero
         weight causing necrotic core (which often sits at patch edges)
         to be suppressed in the accumulated probability map.
         sigma=24 voxels gives much better coverage across the patch.

  Fix 2  morphological_postprocess: per-class min_voxels
         Global min_voxels=50 was silently deleting real necrotic
         predictions (necrotic can be 20-30 voxels on some slices).
         Now: Necrotic=10, Enhancing=20, Edema=50.

  Fix 3  TTA: pad volume to cube before flipping, crop after.
         BraTS volumes are 240x240x155 — not cubic. Flipping axis-2
         (depth=155) and unflipping with [::−1] misaligns by 1 voxel
         on odd dimensions. Padding to cube before TTA ensures perfect
         flip symmetry, then crop back to original shape after.

  Fix 4  _range: use consistent stride — removed the appended
         'total - psize' which created a heavily overlapping region
         at the end of the depth axis (155) causing over-weighting.
         Now ensures last patch is always included cleanly.
"""

import os
import gc
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

from dynatwin.config import (
    TRAIN_DATASET_PATH, PATCH_SIZE, PATCH_STRIDE, NUM_CLASSES,
)
from dynatwin.data_pipeline import load_volume


# ══════════════════════════════════════════════════════════════════
# Gaussian window
# ══════════════════════════════════════════════════════════════════

def _gaussian_kernel(size, sigma_scale=0.25):   # Fix 1: was 0.125
    """3D Gaussian weight kernel; peaks at 1.0 in the centre."""
    kernel = np.ones(size, dtype=np.float32)
    for axis, s in enumerate(size):
        coords = np.arange(s) - s / 2.0
        sigma  = s * sigma_scale
        gauss  = np.exp(-0.5 * (coords / sigma) ** 2)
        shape  = [1, 1, 1]; shape[axis] = s
        kernel *= gauss.reshape(shape)
    return kernel / kernel.max()


_GAUSS = _gaussian_kernel(PATCH_SIZE)


# ══════════════════════════════════════════════════════════════════
# Sliding-window inference
# ══════════════════════════════════════════════════════════════════

def _range(total, psize, stride):
    """
    Fix 4: generate patch start positions cleanly.
    Always includes a patch covering the end of the axis.
    """
    positions = list(range(0, total - psize + 1, stride))
    if positions[-1] + psize < total:
        positions.append(total - psize)
    return sorted(set(positions))


def sliding_window_inference(model, volume: np.ndarray,
                             patch_size=PATCH_SIZE,
                             stride=PATCH_STRIDE,
                             batch_size=4) -> np.ndarray:
    """
    Full-volume segmentation via overlapping 96³ patches with
    Gaussian-weighted accumulation.
    """
    D, H, W, _ = volume.shape
    pd, ph, pw  = patch_size
    gauss       = _GAUSS[..., np.newaxis]

    accum   = np.zeros((D, H, W, NUM_CLASSES), dtype=np.float64)
    weights = np.zeros((D, H, W, 1),           dtype=np.float64)

    origins = [(z, y, x)
               for z in _range(D, pd, stride)
               for y in _range(H, ph, stride)
               for x in _range(W, pw, stride)]

    patches_batch = []; origin_batch = []
    for (z, y, x) in origins:
        patches_batch.append(volume[z:z+pd, y:y+ph, x:x+pw])
        origin_batch.append((z, y, x))
        if len(patches_batch) == batch_size:
            _run_batch(model, patches_batch, origin_batch,
                       patch_size, gauss, accum, weights)
            patches_batch, origin_batch = [], []
    if patches_batch:
        _run_batch(model, patches_batch, origin_batch,
                   patch_size, gauss, accum, weights)

    prob_vol = (accum / np.clip(weights, 1e-8, None)).astype(np.float32)
    return np.argmax(prob_vol, axis=-1).astype(np.uint8)


def _run_batch(model, patches, origins, patch_size, gauss, accum, weights):
    batch_np = np.stack(patches, axis=0)
    preds    = model.predict(batch_np, verbose=0)
    if isinstance(preds, (list, tuple)):
        seg_pred = preds[0]
    elif isinstance(preds, dict):
        seg_pred = preds['seg']
    else:
        seg_pred = preds
    seg_pred = seg_pred.astype(np.float64)

    pd_, ph_, pw_ = patch_size
    for i, (z, y, x) in enumerate(origins):
        p = seg_pred[i] * gauss
        accum[z:z+pd_, y:y+ph_, x:x+pw_]   += p
        weights[z:z+pd_, y:y+ph_, x:x+pw_] += gauss


# ══════════════════════════════════════════════════════════════════
# Test-Time Augmentation
# ══════════════════════════════════════════════════════════════════

def tta_inference(model, volume: np.ndarray,
                  patch_size=PATCH_SIZE,
                  stride=PATCH_STRIDE) -> np.ndarray:
    """
    8-flip TTA with Fix 3: pad to cube before flipping.
    BraTS volumes are 240x240x155 — padding to cube ensures
    flip/unflip is perfectly symmetric on all axes.
    """
    D, H, W, C = volume.shape

    # Fix 3: pad to cube
    max_dim = max(D, H, W)
    pad = [(0, max_dim - D), (0, max_dim - H), (0, max_dim - W), (0, 0)]
    vol_padded = np.pad(volume, pad, mode='constant', constant_values=0)
    Dp, Hp, Wp, _ = vol_padded.shape

    gauss = _gaussian_kernel(patch_size)[..., np.newaxis]

    def _infer_one(vol):
        acc = np.zeros((Dp, Hp, Wp, NUM_CLASSES), dtype=np.float64)
        wt  = np.zeros((Dp, Hp, Wp, 1),           dtype=np.float64)
        pd, ph, pw = patch_size
        origins = [(z, y, x)
                   for z in _range(Dp, pd, stride)
                   for y in _range(Hp, ph, stride)
                   for x in _range(Wp, pw, stride)]
        pb = []; ob = []
        for z, y, x in origins:
            pb.append(vol[z:z+pd, y:y+ph, x:x+pw])
            ob.append((z, y, x))
            if len(pb) == 4:
                _run_batch(model, pb, ob, patch_size, gauss, acc, wt)
                pb, ob = [], []
        if pb:
            _run_batch(model, pb, ob, patch_size, gauss, acc, wt)
        return (acc / np.clip(wt, 1e-8, None)).astype(np.float32)

    prob_sum = np.zeros((Dp, Hp, Wp, NUM_CLASSES), dtype=np.float64)
    for ax0 in [1, -1]:
        for ax1 in [1, -1]:
            for ax2 in [1, -1]:
                v = vol_padded[::ax0, ::ax1, ::ax2, :]
                p = _infer_one(v)[::ax0, ::ax1, ::ax2, :]
                prob_sum += p

    avg = (prob_sum / 8.0).astype(np.float32)
    # Fix 3: crop back to original shape
    avg = avg[:D, :H, :W, :]
    return np.argmax(avg, axis=-1).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
# Morphological post-processing
# ══════════════════════════════════════════════════════════════════

# Fix 2: per-class minimum voxel thresholds
# Necrotic=10  — can be very small, don't over-filter
# Enhancing=20 — small but must be connected
# Edema=50     — large region, small components are FP
_MIN_VOXELS = {1: 10, 2: 50, 3: 20}


def morphological_postprocess(seg: np.ndarray) -> np.ndarray:
    """
    Per-class connected-component filtering + hole filling.
    Fix 2: per-class min_voxels so small necrotic is not deleted.
    """
    out = np.zeros_like(seg)
    for cls in range(1, NUM_CLASSES):
        threshold = _MIN_VOXELS.get(cls, 50)
        labeled, n = ndimage.label(seg == cls)
        for cid in range(1, n + 1):
            comp = labeled == cid
            if comp.sum() >= threshold:
                out[ndimage.binary_fill_holes(comp)] = cls
    return out


# ══════════════════════════════════════════════════════════════════
# Hausdorff Distance 95
# ══════════════════════════════════════════════════════════════════

def true_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pp = np.argwhere(pred_mask > 0).astype(float)
    gp = np.argwhere(gt_mask   > 0).astype(float)
    if len(pp) == 0 or len(gp) == 0:
        return float('nan')
    d1, _ = cKDTree(gp).query(pp, workers=-1)
    d2, _ = cKDTree(pp).query(gp, workers=-1)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


# ══════════════════════════════════════════════════════════════════
# Per-case evaluation
# ══════════════════════════════════════════════════════════════════

def evaluate_case(model, case_id: str, use_tta: bool = True) -> dict:
    X, Y = load_volume(case_id)

    if use_tta:
        seg_pred = tta_inference(model, X)
    else:
        seg_pred = sliding_window_inference(model, X)

    seg_pred = morphological_postprocess(seg_pred)

    result = {'case_id': case_id}

    for name, cls in [('necrotic', 1), ('edema', 2), ('enhancing', 3)]:
        pred_b = (seg_pred == cls).astype(np.uint8)
        gt_b   = (Y == cls).astype(np.uint8)
        inter  = np.sum(pred_b * gt_b)
        denom  = np.sum(pred_b) + np.sum(gt_b)
        result[f'dice_{name}'] = (2.0*inter/denom) if denom > 0 else float('nan')
        result[f'hd95_{name}'] = true_hd95(pred_b, gt_b)

    # Whole tumour
    pred_wt = (seg_pred > 0).astype(np.uint8)
    gt_wt   = (Y > 0).astype(np.uint8)
    inter   = np.sum(pred_wt * gt_wt)
    denom   = np.sum(pred_wt) + np.sum(gt_wt)
    result['dice_whole_tumour'] = (2.0*inter/denom) if denom > 0 else float('nan')
    result['hd95_whole_tumour'] = true_hd95(pred_wt, gt_wt)

    # Tumour core
    pred_tc = np.isin(seg_pred, [1, 3]).astype(np.uint8)
    gt_tc   = np.isin(Y, [1, 3]).astype(np.uint8)
    inter   = np.sum(pred_tc * gt_tc)
    denom   = np.sum(pred_tc) + np.sum(gt_tc)
    result['dice_tumour_core'] = (2.0*inter/denom) if denom > 0 else float('nan')
    result['hd95_tumour_core'] = true_hd95(pred_tc, gt_tc)

    return result


def evaluate_set(model, case_ids: list, use_tta: bool = True,
                 desc: str = '') -> list:
    results = []
    n = len(case_ids)
    for i, cid in enumerate(case_ids):
        print(f"  [{desc}] {i+1}/{n}  {cid}", end='  ')
        try:
            r = evaluate_case(model, cid, use_tta=use_tta)
            print(f"WT={r['dice_whole_tumour']:.3f}  "
                  f"TC={r['dice_tumour_core']:.3f}  "
                  f"ET={r['dice_enhancing']:.3f}  "
                  f"Nec={r['dice_necrotic']:.3f}")
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
        gc.collect()
    return results


print("[evaluate] 3D sliding-window + TTA + morphological pipeline ready.")
