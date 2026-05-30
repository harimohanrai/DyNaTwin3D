# -*- coding: utf-8 -*-
"""
dynatwin/evaluate.py  — 3D Edition
=====================================
Full 3D evaluation pipeline:

  sliding_window_inference  full-volume prediction via overlapping patches
  tta_inference             8-flip test-time augmentation ensemble
  morphological_postprocess connected-component filter + hole fill (NOW CALLED)
  true_hd95                 symmetric 95th-percentile Hausdorff distance
  evaluate_case             per-case Dice + HD95 + prediction volume
  evaluate_set              evaluate all cases in a list → per-case DataFrame
"""

import os
import gc
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from scipy import ndimage

from dynatwin.config import (
    TRAIN_DATASET_PATH, PATCH_SIZE, PATCH_STRIDE, NUM_CLASSES, BATCH_SIZE,
)
from dynatwin.data_pipeline import load_volume, normalize_volume


# ══════════════════════════════════════════════════════════════════
# Gaussian window  (for smooth patch stitching)
# ══════════════════════════════════════════════════════════════════

def _gaussian_kernel(size, sigma_scale=0.125):
    """3D Gaussian weight kernel; peaks at 1.0 in the centre."""
    kernel = np.ones(size, dtype=np.float32)
    for axis, s in enumerate(size):
        coords = np.arange(s) - s / 2.0
        sigma  = s * sigma_scale
        gauss  = np.exp(-0.5 * (coords / sigma) ** 2)
        shape  = [1, 1, 1]; shape[axis] = s
        kernel *= gauss.reshape(shape)
    return kernel / kernel.max()


_GAUSS = _gaussian_kernel(PATCH_SIZE)   # pre-computed once


# ══════════════════════════════════════════════════════════════════
# Sliding-window inference
# ══════════════════════════════════════════════════════════════════

def sliding_window_inference(model, volume: np.ndarray,
                             patch_size=PATCH_SIZE,
                             stride=PATCH_STRIDE,
                             batch_size=4) -> np.ndarray:
    """
    Full-volume segmentation via overlapping 96³ patches.

    Patches are weighted by a Gaussian kernel so boundary artefacts
    cancel out when adjacent patches overlap.  Output is argmax over
    the accumulated softmax scores.

    Parameters
    ----------
    volume     : (D, H, W, 4) normalised float32
    patch_size : tuple, default PATCH_SIZE
    stride     : int, patch step (≤ patch_size for overlap)
    batch_size : patches to forward-pass simultaneously

    Returns
    -------
    np.ndarray uint8, shape (D, H, W) — class-index volume
    """
    D, H, W, _ = volume.shape
    pd, ph, pw  = patch_size
    gauss       = _GAUSS[..., np.newaxis]   # (pd,ph,pw,1) for broadcasting

    accum   = np.zeros((D, H, W, NUM_CLASSES), dtype=np.float64)
    weights = np.zeros((D, H, W, 1),           dtype=np.float64)

    # Build list of (z1,y1,x1) patch origins
    def _range(total, psize):
        positions = list(range(0, total - psize, stride))
        positions.append(total - psize)   # always include the last valid position
        return sorted(set(positions))

    origins = [(z, y, x)
               for z in _range(D, pd)
               for y in _range(H, ph)
               for x in _range(W, pw)]

    # Batch the forward passes
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
    batch_np = np.stack(patches, axis=0)          # (B, pd, ph, pw, 4)
    preds    = model.predict(batch_np, verbose=0)
    # Triple-head model: preds is a list/dict; take 'seg' output
    if isinstance(preds, (list, tuple)):
        seg_pred = preds[0]
    elif isinstance(preds, dict):
        seg_pred = preds['seg']
    else:
        seg_pred = preds
    seg_pred = seg_pred.astype(np.float64)        # (B, pd, ph, pw, 4)

    pd_, ph_, pw_ = patch_size
    for i, (z, y, x) in enumerate(origins):
        p = seg_pred[i] * gauss
        accum[z:z+pd_, y:y+ph_, x:x+pw_]   += p
        weights[z:z+pd_, y:y+ph_, x:x+pw_] += gauss


# ══════════════════════════════════════════════════════════════════
# Test-Time Augmentation  (8-flip ensemble)
# ══════════════════════════════════════════════════════════════════

def tta_inference(model, volume: np.ndarray,
                  patch_size=PATCH_SIZE,
                  stride=PATCH_STRIDE) -> np.ndarray:
    """
    8-flip TTA: original + all combinations of axis-0/1/2 flips.
    Each augmented volume is inferred separately; softmax maps are
    averaged BEFORE argmax to preserve calibration.
    Typically adds +1-2 Dice points at zero training cost.
    """
    D, H, W, _ = volume.shape
    pd, ph, pw  = patch_size
    gauss       = _GAUSS[..., np.newaxis]

    def _infer_one(vol):
        """Return softmax probability volume (D, H, W, 4) for one flip."""
        acc = np.zeros((D, H, W, NUM_CLASSES), dtype=np.float64)
        wt  = np.zeros((D, H, W, 1),           dtype=np.float64)

        def _range(total, ps):
            pos = list(range(0, total - ps, stride))
            pos.append(total - ps)
            return sorted(set(pos))

        origins = [(z, y, x) for z in _range(D, pd)
                   for y in _range(H, ph) for x in _range(W, pw)]
        patches = [vol[z:z+pd, y:y+ph, x:x+pw] for z, y, x in origins]
        batches = [patches[i:i+4] for i in range(0, len(patches), 4)]
        origin_batches = [origins[i:i+4] for i in range(0, len(origins), 4)]

        for pb, ob in zip(batches, origin_batches):
            _run_batch(model, pb, ob, patch_size, gauss, acc, wt)
        return (acc / np.clip(wt, 1e-8, None)).astype(np.float32)

    # 8 flips (2³)
    prob_sum = np.zeros((D, H, W, NUM_CLASSES), dtype=np.float64)
    for ax0 in [1, -1]:
        for ax1 in [1, -1]:
            for ax2 in [1, -1]:
                v = volume[::ax0, ::ax1, ::ax2, :]
                p = _infer_one(v)[::ax0, ::ax1, ::ax2, :]
                prob_sum += p

    avg = (prob_sum / 8.0).astype(np.float32)
    return np.argmax(avg, axis=-1).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
# Morphological post-processing  (nnU-Net style) — ALWAYS CALLED
# ══════════════════════════════════════════════════════════════════

def morphological_postprocess(seg: np.ndarray,
                              min_voxels: int = 50) -> np.ndarray:
    """
    Per-class connected-component filtering + hole filling.

    Small components (< min_voxels) are removed as likely FP.
    Holes inside retained components are filled with binary_fill_holes.

    This function is called on EVERY prediction — it was dead code
    in the previous version.  Removing disconnected FP blobs typically
    improves precision by 3-6 points without any training change.
    """
    out = np.zeros_like(seg)
    for cls in range(1, NUM_CLASSES):
        labeled, n = ndimage.label(seg == cls)
        for cid in range(1, n + 1):
            comp = labeled == cid
            if comp.sum() >= min_voxels:
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
    """
    Full inference pipeline for one case:
      1. Load volume
      2. Sliding-window inference (with TTA if requested)
      3. Morphological post-processing (ALWAYS)
      4. Compute Dice and HD95 per sub-region

    Returns dict with keys: case_id, dice_*, hd95_*
    """
    cp   = os.path.join(TRAIN_DATASET_PATH, case_id)
    X, Y = load_volume(case_id)

    if use_tta:
        seg_pred = tta_inference(model, X)
    else:
        seg_pred = sliding_window_inference(model, X)

    # ALWAYS apply morphological post-processing
    seg_pred = morphological_postprocess(seg_pred)

    # Remap GT: label 4 → 3 already done in load_volume
    result = {'case_id': case_id}

    label_map = {
        'necrotic':   (1, 1),    # pred_class, gt_class (after remap)
        'edema':      (2, 2),
        'enhancing':  (3, 3),
    }
    for name, (pc, gc) in label_map.items():
        pred_b = (seg_pred == pc).astype(np.uint8)
        gt_b   = (Y == gc).astype(np.uint8)
        inter  = np.sum(pred_b * gt_b)
        denom  = np.sum(pred_b) + np.sum(gt_b)
        result[f'dice_{name}'] = (2.0 * inter / denom) if denom > 0 else float('nan')
        result[f'hd95_{name}'] = true_hd95(pred_b, gt_b)

    # Whole-tumour (classes 1-3 merged)
    pred_wt = (seg_pred > 0).astype(np.uint8)
    gt_wt   = (Y > 0).astype(np.uint8)
    inter   = np.sum(pred_wt * gt_wt)
    denom   = np.sum(pred_wt) + np.sum(gt_wt)
    result['dice_whole_tumour'] = (2.0 * inter / denom) if denom > 0 else float('nan')
    result['hd95_whole_tumour'] = true_hd95(pred_wt, gt_wt)

    # Tumour core (necrotic + enhancing)
    pred_tc = np.isin(seg_pred, [1, 3]).astype(np.uint8)
    gt_tc   = np.isin(Y, [1, 3]).astype(np.uint8)
    inter   = np.sum(pred_tc * gt_tc)
    denom   = np.sum(pred_tc) + np.sum(gt_tc)
    result['dice_tumour_core'] = (2.0 * inter / denom) if denom > 0 else float('nan')
    result['hd95_tumour_core'] = true_hd95(pred_tc, gt_tc)

    return result


def evaluate_set(model, case_ids: list, use_tta: bool = True,
                 desc: str = '') -> list:
    """Evaluate model on a list of cases; returns list of result dicts."""
    results = []
    n = len(case_ids)
    for i, cid in enumerate(case_ids):
        print(f"  [{desc}] {i+1}/{n}  {cid}", end='  ')
        try:
            r = evaluate_case(model, cid, use_tta=use_tta)
            print(f"WT={r['dice_whole_tumour']:.3f}  "
                  f"TC={r['dice_tumour_core']:.3f}  "
                  f"ET={r['dice_enhancing']:.3f}")
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
        gc.collect()
    return results

print("[evaluate] 3D sliding-window + TTA + morphological pipeline ready.")
