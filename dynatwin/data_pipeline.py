# -*- coding: utf-8 -*-
"""
dynatwin/data_pipeline.py  — 3D Edition
=========================================
3D patch-based data pipeline for BraTS2020.

Fixes applied vs previous version:
  Fix 1  MIN_TUMOUR_VOXELS=0 — filter block removed entirely.
         Was silently dropping patches and making dataset length
         unpredictable, causing est_steps in train.py to be wrong.
         Background patches must be included so the model learns
         to reject background confidently at inference.

  Fix 2  Shuffle buffer 64 → 256.
         Buffer of 64 with patches_per_case=4 means consecutive
         patches from the same patient end up in the same batch,
         hurting generalisation.  256 spans ~64 cases.

  Fix 3  Volume caching added (_VOL_CACHE dict).
         BraTS volumes are ~60 MB each.  Without caching, every
         epoch re-reads all 220 NIfTI files from disk (13 GB I/O).
         Cache holds all volumes in RAM after first epoch — safe on
         H100 node which has sufficient system memory.
         clear_vol_cache() is called between folds from train.py.

  Fix 4  drop_remainder=True always.
         MirroredStrategy requires equal batch sizes across replicas.
         drop_remainder=augment was passing incomplete batches during
         validation which causes a replica shape mismatch error.

  Fix 5  BraTS20_Training_355 exclusion now uses os.path.basename()
         instead of string equality on the full path.  The old version
         silently failed if scandir returned paths with trailing slashes.

  Fix 6  normalize_volume: all-zero channel detection changed from
         ch > 0 to abs(ch) > 1e-6 to catch channels with only
         near-zero or negative values after resampling artifacts.
"""

import os
import gc
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

from dynatwin.config import (
    TRAIN_DATASET_PATH, SURVIVAL_CSV, PATCH_SIZE, PATCHES_PER_CASE,
    FG_PATCH_PROB, NUM_CLASSES, NUM_MODALITIES, BATCH_SIZE,
    MIN_TUMOUR_VOXELS, NUM_GRADE_CLASSES, HOLDOUT_RATIO, N_FOLDS,
    RANDOM_SEED,
)


# ══════════════════════════════════════════════════════════════════
# Survival CSV helpers
# ══════════════════════════════════════════════════════════════════

def load_survival_df() -> pd.DataFrame:
    """Load BraTS2020 survival CSV; fill missing OS with NaN."""
    if not os.path.exists(SURVIVAL_CSV):
        print(f"[data] WARNING: survival CSV not found at {SURVIVAL_CSV}")
        return pd.DataFrame(columns=['BraTS20ID', 'Age', 'Survival_days',
                                     'Extent_of_Resection'])
    df = pd.read_csv(SURVIVAL_CSV)
    df['Survival_days'] = pd.to_numeric(df['Survival_days'], errors='coerce')
    df['Age']           = pd.to_numeric(df['Age'],           errors='coerce')
    df['case_id'] = df['Brats20ID'].str.strip()
    df = df.set_index('case_id')
    return df


def survival_info(case_id: str, surv_df: pd.DataFrame) -> dict:
    """Return dict with age_norm, log_os, os_mask, eor for one case."""
    eor_map = {'GTR': 0, 'STR': 1}
    if case_id in surv_df.index:
        row  = surv_df.loc[case_id]
        age  = float(row['Age'])           if pd.notna(row['Age'])            else 65.0
        os_d = float(row['Survival_days']) if pd.notna(row['Survival_days'])  else np.nan
        eor  = eor_map.get(str(row.get('Extent_of_Resection', '')), 2)
    else:
        age, os_d, eor = 65.0, np.nan, 2
    os_mask  = 0.0 if np.isnan(os_d) else 1.0
    log_os   = np.log(max(os_d, 1.0)) if os_mask else 0.0
    age_norm = np.clip((age - 40.0) / 40.0, -1.0, 2.0)
    return {'age_norm': age_norm, 'log_os': log_os,
            'os_mask': os_mask, 'eor': eor}


# ══════════════════════════════════════════════════════════════════
# Stratified split
# ══════════════════════════════════════════════════════════════════

def get_stratified_split(surv_df: pd.DataFrame):
    """
    Returns (holdout_ids, folds).
    10% stratified holdout for external validation.
    3-fold stratified CV on remaining 90%.
    Stratification key: OS tertile (short/medium/long/missing).
    """
    all_dirs = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
    # Fix 5: use os.path.basename — robust to trailing slashes
    all_dirs = [d for d in all_dirs
                if os.path.basename(d) != 'BraTS20_Training_355']
    all_ids  = [os.path.basename(d) for d in all_dirs]

    strata = []
    for cid in all_ids:
        if cid in surv_df.index:
            os_d = surv_df.loc[cid, 'Survival_days']
            os_d = float(os_d) if pd.notna(os_d) else np.nan
        else:
            os_d = np.nan
        if np.isnan(os_d):
            strata.append(3)
        elif os_d < 300:
            strata.append(0)
        elif os_d < 450:
            strata.append(1)
        else:
            strata.append(2)
    strata = np.array(strata)

    train_val_ids, holdout_ids, tv_strata, _ = train_test_split(
        all_ids, strata,
        test_size=HOLDOUT_RATIO,
        stratify=strata,
        random_state=RANDOM_SEED,
    )

    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_SEED)
    tv_ids    = np.array(train_val_ids)
    tv_strata = np.array(tv_strata)
    folds     = [(tv_ids[tr].tolist(), tv_ids[val].tolist())
                 for tr, val in skf.split(tv_ids, tv_strata)]

    print(f"[data] Total cases : {len(all_ids)}")
    print(f"[data] Holdout     : {len(holdout_ids)}")
    print(f"[data] Fold sizes  : train≈{len(folds[0][0])}  val≈{len(folds[0][1])}")
    return holdout_ids, folds


# ══════════════════════════════════════════════════════════════════
# Volume loading  (with caching)
# ══════════════════════════════════════════════════════════════════

# Fix 3: in-memory volume cache.
# Volumes are loaded once per fold and reused across epochs.
# Call clear_vol_cache() between folds to free memory.
_VOL_CACHE: dict = {}


def clear_vol_cache():
    """Free all cached volumes — call between folds."""
    global _VOL_CACHE
    _VOL_CACHE.clear()
    gc.collect()
    print("[data] Volume cache cleared.")


def normalize_volume(X: np.ndarray) -> np.ndarray:
    """
    Per-channel percentile normalisation on non-zero voxels (3D).
    Fix 6: use abs(ch) > 1e-6 instead of ch > 0 to handle channels
    with near-zero or negative values from resampling artifacts.
    """
    X = X.astype(np.float32)
    for c in range(X.shape[-1]):
        ch = X[..., c]
        nz = ch[np.abs(ch) > 1e-6]   # Fix 6
        if len(nz) == 0:
            continue
        p1, p99 = np.percentile(nz, 1), np.percentile(nz, 99)
        X[..., c] = np.clip((ch - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    return X


def load_volume(case_id: str):
    """
    Load a BraTS2020 case as full 3D volumes.
    Returns X (D,H,W,4) float32 and Y (D,H,W) uint8.
    Fix 3: results cached in _VOL_CACHE after first load.
    """
    if case_id in _VOL_CACHE:
        return _VOL_CACHE[case_id]

    cp   = os.path.join(TRAIN_DATASET_PATH, case_id)
    mods = ['flair', 't1ce', 't1', 't2']
    vols = [nib.load(os.path.join(cp, f'{case_id}_{m}.nii'))
              .get_fdata(dtype=np.float32) for m in mods]
    seg  = nib.load(os.path.join(cp, f'{case_id}_seg.nii')).get_fdata()

    X = np.stack(vols, axis=-1)
    X = normalize_volume(X)
    Y = seg.astype(np.uint8)
    Y[Y == 4] = 3                    # remap ET 4 → 3

    _VOL_CACHE[case_id] = (X, Y)
    return X, Y


def derive_grade_label(Y: np.ndarray) -> int:
    """GBM proxy: presence of enhancing tumour voxels (label 3) → 1, else 0."""
    return 1 if np.any(Y == 3) else 0


# ══════════════════════════════════════════════════════════════════
# Patch extraction
# ══════════════════════════════════════════════════════════════════

#def random_patch(X: np.ndarray, Y: np.ndarray,
 #                patch_size=PATCH_SIZE, fg_prob=FG_PATCH_PROB):
  #  """Extract one random 96³ patch, tumour-centred with prob fg_prob."""
   # D, H, W    = X.shape[:3]
    #pd_, ph, pw = patch_size

    #if np.random.rand() < fg_prob:
     #   fg = np.argwhere(Y > 0)
      #  if len(fg) > 0:
       #     c = fg[np.random.randint(len(fg))]
        #    z = int(np.clip(c[0] - pd_//2, 0, D - pd_))
         #   y = int(np.clip(c[1] - ph//2,  0, H - ph))
          #  x = int(np.clip(c[2] - pw//2,  0, W - pw))
        #else:
         #   z = np.random.randint(0, max(1, D - pd_))
          #  y = np.random.randint(0, max(1, H - ph))
           # x = np.random.randint(0, max(1, W - pw))
    #else:
     #   z = np.random.randint(0, max(1, D - pd_))
      #  y = np.random.randint(0, max(1, H - ph))
       # x = np.random.randint(0, max(1, W - pw))

    #return (X[z:z+pd_, y:y+ph, x:x+pw].copy(),
     #       Y[z:z+pd_, y:y+ph, x:x+pw].copy())


def random_patch(X: np.ndarray, Y: np.ndarray,
                 patch_size=PATCH_SIZE, fg_prob=FG_PATCH_PROB):
    """
    Extract one random patch, tumour-centred with prob fg_prob.
 
    M3+ change — stratified centroid sampling:
    Original: fg = np.argwhere(Y > 0)
              Centroid sampled uniformly from ALL tumour voxels.
              Since edema dominates volume (~70% of tumour voxels),
              centroid almost always lands in edema. Necrotic and
              enhancing are systematically underrepresented.
 
    Fixed:    Target subregion sampled uniformly from {1, 2, 3} first.
              Then centroid sampled from voxels of that subregion.
              Each subregion gets equal centroid probability regardless
              of its volume. Falls back to any tumour voxel if the
              target subregion is absent in this volume.
    """
    D, H, W    = X.shape[:3]
    pd_, ph, pw = patch_size
 
    if np.random.rand() < fg_prob:
        # Stratified: pick subregion class first, then sample centroid
        target_label = np.random.choice([1, 2, 3])
        fg = np.argwhere(Y == target_label)
        if len(fg) == 0:
            # Fallback: target subregion absent — use any tumour voxel
            fg = np.argwhere(Y > 0)
        if len(fg) > 0:
            c = fg[np.random.randint(len(fg))]
            z = int(np.clip(c[0] - pd_//2, 0, D - pd_))
            y = int(np.clip(c[1] - ph//2,  0, H - ph))
            x = int(np.clip(c[2] - pw//2,  0, W - pw))
        else:
            # No tumour at all — random patch
            z = np.random.randint(0, max(1, D - pd_))
            y = np.random.randint(0, max(1, H - ph))
            x = np.random.randint(0, max(1, W - pw))
    else:
        z = np.random.randint(0, max(1, D - pd_))
        y = np.random.randint(0, max(1, H - ph))
        x = np.random.randint(0, max(1, W - pw))
 
    return (X[z:z+pd_, y:y+ph, x:x+pw].copy(),
            Y[z:z+pd_, y:y+ph, x:x+pw].copy())
# ══════════════════════════════════════════════════════════════════
# 3D Augmentation
# ══════════════════════════════════════════════════════════════════

def augment_3d(X: np.ndarray, Y: np.ndarray):
    """
    Fast 3D augmentation — no scipy elastic deformation.
      • Random axis flips
      • Random 90° rotation in one of three planes
      • Per-channel intensity scale + shift
      • Gaussian noise
      • Channel dropout (simulate missing modality ~15%)
    """
    for axis in range(3):
        if np.random.rand() > 0.5:
            X = np.flip(X, axis=axis).copy()
            Y = np.flip(Y, axis=axis).copy()

    plane = [(0, 1), (0, 2), (1, 2)][np.random.randint(3)]
    k     = np.random.randint(4)
    X = np.rot90(X, k=k, axes=plane).copy()
    Y = np.rot90(Y, k=k, axes=plane).copy()

    for c in range(X.shape[-1]):
        if np.random.rand() > 0.4:
            X[..., c] = np.clip(
                X[..., c] * np.random.uniform(0.85, 1.15)
                + np.random.uniform(-0.07, 0.07), 0.0, 1.0)

    if np.random.rand() > 0.5:
        X = np.clip(X + np.random.normal(0, 0.03, X.shape), 0.0, 1.0)

    if np.random.rand() < 0.15:
        X[..., np.random.randint(NUM_MODALITIES)] = 0.0

    return X.astype(np.float32), Y.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# Generator → tf.data
# ══════════════════════════════════════════════════════════════════

def _make_generator(ids, surv_df, augment=True,
                    patches_per_case=PATCHES_PER_CASE):
    """
    Python generator yielding one patch at a time.
    Fix 1: MIN_TUMOUR_VOXELS filter removed — all patches yielded.
    Fix 3: volumes loaded via load_volume() which uses _VOL_CACHE.
    """
    pd_, ph, pw = PATCH_SIZE

    def _gen():
        id_list = list(ids)
        if augment:
            np.random.shuffle(id_list)
        for cid in id_list:
            try:
                X, Y = load_volume(cid)
            except Exception as e:
                print(f"[data] Skip {cid}: {e}")
                continue
            grade    = derive_grade_label(Y)
            sinfo    = survival_info(cid, surv_df)
            grade_oh = np.array([1 - grade, grade], dtype=np.float32)
            surv_t   = np.array([sinfo['log_os'], sinfo['os_mask']],
                                dtype=np.float32)

            for _ in range(patches_per_case):
                px, py = random_patch(X, Y)
                if augment:
                    px, py = augment_3d(px, py)
                # Fix 1: no MIN_TUMOUR_VOXELS filter — yield all patches
                py_oh = tf.keras.utils.to_categorical(
                    py, num_classes=NUM_CLASSES).astype(np.float32)
                yield px, py_oh, grade_oh, surv_t

    return _gen


def make_dataset(ids, surv_df, shuffle=True, augment=True,
                 batch_size=BATCH_SIZE, patches_per_case=PATCHES_PER_CASE):
    """
    Build a tf.data.Dataset of 3D patches.

    Yields batches of:
      X      (B, 96, 96, 96, 4)  float32 — image patch
      y_seg  (B, 96, 96, 96, 4)  float32 — one-hot seg target
      y_cls  (B, 2)              float32 — grade one-hot
      y_surv (B, 2)              float32 — [log_os, os_mask]
    """
    gen     = _make_generator(ids, surv_df, augment=augment,
                               patches_per_case=patches_per_case)
    pd_, ph, pw = PATCH_SIZE

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((pd_, ph, pw, NUM_MODALITIES), tf.float32),
            tf.TensorSpec((pd_, ph, pw, NUM_CLASSES),    tf.float32),
            tf.TensorSpec((NUM_GRADE_CLASSES,),           tf.float32),
            tf.TensorSpec((2,),                           tf.float32),
        )
    )

    if shuffle:
        # Fix 2: buffer 64 → 256 — spans ~64 cases, breaks patient correlation
        ds = ds.shuffle(buffer_size=256, reshuffle_each_iteration=True)

    # Fix 4: drop_remainder=True always — MirroredStrategy requires
    # equal batch sizes across replicas; incomplete batches cause errors.
    ds = ds.batch(batch_size, drop_remainder=True)

    def _reformat(x, y_seg, y_cls, y_surv):
        return x, {'seg': y_seg, 'cls': y_cls, 'surv': y_surv}

    ds = ds.map(_reformat, num_parallel_calls=tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_optimization.map_fusion          = True
    opts.experimental_optimization.map_parallelization = True
    ds = ds.with_options(opts)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


print("[data_pipeline] 3D patch pipeline ready.")
