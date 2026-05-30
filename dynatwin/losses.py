# -*- coding: utf-8 -*-
"""
dynatwin/losses.py  — 3D Edition
===================================
All 6 issues from code review corrected:

  Issue 1  any_tumour_dice shape mismatch with MirroredStrategy
           → _broadcast_flat() + cast before reduce_sum

  Issue 2  Plain Dice replaced with per-class Focal-Tversky
           Necrotic: alpha=0.30, beta=0.80, gamma=0.90  (hard FN penalty)
           Edema:    alpha=0.45, beta=0.55, gamma=0.75
           Enhancing:alpha=0.35, beta=0.65, gamma=0.85

  Issue 3  _ALPHA was 0.40 — corrected to 0.05
           CE is a small regulariser; Focal-Tversky does the heavy lifting

  Issue 4  BG CE weight comment was wrong.
           BG=1.0 is the LOWEST weight — it under-penalises background.
           Raised to 2.0 so background misclassification is penalised
           more strongly and tumour flooding is better suppressed.
           Tumour weights reduced proportionally.

  Issue 5  _fg() was dead code — any_tumour_dice now calls _fg()

  Issue 6  huber.shape.rank is None during graph build (static property)
           → replaced with tf.cond(tf.equal(tf.rank(huber), 1), ...)
"""

import numpy as np
import tensorflow as tf
from dynatwin.config import NUM_CLASSES

# ── Per-class Focal-Tversky parameters ────────────────────────────
# (alpha=FP_weight, beta=FN_weight, gamma=focus_exponent)
# Necrotic: heavy FN penalty (beta=0.80) — missing necrotic core is
#           the dominant failure mode on BraTS small/irregular regions.
# Edema:    mild parameters — large contiguous region, easier to detect.
# Enhancing: moderate FN penalty — clinically critical for grading.
_FT_PARAMS = {
    1: (0.30, 0.80, 0.90),   # Necrotic
    2: (0.45, 0.55, 0.75),   # Edema
    3: (0.35, 0.65, 0.85),   # Enhancing
}

# ── CE weights ─────────────────────────────────────────────────────
# Issue 4 fix: BG raised from 1.0 to 2.0 so background misclassification
# is penalised more strongly relative to tumour classes.
# BG=2 means: on a background voxel, wrongly predicting tumour costs 2×
# the baseline vs tumour-class voxels at 3-5×.  This suppresses flooding
# while preserving the relative tumour-class emphasis.
_CEW  = tf.constant([2.0, 3.0, 1.5, 5.0], dtype=tf.float32)
#                    BG   Nec  Ede  Enh

# Issue 3 fix: CE fraction 0.40 → 0.05
# CE is a small regulariser that keeps logits calibrated and prevents
# the probability distribution from collapsing.  Focal-Tversky carries
# the main learning signal.  At 0.40 CE was dominating and overwhelming
# the class-specific FN penalty designed into Focal-Tversky.
_ALPHA = 0.05   # CE fraction;  0.95 Focal-Tversky


# ══════════════════════════════════════════════════════════════════
# MirroredStrategy shape helper
# ══════════════════════════════════════════════════════════════════

def _broadcast_flat(a, b):
    sa = tf.shape(a)[0]
    sb = tf.shape(b)[0]
    # Only broadcast if one is scalar-like (aggregation artifact)
    a = tf.cond(tf.equal(sa, 1), lambda: tf.tile(a, [sb]), lambda: a)
    b = tf.cond(tf.equal(sb, 1), lambda: tf.tile(b, [sa]), lambda: b)
    return a, b


# ══════════════════════════════════════════════════════════════════
# Foreground collapse helper  (Issue 5: used by all fg-based metrics)
# ══════════════════════════════════════════════════════════════════

def _fg(yt, yp):
    """Binary foreground masks by collapsing tumour classes 1-3."""
    yt = tf.cast(yt, tf.float32)
    yp = tf.cast(yp, tf.float32)
    ft = tf.clip_by_value(tf.reduce_sum(yt[..., 1:], axis=-1), 0, 1)
    fp = tf.clip_by_value(tf.reduce_sum(yp[..., 1:], axis=-1), 0, 1)
    return ft, fp


# ══════════════════════════════════════════════════════════════════
# Segmentation metrics
# ══════════════════════════════════════════════════════════════════

def _dice_cls(yt, yp, cls, smooth=1.0):
    """Soft Dice for one class channel. Flat reshape handles any input rank."""
    t = tf.reshape(tf.cast(yt, tf.float32)[..., cls], [-1])
    p = tf.reshape(tf.cast(yp, tf.float32)[..., cls], [-1])
    t, p = _broadcast_flat(t, p)
    i = tf.reduce_sum(t * p)
    return (2.0 * i + smooth) / (tf.reduce_sum(t) + tf.reduce_sum(p) + smooth)

def dice_coef(yt, yp):
    return tf.reduce_mean([_dice_cls(yt, yp, i) for i in range(1, NUM_CLASSES)])

def dice_coef_necrotic(yt, yp):  return _dice_cls(yt, yp, 1)
def dice_coef_edema(yt, yp):     return _dice_cls(yt, yp, 2)
def dice_coef_enhancing(yt, yp): return _dice_cls(yt, yp, 3)

def any_tumour_dice(yt, yp, smooth=1.0):
    """
    Binary foreground Dice.  Issue 1+5 fix:
      - uses _fg() (no duplication)
      - _broadcast_flat() guards against MirroredStrategy aggregation shape mismatch
    """
    ft, fp = _fg(yt, yp)
    ft = tf.reshape(ft, [-1])
    fp = tf.reshape(fp, [-1])
    ft, fp = _broadcast_flat(ft, fp)
    i = tf.reduce_sum(ft * fp)
    return (2.0 * i + smooth) / (tf.reduce_sum(ft) + tf.reduce_sum(fp) + smooth)

def mean_iou_soft(yt, yp, smooth=1e-6):
    total = 0.0
    for i in range(1, NUM_CLASSES):
        t = tf.reshape(tf.cast(yt, tf.float32)[..., i], [-1])
        p = tf.reshape(tf.cast(yp, tf.float32)[..., i], [-1])
        t, p = _broadcast_flat(t, p)
        inter = tf.reduce_sum(t * p)
        union = tf.reduce_sum(t) + tf.reduce_sum(p) - inter
        total += (inter + smooth) / (union + smooth)
    return total / (NUM_CLASSES - 1)

def sensitivity(yt, yp):
    ft, fp = _fg(yt, yp)
    ft = tf.reshape(ft, [-1]); fp = tf.reshape(fp, [-1])
    ft, fp = _broadcast_flat(ft, fp)
    tp = tf.reduce_sum(tf.round(tf.clip_by_value(ft * fp,       0, 1)))
    fn = tf.reduce_sum(tf.round(tf.clip_by_value(ft * (1 - fp), 0, 1)))
    return tp / (tp + fn + tf.keras.backend.epsilon())

def precision(yt, yp):
    ft, fp = _fg(yt, yp)
    ft = tf.reshape(ft, [-1]); fp = tf.reshape(fp, [-1])
    ft, fp = _broadcast_flat(ft, fp)
    tp  = tf.reduce_sum(tf.round(tf.clip_by_value(ft * fp,        0, 1)))
    fpp = tf.reduce_sum(tf.round(tf.clip_by_value((1 - ft) * fp,  0, 1)))
    return tp / (tp + fpp + tf.keras.backend.epsilon())

def specificity(yt, yp):
    ft, fp = _fg(yt, yp)
    ft = tf.reshape(ft, [-1]); fp = tf.reshape(fp, [-1])
    ft, fp = _broadcast_flat(ft, fp)
    tn  = tf.reduce_sum(tf.round(tf.clip_by_value((1 - ft) * (1 - fp), 0, 1)))
    fpp = tf.reduce_sum(tf.round(tf.clip_by_value((1 - ft) * fp,       0, 1)))
    return tn / (tn + fpp + tf.keras.backend.epsilon())

SEG_METRICS = [
    any_tumour_dice, dice_coef,
    dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing,
    mean_iou_soft, precision, sensitivity, specificity,
]


# ══════════════════════════════════════════════════════════════════
# Segmentation loss  (Issue 2: Focal-Tversky replaces plain Dice)
# ══════════════════════════════════════════════════════════════════

def _focal_tversky_cls(yt, yp, cls, smooth=1.0):
    """
    Issue 2 fix: per-class Focal-Tversky loss.

    Tversky(alpha, beta) = TP / (TP + alpha*FP + beta*FN)
    Focal exponent gamma focuses training on hard/small regions.

    Necrotic (cls=1): beta=0.80 — aggressively penalises missed voxels
                      (false negatives) since necrotic core is small and
                      clinically critical.
    """
    alpha, beta, gamma = _FT_PARAMS[cls]
    t  = tf.reshape(tf.cast(yt[..., cls], tf.float32), [-1])
    p  = tf.reshape(tf.cast(yp[..., cls], tf.float32), [-1])
    t, p = _broadcast_flat(t, p)
    tp = tf.reduce_sum(t * p)
    fp = tf.reduce_sum((1.0 - t) * p)
    fn = tf.reduce_sum(t * (1.0 - p))
    tversky_idx = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - tversky_idx) ** gamma


def _focal_tversky_loss(yt, yp):
    """
    Sum of per-class Focal-Tversky losses with a small BG Dice term.
    Background Dice (weight 0.1) prevents the model from ignoring
    the background class entirely.
    """
    yt = tf.cast(yt, tf.float32)
    yp = tf.cast(yp, tf.float32)
    # Small background Dice term (not Focal-Tversky — BG is easy and large)
    tb = tf.reshape(yt[..., 0], [-1]); pb = tf.reshape(yp[..., 0], [-1])
    tb, pb = _broadcast_flat(tb, pb)
    db   = (2.0 * tf.reduce_sum(tb*pb) + 1.0) / \
           (tf.reduce_sum(tb) + tf.reduce_sum(pb) + 1.0)
    loss = 0.1 * (1.0 - db)
    # Per-class Focal-Tversky for tumour classes
    for i in range(1, NUM_CLASSES):
        loss += _focal_tversky_cls(yt, yp, i)
    return loss


def _weighted_ce_loss(yt, yp):
    """
    Class-weighted CE.
    Issue 4 fix: BG weight raised to 2.0 (was 1.0) — see _CEW comment.
    """
    yt   = tf.cast(yt, tf.float32)
    yp   = tf.cast(yp, tf.float32)
    logp = tf.math.log(tf.clip_by_value(yp, 1e-7, 1.0))
    return tf.reduce_mean(-tf.reduce_sum(_CEW * yt * logp, axis=-1))


def combined_seg_loss(yt, yp):
    """
    Issue 3 fix: 0.05 CE + 0.95 Focal-Tversky  (was 0.40 CE + 0.60 Dice).
    CE is a small calibration term; Focal-Tversky drives learning.
    """
    return (_ALPHA * _weighted_ce_loss(yt, yp)
            + (1.0 - _ALPHA) * _focal_tversky_loss(yt, yp))


# ══════════════════════════════════════════════════════════════════
# Classification loss
# ══════════════════════════════════════════════════════════════════

def grade_cls_loss(yt, yp):
    yt = tf.cast(yt, tf.float32); yp = tf.cast(yp, tf.float32)
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(yt, yp))


# ══════════════════════════════════════════════════════════════════
# Survival regression loss
# ══════════════════════════════════════════════════════════════════

def masked_survival_loss(yt, yp):
    """
    Masked Huber loss on log(OS).
    Issue 6 fix: huber.shape.rank is None during graph tracing (static
    property). Use tf.cond(tf.equal(tf.rank(huber), 1), ...) instead.

    yt : (B, 2) — [:, 0]=log_os, [:, 1]=os_mask (1=valid, 0=missing)
    yp : (B, 1) — predicted log_os
    """
    yt     = tf.cast(yt, tf.float32)
    yp     = tf.cast(yp, tf.float32)
    target = yt[:, 0:1]
    mask   = yt[:, 1:2]
    huber  = tf.keras.losses.Huber(delta=0.5, reduction='none')(target, yp)
    # Issue 6 fix: use tf.rank (dynamic) not .shape.rank (static/None in graph)
    huber  = tf.cond(
        tf.equal(tf.rank(huber), 1),
        lambda: tf.expand_dims(huber, -1),
        lambda: huber,
    )
    return tf.reduce_sum(huber * mask) / (tf.reduce_sum(mask) + 1e-8)


# ══════════════════════════════════════════════════════════════════
# Custom objects dict
# ══════════════════════════════════════════════════════════════════

def get_custom_objects():
    from dynatwin.models import DropPath
    return {
        'combined_seg_loss':    combined_seg_loss,
        'grade_cls_loss':       grade_cls_loss,
        'masked_survival_loss': masked_survival_loss,
        'any_tumour_dice':      any_tumour_dice,
        'dice_coef':            dice_coef,
        'dice_coef_necrotic':   dice_coef_necrotic,
        'dice_coef_edema':      dice_coef_edema,
        'dice_coef_enhancing':  dice_coef_enhancing,
        'mean_iou_soft':        mean_iou_soft,
        'precision':            precision,
        'sensitivity':          sensitivity,
        'specificity':          specificity,
        'DropPath':             DropPath,
    }

print("[losses] Focal-Tversky + all 6 review fixes applied.")
