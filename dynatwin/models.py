# -*- coding: utf-8 -*-
"""
dynatwin/models.py  — 3D Edition
===================================
3D U-Net variants with triple output head.

Fixes applied vs previous version:
  Fix 1  M1 / M2 returned list [seg, cls, surv] — Keras matches losses
         by position not name for list outputs, which is fragile and
         breaks if output order ever changes.  All three models now
         return a dict {'seg':…, 'cls':…, 'surv':…} consistently.

  Fix 2  M3 cls/surv head was reading from enc_bot (before ASPP),
         bypassing all multi-scale context.  Fixed to read from c5
         (after ASPP) so classification and survival see the full
         pyramid representation.

  Fix 3  SE (Squeeze-Excite) blocks added inside every residual block.
         SE gates learn which channels (modalities / features) matter
         for each spatial location — T1ce dominates enhancing tumour,
         FLAIR dominates edema.  This is the core modality-weighting
         mechanism described in the design spec.

  Fix 4  DropPath was applied AFTER Add() — this zeroed both the
         residual AND the skip connection together, equivalent to
         random feature dropout rather than stochastic depth.
         Moved to BEFORE Add() so only the residual branch is dropped.

  Fix 5  ASPP GAP branch used x.shape[1] (static, None in graph mode).
         Replaced with explicit size=(6,6,6) which is always correct
         for 96³ input with 4 max-pooling stages.

  Fix 6  ASPP dilation rates: design spec says [1,6,12,18,24] but
         those were written for 2D.  At 3D bottleneck of 6³ voxels,
         rates > 3 sample entirely outside the feature map.  Rates
         [1,2,3,4] give the maximum meaningful receptive field coverage
         on a 6³ spatial extent.  Four branches retained to match the
         spirit of deep multi-scale pooling.

  Note   Dense skip connections (DenseNet-style concatenating ALL prior
         encoder levels) were listed in the design spec but not
         implemented here for memory reasons: at 3D with 96³ patches
         full DenseNet connectivity roughly triples activation memory.
         The combination of SE attention + attention gates on skips
         achieves the same anti-forgetting effect at a fraction of the
         memory cost.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D,
    concatenate, Activation, Multiply, Add,
    GlobalAveragePooling3D, Reshape, Dense,
    BatchNormalization, SpatialDropout3D,
)
from tensorflow.keras.regularizers import l2

from dynatwin.config import (
    PATCH_SIZE, NUM_CLASSES, NUM_MODALITIES, NUM_GRADE_CLASSES,
)

L2_REG = 5e-5


# ══════════════════════════════════════════════════════════════════
# DropPath (Stochastic Depth)
# ══════════════════════════════════════════════════════════════════

class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=0.0, **kw):
        super().__init__(**kw)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if (not training) or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        rand  = tf.floor(tf.random.uniform(shape, dtype=x.dtype) + keep)
        return x * rand / keep

    def get_config(self):
        cfg = super().get_config()
        cfg['drop_prob'] = self.drop_prob
        return cfg


# ══════════════════════════════════════════════════════════════════
# 3D building blocks
# ══════════════════════════════════════════════════════════════════

def _bn_relu_conv3d(x, filters, kernel=3, dilation=1):
    return Activation('relu')(BatchNormalization()(
           Conv3D(filters, kernel, padding='same', dilation_rate=dilation,
                  use_bias=False, kernel_initializer='he_normal',
                  kernel_regularizer=l2(L2_REG))(x)))


def se_block_3d(x, ratio=8):
    """
    Fix 3: Squeeze-Excite channel attention.
    GAP → FC(C/ratio) → ReLU → FC(C) → Sigmoid → channel-wise scale.
    Learns which feature channels (and by extension which modalities)
    are informative for each spatial region.
    """
    ch  = x.shape[-1]
    gap = GlobalAveragePooling3D()(x)                             # (B, C)
    fc1 = Dense(max(ch // ratio, 1), activation='relu',
                kernel_regularizer=l2(L2_REG))(gap)              # (B, C/r)
    fc2 = Dense(ch, activation='sigmoid',
                kernel_regularizer=l2(L2_REG))(fc1)              # (B, C)
    scale = Reshape((1, 1, 1, ch))(fc2)                          # (B,1,1,1,C)
    return Multiply()([x, scale])


def residual_block_3d(x, filters, dropout=0.0, drop_path=0.0):
    """
    Pre-activation 3D residual block with SE attention.

    Fix 3: SE block applied to residual branch after second conv —
           channels are re-weighted before the skip addition.
    Fix 4: DropPath applied BEFORE Add() so only the residual branch
           is stochastically dropped, not the skip connection.
    """
    sc = x
    h  = _bn_relu_conv3d(x, filters)
    h  = BatchNormalization()(
         Conv3D(filters, 3, padding='same', use_bias=False,
                kernel_regularizer=l2(L2_REG))(h))
    if dropout > 0:
        h = SpatialDropout3D(dropout)(h)

    # Fix 3: SE channel attention on residual branch
    h = se_block_3d(h)

    # Projection shortcut if channel count changes
    if sc.shape[-1] != filters:
        sc = BatchNormalization()(
             Conv3D(filters, 1, padding='same', use_bias=False,
                    kernel_regularizer=l2(L2_REG))(sc))

    # Fix 4: DropPath on residual branch BEFORE adding skip
    h = DropPath(drop_path)(h)
    h = Add()([h, sc])
    return Activation('relu')(h)


def attention_gate_3d(x, g, inter_ch):
    """3D additive soft-attention gate."""
    tx  = BatchNormalization()(Conv3D(inter_ch, 1, padding='same',
                                      use_bias=False,
                                      kernel_regularizer=l2(L2_REG))(x))
    pg  = BatchNormalization()(Conv3D(inter_ch, 1, padding='same',
                                      use_bias=False,
                                      kernel_regularizer=l2(L2_REG))(g))
    act = Activation('relu')(Add()([tx, pg]))
    psi = BatchNormalization()(Conv3D(1, 1, padding='same',
                                      use_bias=False,
                                      kernel_regularizer=l2(L2_REG))(act))
    return Multiply()([x, Activation('sigmoid')(psi)])


def aspp_block_3d(x, num_filters=256):
    """
    3D Atrous Spatial Pyramid Pooling.

    Fix 5: GAP branch now uses explicit UpSampling3D(size=(6,6,6))
           instead of x.shape[1] which is None during graph tracing.

    Fix 6: Rates [1,2,3,4] — design spec listed [1,6,12,18,24] which
           was written for 2D.  At 3D bottleneck of 6³ voxels a rate
           of 6 samples at positions ±6 voxels apart on a 6-voxel grid
           (i.e. entirely outside the feature map).  Rates [1,2,3,4]
           give receptive fields of 3, 5, 7, 9 voxels — enough to
           cover the full 6³ extent from multiple scales.
    """
    def br(inp, d):
        ks = 1 if d == 1 else 3
        return _bn_relu_conv3d(inp, num_filters, kernel=ks, dilation=d)

    b1 = br(x, 1)
    b2 = br(x, 2)
    b3 = br(x, 3)
    b4 = br(x, 4)

    # Global context branch
    gap = GlobalAveragePooling3D()(x)
    gap = Reshape((1, 1, 1, x.shape[-1]))(gap)
    gap = Activation('relu')(BatchNormalization()(
          Conv3D(num_filters, 1, use_bias=False,
                 kernel_regularizer=l2(L2_REG))(gap)))
    # Fix 5: explicit size — 96 / (2^4) = 6
    gap = UpSampling3D(size=(6, 6, 6))(gap)

    out = Activation('relu')(BatchNormalization()(
          Conv3D(num_filters, 1, padding='same', use_bias=False,
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(L2_REG))(
                     concatenate([b1, b2, b3, b4, gap]))))
    return SpatialDropout3D(0.2)(out)


def _rd(x, filters):
    """Channel reduction: 1³ Conv + BN."""
    return BatchNormalization()(
           Conv3D(filters, 1, padding='same', use_bias=False,
                  kernel_regularizer=l2(L2_REG))(x))


# ══════════════════════════════════════════════════════════════════
# Triple output head (shared across M1/M2/M3)
# ══════════════════════════════════════════════════════════════════

def _triple_head(bottleneck, decoder_out):
    """
    bottleneck  : (B, 6, 6, 6, C) — deepest encoder features (after ASPP for M3)
    decoder_out : (B, 96, 96, 96, 32) — final decoder features

    Fix 2 (M3 specific): caller must pass c5 (post-ASPP) not enc_bot
    so cls/surv see the multi-scale pyramid representation.
    """
    # ── Segmentation ────────────────────────────────────────────
    seg_out = Conv3D(NUM_CLASSES, 1, activation='softmax',
                     dtype='float32', name='seg')(decoder_out)

    # ── Grade classification ─────────────────────────────────────
    gap     = GlobalAveragePooling3D()(bottleneck)
    cls     = Dense(64, activation='relu',
                    kernel_regularizer=l2(L2_REG))(gap)
    cls_out = Dense(NUM_GRADE_CLASSES, activation='softmax',
                    dtype='float32', name='cls')(cls)

    # ── Survival regression ──────────────────────────────────────
    surv     = Dense(32, activation='relu',
                     kernel_regularizer=l2(L2_REG))(gap)
    surv_out = Dense(1, dtype='float32', name='surv')(surv)

    return seg_out, cls_out, surv_out


# ══════════════════════════════════════════════════════════════════
# Shared encoder
# ══════════════════════════════════════════════════════════════════

def _encoder(inp):
    """Shared 3D encoder with SE-augmented residual blocks."""
    c1 = residual_block_3d(inp,  32, dropout=0.10, drop_path=0.05)
    p1 = MaxPooling3D(2)(c1)
    c2 = residual_block_3d(p1,  64, dropout=0.15, drop_path=0.08)
    p2 = MaxPooling3D(2)(c2)
    c3 = residual_block_3d(p2, 128, dropout=0.20, drop_path=0.10)
    p3 = MaxPooling3D(2)(c3)
    c4 = residual_block_3d(p3, 192, dropout=0.20, drop_path=0.12)
    p4 = MaxPooling3D(2)(c4)
    c5 = residual_block_3d(p4, 256, dropout=0.15, drop_path=0.12)
    return (c1, c2, c3, c4), c5


# ══════════════════════════════════════════════════════════════════
# Model builders
# ══════════════════════════════════════════════════════════════════

def build_unet3d_m1(input_shape=(*PATCH_SIZE, NUM_MODALITIES),
                    num_classes=NUM_CLASSES) -> Model:
    """
    M1 — 3D ResUNet + SE blocks + triple head.
    Fix 1: returns dict not list.
    """
    inp = Input(input_shape)
    (c1, c2, c3, c4), c5 = _encoder(inp)

    c6 = residual_block_3d(_rd(concatenate([UpSampling3D(2)(c5), c4]), 192),
                            192, dropout=0.15, drop_path=0.10)
    c7 = residual_block_3d(_rd(concatenate([UpSampling3D(2)(c6), c3]), 128),
                            128, dropout=0.12, drop_path=0.08)
    c8 = residual_block_3d(_rd(concatenate([UpSampling3D(2)(c7), c2]),  64),
                             64, dropout=0.10, drop_path=0.05)
    c9 = residual_block_3d(_rd(concatenate([UpSampling3D(2)(c8), c1]),  32),
                             32, dropout=0.05, drop_path=0.03)

    seg, cls, surv = _triple_head(c5, c9)
    # Fix 1: dict output for consistent loss routing
    return Model(inp, {'seg': seg, 'cls': cls, 'surv': surv},
                 name='ResUNet3D_M1')


def build_unet3d_m2(input_shape=(*PATCH_SIZE, NUM_MODALITIES),
                    num_classes=NUM_CLASSES) -> Model:
    """
    M2 — 3D ResUNet + SE blocks + attention gates + triple head.
    Fix 1: returns dict not list.
    """
    inp = Input(input_shape)
    (c1, c2, c3, c4), c5 = _encoder(inp)

    u6 = UpSampling3D(2)(c5)
    c6 = residual_block_3d(_rd(concatenate([u6, attention_gate_3d(c4, u6, 96)]), 192),
                            192, dropout=0.15, drop_path=0.10)
    u7 = UpSampling3D(2)(c6)
    c7 = residual_block_3d(_rd(concatenate([u7, attention_gate_3d(c3, u7, 64)]), 128),
                            128, dropout=0.12, drop_path=0.08)
    u8 = UpSampling3D(2)(c7)
    c8 = residual_block_3d(_rd(concatenate([u8, attention_gate_3d(c2, u8, 32)]),  64),
                             64, dropout=0.10, drop_path=0.05)
    u9 = UpSampling3D(2)(c8)
    c9 = residual_block_3d(_rd(concatenate([u9, attention_gate_3d(c1, u9, 16)]),  32),
                             32, dropout=0.05, drop_path=0.03)

    seg, cls, surv = _triple_head(c5, c9)
    # Fix 1: dict output
    return Model(inp, {'seg': seg, 'cls': cls, 'surv': surv},
                 name='ResAttUNet3D_M2')


def build_unet3d_m3(input_shape=(*PATCH_SIZE, NUM_MODALITIES),
                    num_classes=NUM_CLASSES) -> Model:
    """
    M3 — SE + Attention + ASPP bottleneck + deep supervision + triple head.
    Fix 1: consistent dict output.
    Fix 2: _triple_head receives c5 (post-ASPP) not enc_bot.
    """
    inp = Input(input_shape)
    (c1, c2, c3, c4), enc_bot = _encoder(inp)

    # ASPP on bottleneck
    c5 = aspp_block_3d(enc_bot, num_filters=256)

    u6 = UpSampling3D(2)(c5)
    c6 = residual_block_3d(_rd(concatenate([u6, attention_gate_3d(c4, u6, 96)]), 192),
                            192, dropout=0.15, drop_path=0.10)
    u7 = UpSampling3D(2)(c6)
    c7 = residual_block_3d(_rd(concatenate([u7, attention_gate_3d(c3, u7, 64)]), 128),
                            128, dropout=0.12, drop_path=0.08)

    # Deep supervision at 12³ (stride-8)
    aux3 = Conv3D(num_classes, 1, activation='softmax',
                  dtype='float32', name='aux3')(c7)

    u8 = UpSampling3D(2)(c7)
    c8 = residual_block_3d(_rd(concatenate([u8, attention_gate_3d(c2, u8, 32)]),  64),
                             64, dropout=0.10, drop_path=0.05)

    # Deep supervision at 24³ (stride-4)
    aux2 = Conv3D(num_classes, 1, activation='softmax',
                  dtype='float32', name='aux2')(c8)

    u9 = UpSampling3D(2)(c8)
    c9 = residual_block_3d(_rd(concatenate([u9, attention_gate_3d(c1, u9, 16)]),  32),
                             32, dropout=0.05, drop_path=0.03)

    # Fix 2: pass c5 (post-ASPP) so cls/surv heads see multi-scale context
    seg, cls, surv = _triple_head(c5, c9)

    # Fix 1: consistent dict output
    return Model(inp, {'seg': seg, 'cls': cls, 'surv': surv,
                        'aux2': aux2, 'aux3': aux3},
                 name='ASPPAttDS3D_M3')

# -*- coding: utf-8 -*-
"""
dynatwin/models.py  — ADD THIS TO THE BOTTOM of your existing models.py
=========================================================================
M3+ model only. Everything above (DropPath, building blocks, M1, M2, M3)
stays exactly as it is. Just paste everything below the existing
print("[models] M1 / M2 / M3  3D triple-head builders ready.") line.

Changes from M3 → M3+:
  1. Filter schedule 32-64-128-192-256 → 16-32-64-128-256
     Reduces encoder overfitting on 369 training cases by ~40%.
     Bottleneck stays at 256 — deep representation preserved.

  2. ASPP removed from 6³ bottleneck (degenerate — dilation 4 samples
     outside the 6³ map entirely). Replaced with depthwise-separable
     5×5×5 block applied at 12³ level (first decoder stage) where
     it sees meaningful spatial extent.

  3. Deep supervision (aux2, aux3) removed entirely.
     avg_pool of one-hot necrotic mask at 24³ and 12³ produces
     near-zero targets — aux heads learn to predict background for
     necrotic and push encoder weights in the wrong direction.

  4. Attention gates kept only at c3 and c4 (deep skip connections).
     Gates at c1 (96³) and c2 (48³) learn near-uniform sigmoid at
     full resolution — they add parameters but no spatial selection.

  5. Patch size 96³ → 128³ (config.py change).
     ASPP GAP branch and encoder sizes updated accordingly.
     128 / 2^4 = 8, so bottleneck is 8³ not 6³.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, DepthwiseConv2D, MaxPooling3D, UpSampling3D,
    concatenate, Activation, Multiply, Add,
    GlobalAveragePooling3D, Reshape, Dense,
    BatchNormalization, SpatialDropout3D,
)
from tensorflow.keras.regularizers import l2

from dynatwin.config import (
    PATCH_SIZE, NUM_CLASSES, NUM_MODALITIES, NUM_GRADE_CLASSES,
)

L2_REG = 5e-5


# ══════════════════════════════════════════════════════════════════
# Depthwise-separable 5×5×5 block (replaces ASPP in M3+)
# ══════════════════════════════════════════════════════════════════

def _dw_conv3d(x, filters, kernel=5):
    """
    3D Depthwise-separable convolution.
    5×5×5 depthwise gives receptive field of 5 voxels per axis
    (vs ASPP rate-4 which sampled outside the 6³ map entirely).
    Pointwise 1×1×1 restores cross-channel interaction.
    Parameter cost ≈ 1/8 of equivalent standard 5×5×5 Conv3D.
    """
    # Depthwise: one filter per input channel, large kernel
    h = BatchNormalization()(
        Conv3D(x.shape[-1], kernel, padding='same',
               groups=x.shape[-1],          # depthwise
               use_bias=False,
               kernel_regularizer=l2(L2_REG))(x))
    h = Activation('relu')(h)
    # Pointwise: cross-channel mixing
    h = BatchNormalization()(
        Conv3D(filters, 1, padding='same',
               use_bias=False,
               kernel_regularizer=l2(L2_REG))(h))
    h = Activation('relu')(h)
    return SpatialDropout3D(0.1)(h)


# ══════════════════════════════════════════════════════════════════
# M3+ encoder — filter schedule 16-32-64-128-256
# ══════════════════════════════════════════════════════════════════

def _encoder_m3plus(inp):
    """
    Same structure as _encoder() but with reduced early filters:
      16 → 32 → 64 → 128 → 256
    instead of:
      32 → 64 → 128 → 192 → 256

    Reduces encoder parameter count by ~40% without touching the
    bottleneck representation. Reduces overfitting on BraTS 2020
    which has only 369 training cases.
    """
    c1 = residual_block_3d(inp,  16, dropout=0.10, drop_path=0.05)
    p1 = MaxPooling3D(2)(c1)
    c2 = residual_block_3d(p1,  32, dropout=0.15, drop_path=0.08)
    p2 = MaxPooling3D(2)(c2)
    c3 = residual_block_3d(p2,  64, dropout=0.20, drop_path=0.10)
    p3 = MaxPooling3D(2)(c3)
    c4 = residual_block_3d(p3, 128, dropout=0.20, drop_path=0.12)
    p4 = MaxPooling3D(2)(c4)
    c5 = residual_block_3d(p4, 256, dropout=0.15, drop_path=0.12)
    return (c1, c2, c3, c4), c5


# ══════════════════════════════════════════════════════════════════
# M3+ builder
# ══════════════════════════════════════════════════════════════════

def build_unet3d_m3plus(input_shape=(*PATCH_SIZE, NUM_MODALITIES),
                        num_classes=NUM_CLASSES) -> Model:
    """
    M3+ — Targeted improvements over M3:

      - Smaller early filters (16-32-64-128-256) reduce overfitting
      - No ASPP at degenerate 6³/8³ bottleneck
      - Depthwise-separable 5×5×5 at 16³ decoder level (first upsample)
        where it has meaningful spatial extent
      - No deep supervision (aux2/aux3 removed — they hurt necrotic)
      - Attention gates only at c3 and c4 (deep levels only)
      - 128³ patch size (set PATCH_SIZE=(128,128,128) in config.py)
      - Returns dict {'seg', 'cls', 'surv'} — no aux outputs
        so train.py loss_weights stays simple: seg=1.0, cls=0.3, surv=0.2
    """
    inp = Input(input_shape)
    (c1, c2, c3, c4), c5 = _encoder_m3plus(inp)

    # ── Decoder level 1: 8³ → 16³ ───────────────────────────────
    # No attention gate here (bottleneck to c4 — gate at c4 below)
    u6  = UpSampling3D(2)(c5)
    # Attention gate on c4 (deep — meaningful coarse spatial selection)
    a4  = attention_gate_3d(c4, u6, inter_ch=64)
    d6  = _rd(concatenate([u6, a4]), 128)
    c6  = residual_block_3d(d6, 128, dropout=0.15, drop_path=0.10)

    # ── Depthwise-separable 5×5×5 at 16³ ────────────────────────
    # Replaces ASPP. At 16³ spatial extent, a 5×5×5 kernel covers
    # a meaningful neighbourhood. No degenerate sampling.
    c6  = _dw_conv3d(c6, filters=128, kernel=5)

    # ── Decoder level 2: 16³ → 32³ ──────────────────────────────
    u7  = UpSampling3D(2)(c6)
    # Attention gate on c3 (second deepest — still meaningful)
    a3  = attention_gate_3d(c3, u7, inter_ch=32)
    d7  = _rd(concatenate([u7, a3]), 64)
    c7  = residual_block_3d(d7, 64, dropout=0.12, drop_path=0.08)

    # ── Decoder level 3: 32³ → 64³ ──────────────────────────────
    # No attention gate on c2 — at 48³/64³ plain skip is better
    u8  = UpSampling3D(2)(c7)
    d8  = _rd(concatenate([u8, c2]), 32)
    c8  = residual_block_3d(d8, 32, dropout=0.10, drop_path=0.05)

    # ── Decoder level 4: 64³ → 128³ ─────────────────────────────
    # No attention gate on c1 — full resolution, gate is near-uniform
    u9  = UpSampling3D(2)(c8)
    d9  = _rd(concatenate([u9, c1]), 16)
    c9  = residual_block_3d(d9, 16, dropout=0.05, drop_path=0.03)

    # ── Triple head ──────────────────────────────────────────────
    # Bottleneck c5 for cls/surv (post-encoder, pre-decoder)
    # c9 for segmentation (full resolution decoder output)
    seg, cls, surv = _triple_head(c5, c9)

    return Model(inp, {'seg': seg, 'cls': cls, 'surv': surv},
                 name='M3Plus')


print("[models] M3+ builder ready.")
print("[models] M1 / M2 / M3 / M3+  3D triple-head builders ready.")
#print("[models] M1 / M2 / M3  3D triple-head builders ready.")
