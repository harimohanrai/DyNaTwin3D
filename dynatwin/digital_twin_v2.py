# -*- coding: utf-8 -*-
"""
dynatwin/digital_twin_v2.py  — M3+ Edition
============================================
Corrected version of digital_twin.py for use with M3+.
Original digital_twin.py is untouched — M1/M2/M3 still use that.

Fixes applied:

  Fix 1  predictions_to_density weights corrected.
         Original: necrotic=1.0, edema=0.4, enhancing=1.0
         Necrotic core is dead tissue — assigning it density 1.0
         (same as actively proliferating enhancing tumour) causes
         the PINN to calibrate D/rho against a physically wrong map.
         Fixed: necrotic=0.30, edema=0.15, enhancing=1.0
         This reflects the biological reality that necrotic tissue
         has near-zero viable cell density.

  Fix 2  TumorGrowthModel._lap periodic boundary → zero-flux (Neumann).
         np.roll wraps around the volume edges — tumour cells that
         diffuse off one face reappear on the opposite face.
         For a 240×240×155 skull volume this is physically wrong.
         Zero-flux boundary (du/dn=0 at boundary) correctly models
         no-flux at the skull surface.

  Fix 3  SurvivalPredictor now accepts model_pred_os from the M3+
         surv head (trained log_os prediction) and blends it with
         the PINN-based OS estimate.
         Original code discarded the trained surv head output entirely.

  Fix 4  train.py integration — extract surv head prediction from
         best_model before building SurvivalPredictor.
         See usage example at the bottom of this file.
"""

import numpy as np
import tensorflow as tf


# ══════════════════════════════════════════════════════════════════
# Density helpers  (Fix 1)
# ══════════════════════════════════════════════════════════════════

def predictions_to_density(seg_vol: np.ndarray,
                            wc: float = 0.30,
                            we: float = 1.00,
                            wd: float = 0.15) -> np.ndarray:
    """
    Convert argmax segmentation volume → scalar tumour density [0,1].

    Fix 1: corrected per-class density weights.
      wc=0.30  necrotic  — dead tissue, low viable cell density
      we=1.00  enhancing — actively proliferating, high density
      wd=0.15  edema     — infiltrating cells, sparse density

    Original used wc=1.0 (necrotic same as enhancing) which caused
    the PINN Fisher-KPP model to treat dead core as maximally active
    tumour, producing physically wrong D/rho calibration.

    Works for both 2D (H,W) and 3D (D,H,W) argmax volumes.
    Labels must be post-remap: 0=bg, 1=necrotic, 2=edema, 3=enhancing.
    """
    c = np.zeros(seg_vol.shape, dtype=np.float32)
    c[seg_vol == 1] = wc   # necrotic  → low density
    c[seg_vol == 2] = wd   # edema     → infiltrating
    c[seg_vol == 3] = we   # enhancing → high density
    return np.clip(c, 0.0, 1.0)


def density_to_multiclass(c: np.ndarray) -> np.ndarray:
    """Inverse map: density scalar → argmax label. Unchanged from v1."""
    seg = np.zeros_like(c, dtype=np.uint8)
    seg[(c > 0.10) & (c <= 0.40)] = 2   # edema band
    seg[(c > 0.40) & (c <= 0.70)] = 1   # necrotic band
    seg[c > 0.70]                  = 3   # enhancing band
    return seg


# ══════════════════════════════════════════════════════════════════
# PDE simulator  (Fix 2 — zero-flux boundary)
# ══════════════════════════════════════════════════════════════════

class TumorGrowthModel:
    """
    Fisher-KPP reaction-diffusion model (3D).

    Fix 2: _lap now uses zero-flux (Neumann) boundary conditions.
    np.roll wraps the domain periodically — tumour diffusing off one
    face reappears on the opposite face, which is physically wrong
    for a finite skull volume.  Zero-flux correctly models no-flux
    at the skull boundary: du/dn = 0.
    """

    def __init__(self, D: float = 5e-4, rho: float = 0.02,
                 dx: float = 1.0, dt: float = 0.1):
        self.D   = float(D)
        self.rho = float(rho)
        self.dx  = float(dx)
        self.dt  = float(dt)

    def _lap(self, u: np.ndarray) -> np.ndarray:
        """
        Fix 2: 3D finite-difference Laplacian with zero-flux boundary.
        For each axis, the neighbour at the boundary is set equal to
        the boundary voxel itself (du/dn = 0 → neighbour = self).
        This correctly prevents flux across the domain edge.
        """
        d   = self.dx ** 2
        lap = -6.0 * u.copy()

        for axis in range(3):
            # Build index helpers for boundary slices
            def _sl(a, s):
                return tuple(s if i == a else slice(None)
                             for i in range(3))

            # Forward neighbour (shift -1 along axis)
            fwd = np.roll(u, -1, axis=axis)
            # Zero-flux: last slice forward neighbour = last slice itself
            fwd[_sl(axis, -1)] = u[_sl(axis, -1)]

            # Backward neighbour (shift +1 along axis)
            bwd = np.roll(u, 1, axis=axis)
            # Zero-flux: first slice backward neighbour = first slice itself
            bwd[_sl(axis, 0)] = u[_sl(axis, 0)]

            lap = lap + fwd + bwd

        return lap / d

    def simulate(self, c0: np.ndarray,
                 days: int = 90,
                 every: int = 7):
        """
        Euler integration of Fisher-KPP PDE.
        Returns (history_list, time_list).
        """
        c      = c0.copy().astype(np.float32)
        n      = int(days / self.dt)
        s      = max(1, int(every / self.dt))
        hist   = [c.copy()]
        times  = [0.0]

        for i in range(1, n + 1):
            c = np.clip(
                c + self.dt * (self.D * self._lap(c) + self.rho * c * (1 - c)),
                0.0, 1.0)
            if i % s == 0:
                hist.append(c.copy())
                times.append(i * self.dt)

        return hist, times


# ══════════════════════════════════════════════════════════════════
# Patient Digital Twin
# ══════════════════════════════════════════════════════════════════

class PatientDigitalTwin:
    """Unchanged from v1 except uses corrected TumorGrowthModel."""

    def __init__(self, c0: np.ndarray,
                 Dr: tuple = (0.1, 2.0),
                 Rr: tuple = (0.01, 0.10)):
        self.c0        = c0.astype(np.float32)
        self.D_range   = Dr
        self.rho_range = Rr

    def _sim(self, days: int = 90, every: int = 7,
             D: float = None, rho: float = None) -> dict:
        D   = D   if D   is not None else np.random.uniform(*self.D_range)
        rho = rho if rho is not None else np.random.uniform(*self.rho_range)

        # Fix 2: TumorGrowthModel now uses zero-flux Laplacian
        hist, times = TumorGrowthModel(D, rho).simulate(
            self.c0, days, every)

        co, ed, en = [], [], []
        for c in hist:
            seg = density_to_multiclass(c)
            co.append((seg == 1).astype(np.uint8))
            ed.append((seg == 2).astype(np.uint8))
            en.append((seg == 3).astype(np.uint8))

        return {'days': times, 'core': co, 'edema': ed,
                'enhancing': en, 'D': D, 'rho': rho}

    def predict_progression(self, days: int = 90,
                            num_scenarios: int = 5,
                            every: int = 7) -> list:
        return [self._sim(days, every) for _ in range(num_scenarios)]


# ══════════════════════════════════════════════════════════════════
# PINN Calibrator  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════

class PINNCalibrator(tf.keras.Model):
    """Unchanged from v1. Operates on corrected density map from Fix 1."""

    def __init__(self, twin: PatientDigitalTwin, hidden: int = 64):
        super().__init__()
        self.twin  = twin
        self.logD  = tf.Variable(tf.math.log(tf.constant(
            float(np.mean(twin.D_range)), dtype=tf.float32)))
        self.logR  = tf.Variable(tf.math.log(tf.constant(
            float(np.mean(twin.rho_range)), dtype=tf.float32)))
        self.mlp   = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(3,)),
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(hidden, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        self._lap_cache = self._vlap(twin.c0).astype(np.float32)

    @property
    def D(self):   return tf.exp(self.logD)

    @property
    def R(self):   return tf.exp(self.logR)

    def call(self, xyz):
        return self.mlp(xyz)

    @staticmethod
    def _vlap(v: np.ndarray) -> np.ndarray:
        """Vectorised Laplacian for PINN (periodic ok here — used for
        sampling physics residual interior points only, not propagation)."""
        return (-6 * v
                + np.roll(v,  1, 0) + np.roll(v, -1, 0)
                + np.roll(v,  1, 1) + np.roll(v, -1, 1)
                + np.roll(v,  1, 2) + np.roll(v, -1, 2))

    def sample_points(self, n: int = 30000):
        shape = self.twin.c0.shape
        if len(shape) == 3:
            Z, H, W = shape
        else:
            Z = 1; H, W = shape

        iz = np.random.randint(0, max(1, Z), n)
        iy = np.random.randint(0, H, n)
        ix = np.random.randint(0, W, n)

        xs = ix / (W - 1) * 2 - 1
        ys = iy / (H - 1) * 2 - 1
        zs = iz / (max(1, Z) - 1) * 2 - 1 if Z > 1 else np.zeros(n)

        coords = np.stack([xs, ys, zs], axis=-1).astype(np.float32)

        flat = self.twin.c0.reshape(-1)
        idx  = iz * H * W + iy * W + ix
        c0v  = flat[idx].reshape(-1, 1).astype(np.float32)

        flat_lap = self._lap_cache.reshape(-1)
        lapv     = flat_lap[idx].reshape(-1, 1).astype(np.float32)

        return coords, c0v, lapv

    def train_step(self, opt_mlp, opt_phys,
                   n: int = 30000, beta: float = 1.0):
        xyz, c0v, lapv = self.sample_points(n)
        xt = tf.constant(xyz)
        ct = tf.constant(c0v)
        lt = tf.constant(lapv)

        with tf.GradientTape() as t1:
            dl = tf.reduce_mean(tf.square(self.mlp(xt) - ct))
        opt_mlp.apply_gradients(
            zip(t1.gradient(dl, self.mlp.trainable_variables),
                self.mlp.trainable_variables))

        with tf.GradientTape() as t2:
            mask = tf.cast(ct > 0.05, tf.float32)
            res  = self.D * lt + self.R * ct * (1 - ct)
            pl   = tf.reduce_sum(mask * tf.square(res)) / (
                   tf.reduce_sum(mask) + 1e-8)
        opt_phys.apply_gradients(
            zip(t2.gradient(pl, [self.logD, self.logR]),
                [self.logD, self.logR]))

        return (float(dl + beta * pl), float(dl),
                float(pl), float(self.D), float(self.R))

    def calibrate(self, epochs: int = 500,
                  lr: float = 1e-3,
                  beta: float = 1.0) -> dict:
        opt_mlp  = tf.keras.optimizers.Adam(lr)
        opt_phys = tf.keras.optimizers.Adam(lr)
        D = rho = pl = dl = 0.0

        for ep in range(1, epochs + 1):
            _, dl, pl, D, rho = self.train_step(
                opt_mlp, opt_phys, 30000, beta)
            if ep % 100 == 0 or ep == epochs:
                print(f"  PINN ep{ep:4d}  D={D:.3e}  ρ={rho:.3e}  "
                      f"phys={pl:.5f}  data={dl:.5f}")

        return {'D': D, 'rho': rho,
                'physics_loss': pl, 'data_loss': dl}


# ══════════════════════════════════════════════════════════════════
# Survival Predictor  (Fix 3 — uses M3+ surv head output)
# ══════════════════════════════════════════════════════════════════

class SurvivalPredictor:
    """
    Parametric survival model combining:
      • PINN-calibrated D/ρ infiltration index
      • Patient age from CSV
      • Extent of resection (GTR=0, STR=1, NA=2) from CSV
      • Predicted tumour volume from segmentation
      • Fix 3: M3+ surv head predicted log_os (blended 50/50 with PINN)

    Fix 3: original code discarded the trained surv head output entirely.
    M3+ surv head is trained on actual OS data — blending it with the
    PINN estimate gives a more calibrated OS prediction than either alone.
    """
    EOR_FACTOR = {0: 1.00,   # GTR — best prognosis
                  1: 0.75,   # STR
                  2: 0.85}   # Unknown

    def __init__(self, twin: PatientDigitalTwin,
                 pinn_params: dict = None,
                 csv_info:    dict = None,
                 model_pred_os: float = None):
        """
        Parameters
        ----------
        twin            : PatientDigitalTwin
        pinn_params     : dict with keys 'D', 'rho' from PINNCalibrator.calibrate()
        csv_info        : dict with keys 'age_norm', 'eor', 'log_os', 'os_mask'
        model_pred_os   : Fix 3 — predicted OS in days from M3+ surv head.
                          Pass None to use PINN only (M1/M2/M3 behaviour).
        """
        self.twin          = twin
        self.pinn_params   = pinn_params   or {}
        self.csv_info      = csv_info      or {}
        self.model_pred_os = model_pred_os  # None for M1/M2/M3

    def estimate_survival(self, scenario: dict,
                          horizon: int = 365) -> dict:
        times = list(range(0, horizon + 1, 30))

        D  = float(self.pinn_params.get('D',   np.mean(self.twin.D_range)))
        R  = float(self.pinn_params.get('rho', np.mean(self.twin.rho_range)))
        dr = D / (R + 1e-9)

        # PINN-based OS estimate from D/rho ratio
        pinn_os = float(np.clip(500 - dr * 20, 90, 1000))

        # Age adjustment
        age_norm = float(self.csv_info.get('age_norm', 0.0))
        pinn_os  = pinn_os * np.exp(-0.15 * age_norm)

        # Extent of resection adjustment
        eor_code = int(self.csv_info.get('eor', 2))
        pinn_os  = pinn_os * self.EOR_FACTOR.get(eor_code, 0.85)

        # Fix 3: blend PINN OS with M3+ surv head prediction if available
        if self.model_pred_os is not None and self.model_pred_os > 0:
            # 50/50 blend — equal weight to physics and data-driven estimate
            base_os = 0.50 * pinn_os + 0.50 * float(self.model_pred_os)
            print(f"  [survival] PINN OS={pinn_os:.0f}d  "
                  f"Model OS={self.model_pred_os:.0f}d  "
                  f"Blended OS={base_os:.0f}d")
        else:
            base_os = pinn_os

        # Survival curve
        lam  = np.log(2) / max(1.0, base_os)
        V50  = 25000
        days = scenario['days']
        probs = []

        for t in times:
            idx = int(np.argmin([abs(d - t) for d in days]))
            vol = sum(
                int(np.sum(scenario[c][idx] > 0))
                for c in ['core', 'edema', 'enhancing'])
            vf  = 0.5 + 1.0 / (1.0 + np.exp(-(vol - V50) / 8000))
            probs.append(float(np.clip(np.exp(-lam * vf * t), 0.0, 1.0)))

        med = next((t for t, p in zip(times, probs) if p <= 0.5),
                   times[-1])

        return {
            'time_points':               times,
            'survival_probability':      probs,
            'median_survival':           med,
            'D_rho_ratio':               round(dr, 4),
            'calibrated_median_os_days': round(base_os, 1),
            'pinn_os_days':              round(pinn_os, 1),
            'model_os_days':             round(float(self.model_pred_os), 1)
                                         if self.model_pred_os else None,
        }


# ══════════════════════════════════════════════════════════════════
# Uncertainty Analyzer  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════

class UncertaintyAnalyzer:
    def monte_carlo_uncertainty(self, twin: PatientDigitalTwin,
                                n: int = 20,
                                days: int = 90) -> dict:
        sc    = [twin.predict_progression(days, 1)[0] for _ in range(n)]
        bands = {}
        T     = len(sc[0]['days'])

        for cls in ['core', 'edema', 'enhancing']:
            vols = [[int(np.sum(s[cls][t] > 0)) for s in sc]
                    for t in range(T)]
            bands[cls] = {
                'median': [np.median(v)          for v in vols],
                'lower':  [np.percentile(v, 2.5) for v in vols],
                'upper':  [np.percentile(v, 97.5)for v in vols],
                'days':   sc[0]['days'],
            }
        return bands


# ══════════════════════════════════════════════════════════════════
# train.py integration — Fix 3 usage example
# ══════════════════════════════════════════════════════════════════
#
# In train.py, replace the digital twin block with this for M3+:
#
#   from dynatwin.digital_twin_v2 import (
#       predictions_to_density, PatientDigitalTwin,
#       PINNCalibrator, UncertaintyAnalyzer, SurvivalPredictor,
#   )
#
#   demo_id  = holdout_ids[0]
#   X, Y     = load_volume(demo_id)
#   seg_pred = sliding_window_inference(best_model, X)
#   seg_pred = morphological_postprocess(seg_pred)
#
#   # Fix 1: corrected density weights (necrotic=0.30, not 1.0)
#   c0 = predictions_to_density(seg_pred)
#
#   # Fix 3: extract surv head prediction from M3+
#   pd_, ph, pw = PATCH_SIZE
#   X_patch = tf.expand_dims(
#       tf.convert_to_tensor(
#           X[:pd_, :ph, :pw], dtype=tf.float32), axis=0)
#   model_out    = best_model(X_patch, training=False)
#   pred_log_os  = float(model_out['surv'][0, 0])
#   pred_os_days = float(np.exp(np.clip(pred_log_os, 0, 8)))
#
#   twin  = PatientDigitalTwin(c0)
#   pinn  = PINNCalibrator(twin)
#   pr    = pinn.calibrate(500, 1e-3, 0.8)
#   twin.D_range   = (0.85 * pr['D'],   1.15 * pr['D'])
#   twin.rho_range = (0.85 * pr['rho'], 1.15 * pr['rho'])
#   sinfo     = survival_info(demo_id, surv_df)
#   scenarios = twin.predict_progression(90, 5)
#
#   # Pass model_pred_os for Fix 3 blending
#   sp   = SurvivalPredictor(twin, pr, sinfo,
#                            model_pred_os=pred_os_days)
#   surv = sp.estimate_survival(scenarios[0])
#
# For M1/M2/M3 keep using digital_twin.py (original) — no changes there.
# ══════════════════════════════════════════════════════════════════

print("[digital_twin_v2] Fix1(density) Fix2(boundary) Fix3(surv-blend) ready.")
