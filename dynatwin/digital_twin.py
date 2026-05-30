# -*- coding: utf-8 -*-
"""
dynatwin/digital_twin.py  — 3D Edition
=========================================
Physics-Informed Digital Twin using BraTS2020 survival CSV.

Changes from v4/v5:
  • SurvivalPredictor now uses CSV age + extent-of-resection as
    additional features alongside PINN-calibrated D/ρ.
  • predictions_to_density works on 3D predicted volumes.
  • TumorGrowthModel unchanged (3D PDE solver).
"""

import numpy as np
import tensorflow as tf


# ══════════════════════════════════════════════════════════════════
# Density helpers  (3D-aware)
# ══════════════════════════════════════════════════════════════════

def predictions_to_density(seg_vol: np.ndarray,
                           wc=1.0, we=1.0, wd=0.4) -> np.ndarray:
    """
    Convert argmax segmentation volume → scalar tumour density [0,1].
    Works for both 2D (H,W) and 3D (D,H,W) volumes.
    """
    c = np.zeros(seg_vol.shape, dtype=np.float32)
    c[seg_vol == 1] = wc   # necrotic
    c[seg_vol == 2] = wd   # edema
    c[seg_vol == 3] = we   # enhancing
    return np.clip(c, 0, 1)


def density_to_multiclass(c: np.ndarray) -> np.ndarray:
    seg = np.zeros_like(c, dtype=np.uint8)
    seg[(c > 0.15) & (c <= 0.50)] = 2
    seg[(c > 0.50) & (c <= 0.75)] = 1
    seg[c > 0.75]                  = 3
    return seg


# ══════════════════════════════════════════════════════════════════
# PDE simulator
# ══════════════════════════════════════════════════════════════════

class TumorGrowthModel:
    """Fisher-KPP reaction-diffusion model (3D)."""
    def __init__(self, D=5e-4, rho=0.02, dx=1.0, dt=0.1):
        self.D=float(D); self.rho=float(rho)
        self.dx=float(dx); self.dt=float(dt)

    def _lap(self, u):
        d = self.dx**2
        return (-6*u + np.roll(u,1,0)+np.roll(u,-1,0)
                     + np.roll(u,1,1)+np.roll(u,-1,1)
                     + np.roll(u,1,2)+np.roll(u,-1,2)) / d

    def simulate(self, c0, days=90, every=7):
        c=c0.copy().astype(np.float32)
        n=int(days/self.dt); s=max(1,int(every/self.dt))
        hist=[c.copy()]; times=[0.0]
        for i in range(1,n+1):
            c=np.clip(c+self.dt*(self.D*self._lap(c)+self.rho*c*(1-c)),0,1)
            if i%s==0: hist.append(c.copy()); times.append(i*self.dt)
        return hist, times


# ══════════════════════════════════════════════════════════════════
# Patient Digital Twin
# ══════════════════════════════════════════════════════════════════

class PatientDigitalTwin:
    def __init__(self, c0, Dr=(0.1,2.0), Rr=(0.01,0.10)):
        self.c0=c0.astype(np.float32)
        self.D_range=Dr; self.rho_range=Rr

    def _sim(self, days=90, every=7, D=None, rho=None):
        D   = D   if D   is not None else np.random.uniform(*self.D_range)
        rho = rho if rho is not None else np.random.uniform(*self.rho_range)
        hist,times = TumorGrowthModel(D,rho).simulate(self.c0,days,every)
        co,ed,en=[],[],[]
        for c in hist:
            seg=density_to_multiclass(c)
            co.append((seg==1).astype(np.uint8))
            ed.append((seg==2).astype(np.uint8))
            en.append((seg==3).astype(np.uint8))
        return {'days':times,'core':co,'edema':ed,'enhancing':en,'D':D,'rho':rho}

    def predict_progression(self,days=90,num_scenarios=5,every=7):
        return [self._sim(days,every) for _ in range(num_scenarios)]


# ══════════════════════════════════════════════════════════════════
# PINN Calibrator
# ══════════════════════════════════════════════════════════════════

class PINNCalibrator(tf.keras.Model):
    def __init__(self, twin, hidden=64):
        super().__init__(); self.twin=twin
        self.logD=tf.Variable(tf.math.log(tf.constant(
            float(np.mean(twin.D_range)),dtype=tf.float32)))
        self.logR=tf.Variable(tf.math.log(tf.constant(
            float(np.mean(twin.rho_range)),dtype=tf.float32)))
        self.mlp=tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(3,)),
            tf.keras.layers.Dense(hidden,activation='tanh'),
            tf.keras.layers.Dense(hidden,activation='tanh'),
            tf.keras.layers.Dense(1,activation='sigmoid'),
        ])
        self._lap=self._vlap(twin.c0).astype(np.float32)

    @property
    def D(self): return tf.exp(self.logD)
    @property
    def R(self): return tf.exp(self.logR)
    def call(self,xyz): return self.mlp(xyz)

    @staticmethod
    def _vlap(v):
        return (-6*v+np.roll(v,1,0)+np.roll(v,-1,0)
                    +np.roll(v,1,1)+np.roll(v,-1,1)
                    +np.roll(v,1,2)+np.roll(v,-1,2))

    def sample_points(self, n=30000):
        shape=self.twin.c0.shape
        if len(shape)==3: Z,H,W=shape
        else: Z=1; H,W=shape
        iz=np.random.randint(0,max(1,Z),n)
        iy=np.random.randint(0,H,n); ix=np.random.randint(0,W,n)
        xs=ix/(W-1)*2-1; ys=iy/(H-1)*2-1
        zs=iz/(max(1,Z)-1)*2-1 if Z>1 else np.zeros(n)
        coords=np.stack([xs,ys,zs],axis=-1).astype(np.float32)
        c0v=self.twin.c0.reshape(-1)[iz*H*W+iy*W+ix].reshape(-1,1).astype(np.float32) \
            if len(shape)==3 else \
            self.twin.c0[iy,ix].reshape(-1,1).astype(np.float32)
        lapv=self._lap.reshape(-1)[iz*H*W+iy*W+ix].reshape(-1,1).astype(np.float32) \
             if len(shape)==3 else \
             self._lap[iy,ix].reshape(-1,1).astype(np.float32)
        return coords,c0v,lapv

    def train_step(self,om,op,n=30000,b=1.0):
        xyz,c0v,lapv=self.sample_points(n)
        xt=tf.constant(xyz); ct=tf.constant(c0v); lt=tf.constant(lapv)
        with tf.GradientTape() as t1:
            dl=tf.reduce_mean(tf.square(self.mlp(xt)-ct))
        om.apply_gradients(zip(t1.gradient(dl,self.mlp.trainable_variables),
                               self.mlp.trainable_variables))
        with tf.GradientTape() as t2:
            mask=tf.cast(ct>0.05,tf.float32)
            res=self.D*lt+self.R*ct*(1-ct)
            pl=tf.reduce_sum(mask*tf.square(res))/(tf.reduce_sum(mask)+1e-8)
        op.apply_gradients(zip(t2.gradient(pl,[self.logD,self.logR]),
                               [self.logD,self.logR]))
        return float(dl+b*pl),float(dl),float(pl),float(self.D),float(self.R)

    def calibrate(self, epochs=500, lr=1e-3, beta=1.0):
        om=tf.keras.optimizers.Adam(lr); op=tf.keras.optimizers.Adam(lr)
        D=rho=pl=dl=0
        for ep in range(1,epochs+1):
            _,dl,pl,D,rho=self.train_step(om,op,30000,beta)
            if ep%100==0 or ep==epochs:
                print(f"  PINN ep{ep:4d}  D={D:.3e}  ρ={rho:.3e}  "
                      f"phys={pl:.5f}  data={dl:.5f}")
        return {'D':D,'rho':rho,'physics_loss':pl,'data_loss':dl}


# ══════════════════════════════════════════════════════════════════
# Survival Predictor  (now uses CSV features)
# ══════════════════════════════════════════════════════════════════

class SurvivalPredictor:
    """
    Parametric survival model combining:
      • PINN-calibrated D/ρ infiltration index
      • Patient age from CSV
      • Extent of resection (GTR=0, STR=1, NA=2) from CSV
      • Predicted tumour volume from segmentation
    """
    EOR_FACTOR = {0: 1.0, 1: 0.75, 2: 0.85}   # GTR best prognosis

    def __init__(self, twin, pinn_params=None, csv_info=None):
        self.twin       = twin
        self.pinn_params= pinn_params or {}
        self.csv_info   = csv_info   or {}   # {'age_norm', 'eor', 'log_os'}

    def estimate_survival(self, scenario: dict, horizon=365) -> dict:
        times = list(range(0, horizon+1, 30))
        D   = float(self.pinn_params.get('D', np.mean(self.twin.D_range)))
        R   = float(self.pinn_params.get('rho', np.mean(self.twin.rho_range)))
        dr  = D / (R + 1e-9)

        # Base OS from D/ρ ratio
        base_os = float(np.clip(500 - dr * 20, 90, 1000))

        # Adjust for age (older → shorter OS)
        age_norm = float(self.csv_info.get('age_norm', 0.0))
        base_os  = base_os * np.exp(-0.15 * age_norm)

        # Adjust for extent of resection
        eor_code = int(self.csv_info.get('eor', 2))
        base_os  = base_os * self.EOR_FACTOR.get(eor_code, 0.85)

        lam  = np.log(2) / max(1, base_os)
        V50  = 25000
        days = scenario['days']
        probs = []
        for t in times:
            idx = np.argmin([abs(d - t) for d in days])
            vol = sum(int(np.sum(scenario[c][idx] > 0))
                      for c in ['core','edema','enhancing'])
            vf  = 0.5 + 1.0 / (1.0 + np.exp(-(vol - V50) / 8000))
            probs.append(float(np.clip(np.exp(-lam * vf * t), 0, 1)))

        med = next((t for t,p in zip(times,probs) if p<=0.5), times[-1])
        return {
            'time_points':              times,
            'survival_probability':     probs,
            'median_survival':          med,
            'D_rho_ratio':              round(dr, 4),
            'calibrated_median_os_days': round(base_os, 1),
        }


class UncertaintyAnalyzer:
    def monte_carlo_uncertainty(self, twin, n=20, days=90):
        sc=[twin.predict_progression(days,1)[0] for _ in range(n)]
        bands={}; T=len(sc[0]['days'])
        for cls in ['core','edema','enhancing']:
            vols=[[int(np.sum(s[cls][t]>0)) for s in sc] for t in range(T)]
            bands[cls]={'median':[np.median(v) for v in vols],
                        'lower':[np.percentile(v,2.5) for v in vols],
                        'upper':[np.percentile(v,97.5) for v in vols],
                        'days':sc[0]['days']}
        return bands

print("[digital_twin] PINN + CSV-augmented SurvivalPredictor ready.")
