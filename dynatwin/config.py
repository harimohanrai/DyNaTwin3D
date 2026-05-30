"""
dynatwin/config.py  — 3D Edition
==================================
All constants, GPU setup, and paths.

GPU selection: automatic health check on all available GPUs.
  1. Queries ECC uncorrectable error count for each GPU.
  2. Respects CUDA_VISIBLE_DEVICES if set — only considers allowed GPUs.
  3. Selects the GPU with most free memory AND zero/lowest ECC errors.
  4. Falls back to next healthiest GPU if preferred one is unhealthy.
  5. Logs the decision clearly so you always know which GPU is active.
"""

import os
import subprocess
import warnings

# XLA flags — set BEFORE tensorflow import
_xla = ('--xla_gpu_strict_conv_algorithm_picker=false '
        '--xla_gpu_enable_latency_hiding_scheduler=true')
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' ' + _xla).strip()

# Suppress TF C++ noise BEFORE import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf          # noqa: E402
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# ══════════════════════════════════════════════════════════════════
# GPU health check + automatic selection
# ══════════════════════════════════════════════════════════════════

def _query_gpu_stats():
    """
    Query nvidia-smi for per-GPU:
      - free memory (MiB)
      - uncorrectable ECC DRAM error count
    Returns list of dicts, one per GPU, indexed by device order.
    """
    stats = []
    try:
        mem_out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,memory.free,memory.total',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip().split('\n')

        ecc_out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,ecc.errors.uncorrected.volatile.dram',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip().split('\n')

        ecc_map = {}
        for line in ecc_out:
            parts = [p.strip() for p in line.split(',')]
            idx   = int(parts[0])
            try:
                ecc = int(parts[1])
            except (ValueError, IndexError):
                ecc = 0   # N/A = ECC not enabled, treat as clean
            ecc_map[idx] = ecc

        for line in mem_out:
            parts     = [p.strip() for p in line.split(',')]
            idx       = int(parts[0])
            free_mib  = int(parts[1])
            total_mib = int(parts[2])
            ecc       = ecc_map.get(idx, 0)
            stats.append({
                'index':      idx,
                'free_mib':   free_mib,
                'total_mib':  total_mib,
                'ecc_errors': ecc,
            })

    except Exception as e:
        print(f"[config] WARNING: nvidia-smi query failed: {e}")

    return stats


def _select_best_gpu(stats, ecc_threshold=50):
    """
    Select best GPU:
      1. Respect CUDA_VISIBLE_DEVICES if set — filter to allowed GPUs only.
      2. Prefer GPUs with ECC uncorrectable errors <= threshold.
      3. Among healthy GPUs pick the one with most free memory.
      4. If ALL GPUs are unhealthy, pick least-bad with warning.
    """
    if not stats:
        return None

    # Fix: respect CUDA_VISIBLE_DEVICES — filter stats to allowed GPUs only
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible is not None:
        allowed = [int(x.strip()) for x in visible.split(',')]
        filtered = [g for g in stats if g['index'] in allowed]
        if filtered:
            print(f"[config] CUDA_VISIBLE_DEVICES={visible} "
                  f"— restricting to GPU(s): {allowed}")
            stats = filtered
        else:
            print(f"[config] WARNING: CUDA_VISIBLE_DEVICES={visible} "
                  f"matched no GPUs — using all available")

    healthy   = [g for g in stats if g['ecc_errors'] <= ecc_threshold]
    candidate = healthy if healthy else stats

    best = max(candidate, key=lambda g: g['free_mib'])

    if not healthy:
        print(f"[config] WARNING: All GPUs exceed ECC threshold "
              f"({ecc_threshold}). Using least-bad GPU:{best['index']} "
              f"(ECC errors: {best['ecc_errors']}). "
              f"Contact server admin.")
    return best['index']


# ── Run health check ────────────────────────────────────────────────
_gpu_stats = _query_gpu_stats()

if _gpu_stats:
    print("[config] GPU health check:")
    for g in _gpu_stats:
        status = "healthy" if g['ecc_errors'] <= 50 else "UNHEALTHY"
        print(f"  GPU:{g['index']}  free={g['free_mib']:,} MiB / "
              f"{g['total_mib']:,} MiB  "
              f"ECC_uncorr={g['ecc_errors']}  [{status}]")
    _selected_idx = _select_best_gpu(_gpu_stats)
    print(f"[config] Selected GPU:{_selected_idx}")
else:
    _selected_idx = None
    print("[config] WARNING: Could not query GPU stats")

# ── Apply GPU selection ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for _g in gpus:
        tf.config.experimental.set_memory_growth(_g, True)

    if _selected_idx is not None and _selected_idx < len(gpus):
        tf.config.set_visible_devices([gpus[_selected_idx]], 'GPU')
        print(f"[config] Using GPU:{_selected_idx} exclusively")
    else:
        print("[config] GPU selection fallback — using all available GPUs")

    tf.config.optimizer.set_jit(True)
else:
    print("[config] WARNING: No GPU found — CPU only")

tf.config.experimental.enable_tensor_float_32_execution(False)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()
NUM_GPUS  = strategy.num_replicas_in_sync
print(f"[config] MirroredStrategy: {NUM_GPUS} replica(s)")
print(f"[config] Precision: {tf.keras.mixed_precision.global_policy().name}")


# ══════════════════════════════════════════════════════════════════
# 3D volume constants
# ══════════════════════════════════════════════════════════════════
PATCH_SIZE        = (128, 128, 128)
PATCH_STRIDE      = 48
PATCHES_PER_CASE  = 6
FG_PATCH_PROB     = 0.85

FULL_VOL_SHAPE    = (240, 240, 155)
NUM_MODALITIES    = 4
NUM_CLASSES       = 4
NUM_GRADE_CLASSES = 2

# ══════════════════════════════════════════════════════════════════
# Training hyper-parameters
# ══════════════════════════════════════════════════════════════════
EPOCHS            = 60
BATCH_SIZE        = 4
MIN_TUMOUR_VOXELS = 500

# ══════════════════════════════════════════════════════════════════
# Data split
# ══════════════════════════════════════════════════════════════════
HOLDOUT_RATIO     = 0.10
N_FOLDS           = 3
RANDOM_SEED       = 42

# ══════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════
TRAIN_DATASET_PATH = (
    '/home/ubuntu/Hari/DynaTwin/Brats2020/'
    'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
)

_HERE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_HERE, '..', 'outputs_m3')
)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SURVIVAL_CSV = os.path.join(TRAIN_DATASET_PATH, 'survival_info.csv')
CSV_PATH     = os.path.join(OUTPUT_DIR, 'ablation_results.csv')

print(f"[config] PATCH_SIZE       : {PATCH_SIZE}")
print(f"[config] BATCH_SIZE       : {BATCH_SIZE}  ({BATCH_SIZE//NUM_GPUS}/GPU)")
print(f"[config] PATCHES_PER_CASE : {PATCHES_PER_CASE}")
print(f"[config] MIN_TUMOUR_VOXELS: {MIN_TUMOUR_VOXELS}")
print(f"[config] OUTPUT_DIR       : {OUTPUT_DIR}")
