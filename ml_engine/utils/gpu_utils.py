"""
GPU Utility Module for InferX-ML
Automatic GPU detection and configuration for TensorFlow, PyTorch, XGBoost, and LightGBM.
If a GPU is available, it will be used automatically. Otherwise, falls back to CPU.
"""
import os
import logging

logger = logging.getLogger(__name__)

# ─── Cache ────────────────────────────────────────────────────────────
_gpu_info_cache = None
_tf_configured = False


def detect_gpus() -> dict:
    """
    Detect available GPUs and return a summary dict.

    Returns:
        dict with keys:
            gpu_available (bool): Whether any GPU is accessible
            gpu_count (int): Number of GPUs detected
            gpu_devices (list[dict]): Per-device info (name, memory_mb)
            cuda_available (bool): Whether CUDA toolkit is reachable
            tensorflow_gpu (bool): TF can see a GPU
            pytorch_gpu (bool): PyTorch can see a GPU
    """
    global _gpu_info_cache
    if _gpu_info_cache is not None:
        return _gpu_info_cache

    info = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_devices": [],
        "cuda_available": False,
        "tensorflow_gpu": False,
        "pytorch_gpu": False,
    }

    # ── Try NVIDIA SMI via pynvml ──────────────────────────────────
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        info["cuda_available"] = True
        info["gpu_count"] = count
        info["gpu_available"] = count > 0
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["gpu_devices"].append({
                "index": i,
                "name": name,
                "memory_total_mb": round(mem.total / (1024 ** 2)),
                "memory_free_mb": round(mem.free / (1024 ** 2)),
                "memory_used_mb": round(mem.used / (1024 ** 2)),
            })
        pynvml.nvmlShutdown()
    except Exception:
        pass  # pynvml not installed or no NVIDIA driver

    # ── TensorFlow GPU check ──────────────────────────────────────
    try:
        import tensorflow as tf
        tf_gpus = tf.config.list_physical_devices("GPU")
        info["tensorflow_gpu"] = len(tf_gpus) > 0
        if not info["gpu_available"] and tf_gpus:
            info["gpu_available"] = True
            info["gpu_count"] = len(tf_gpus)
    except Exception:
        pass

    # ── PyTorch GPU check ─────────────────────────────────────────
    try:
        import torch
        info["pytorch_gpu"] = torch.cuda.is_available()
        if not info["gpu_available"] and info["pytorch_gpu"]:
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass

    _gpu_info_cache = info
    return info


def configure_tensorflow_gpu():
    """
    Configure TensorFlow to use the GPU if available.
    - Enables memory growth so TF doesn't pre-allocate all VRAM.
    - Enables mixed-precision (float16) on capable GPUs for speed.
    Falls back to CPU silently if no GPU is found.
    """
    global _tf_configured
    if _tf_configured:
        return

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # already initialised

            # Mixed precision for Tensor Cores (Volta+, compute >= 7.0)
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                logger.info("[GPU] TensorFlow mixed-precision (float16) enabled")
            except Exception:
                pass

            logger.info(
                f"[GPU] TensorFlow configured — {len(gpus)} GPU(s) available: "
                + ", ".join(g.name for g in gpus)
            )
        else:
            logger.info("[GPU] No TensorFlow GPU found — running on CPU")

    except ImportError:
        logger.debug("[GPU] TensorFlow not installed")

    _tf_configured = True


def get_torch_device():
    """
    Return the best available torch device.

    Returns:
        torch.device  ('cuda' if available, else 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"[GPU] PyTorch using CUDA — {torch.cuda.get_device_name(0)}")
            return device
    except ImportError:
        pass

    try:
        import torch
        device = torch.device("cpu")
    except ImportError:
        device = None  # torch not installed
    logger.info("[GPU] PyTorch using CPU")
    return device


def get_xgboost_gpu_params() -> dict:
    """
    Return XGBoost params to use GPU if CUDA is available.
    Merge these into your XGBClassifier / XGBRegressor params.

    Returns:
        dict — e.g. {'tree_method': 'gpu_hist', 'device': 'cuda'}
              or {} for CPU fallback
    """
    gpu_info = detect_gpus()
    if gpu_info["gpu_available"]:
        try:
            import xgboost as xgb
            # XGBoost >= 2.0 uses 'device' param
            major = int(xgb.__version__.split(".")[0])
            if major >= 2:
                logger.info("[GPU] XGBoost will use CUDA (device='cuda')")
                return {"tree_method": "hist", "device": "cuda"}
            else:
                logger.info("[GPU] XGBoost will use gpu_hist")
                return {"tree_method": "gpu_hist"}
        except Exception:
            pass
    return {}


def get_lightgbm_gpu_params() -> dict:
    """
    Return LightGBM params to use GPU if CUDA is available.
    Merge these into your LGBMClassifier / LGBMRegressor params.

    Returns:
        dict — e.g. {'device': 'gpu'} or {} for CPU fallback
    """
    gpu_info = detect_gpus()
    if gpu_info["gpu_available"]:
        try:
            import lightgbm
            logger.info("[GPU] LightGBM will use GPU")
            return {"device": "gpu"}
        except Exception:
            pass
    return {}


def log_gpu_info():
    """
    Log a human-readable summary of GPU availability.
    Call this once at the start of a training job.
    """
    info = detect_gpus()
    if info["gpu_available"]:
        devices_str = ", ".join(
            f"{d['name']} ({d['memory_total_mb']} MB)" for d in info["gpu_devices"]
        ) or f"{info['gpu_count']} device(s)"
        print(
            f"[GPU] ✓ GPU detected — {devices_str}  |  "
            f"TF-GPU: {info['tensorflow_gpu']}  |  "
            f"PyTorch-GPU: {info['pytorch_gpu']}",
            flush=True,
        )
    else:
        print("[GPU] ✗ No GPU detected — training will use CPU", flush=True)
    return info
