"""
System Routes — GPU info and system diagnostics
"""
from flask import Blueprint, jsonify

system_bp = Blueprint('system', __name__)


@system_bp.route('/gpu-info', methods=['GET'])
def gpu_info():
    """
    Return GPU availability and device details.
    If a GPU is detected it will be used automatically during training.
    """
    try:
        from ml_engine.utils.gpu_utils import detect_gpus
        info = detect_gpus()
    except ImportError:
        info = {
            "gpu_available": False,
            "gpu_count": 0,
            "gpu_devices": [],
            "cuda_available": False,
            "tensorflow_gpu": False,
            "pytorch_gpu": False,
            "error": "ml_engine.utils not available",
        }

    return jsonify(info), 200
