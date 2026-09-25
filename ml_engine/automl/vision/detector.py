"""
Object Detection Module (YOLOv8)
Supports: YOLOv8 (ultralytics) architecture for bounding box detection.
GPU: Automatically uses CUDA if available.
"""
import os
import glob
import shutil
import json
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetector:
    """
    Object Detection using Ultralytics YOLOv8.
    Requires images and annotations in YOLO format with a data.yaml.
    """
    
    def __init__(
        self,
        model_size: str = 'n',  # 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xtra)
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        learning_rate: float = 0.01,
        patience: int = 20
    ):
        """
        Initialize Object Detector
        
        Args:
            model_size: YOLOv8 model size (n, s, m, l, x)
            epochs: Training epochs
            batch_size: Batch size (-1 for auto)
            imgsz: Image size for training
            learning_rate: Initial learning rate
            patience: Early stopping patience
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics is required for Object Detection. Install with: pip install ultralytics")
            
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.learning_rate = learning_rate
        self.patience = patience
        
        # State
        self.model_ = None
        self.classes_ = []
        self.metrics_ = {}
        self.best_weights_path_ = None
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[ObjectDetector] Initialized YOLOv8{model_size} on device: {self.device}")

    def fit_from_directory(
        self,
        data_yaml_path: str,
        project_dir: str = 'runs/detect',
        name: str = 'train',
        verbose: bool = True
    ) -> 'ObjectDetector':
        """
        Train YOLO model from a data.yaml configuration.
        
        Args:
            data_yaml_path: Absolute path to the ultralytics data.yaml file
            project_dir: Where to save training runs
            name: Name of this specific run
            verbose: Print training output
            
        Returns:
            self
        """
        # Load pre-trained weights to start (transfer learning)
        base_model_name = f'yolov8{self.model_size}.pt'
        print(f"[ObjectDetector] Loading base model: {base_model_name}")
        self.model_ = YOLO(base_model_name)
        
        # Start training
        print(f"[ObjectDetector] Starting training for {self.epochs} epochs (Batch: {self.batch_size}, Imgsz: {self.imgsz})")
        results = self.model_.train(
            data=data_yaml_path,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.imgsz,
            lr0=self.learning_rate,
            patience=self.patience,
            device=self.device,
            project=project_dir,
            name=name,
            verbose=verbose,
            exist_ok=True  # Overwrite if name exists
        )
        
        # Extract metrics from training
        # Ultralytics results objects contain pandas dataframes of the run
        self.best_weights_path_ = str(results.save_dir / 'weights' / 'best.pt')
        
        # Extract class names from the model's internal names dictionary
        self.classes_ = [self.model_.names[i] for i in range(len(self.model_.names))]
        
        # Extract some basic metrics if available
        try:
            metrics_dict = results.results_dict
            self.metrics_ = {
                'mAP50': float(metrics_dict.get('metrics/mAP50(B)', 0.0)),
                'mAP50-95': float(metrics_dict.get('metrics/mAP50-95(B)', 0.0)),
                'precision': float(metrics_dict.get('metrics/precision(B)', 0.0)),
                'recall': float(metrics_dict.get('metrics/recall(B)', 0.0))
            }
        except Exception as e:
            print(f"[ObjectDetector] Warning: Could not extract detailed metrics: {e}")
            self.metrics_ = {}
            
        return self

    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Predict bounding boxes on an image or list of images.
        
        Args:
            source: Image path, numpy array, PIL Image, or list of them
            conf: Confidence threshold
            iou: NMS IOU threshold
            
        Returns:
            List of prediction dictionaries. Each image gets a dict.
        """
        if self.model_ is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        # Run inference
        results = self.model_.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        
        parsed_predictions = []
        for r in results:
            # Parse boxes for this single image
            boxes = r.boxes
            img_preds = []
            
            if boxes is not None and len(boxes) > 0:
                # xyxy format: [xmin, ymin, xmax, ymax]
                coords = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    cls_id = cls_ids[i]
                    img_preds.append({
                        'class_id': int(cls_id),
                        'class_name': self.model_.names[cls_id],
                        'confidence': float(confs[i]),
                        'bbox': [float(x) for x in coords[i]]  # [xmin, ymin, xmax, ymax]
                    })
            
            parsed_predictions.append({
                'num_detections': len(img_preds),
                'detections': img_preds,
                # Note: Ultralytics doesn't output traditional predict_proba arrays natively
                # because the output grid is huge and sparse. We return the localized confidences.
            })
            
        return parsed_predictions

    def save(self, path: str):
        """
        Save the model weights and necessary metadata.
        For YOLO, we primarily just need to copy the best.pt file.
        
        Args:
            path: Target path to save the weights.
        """
        if self.best_weights_path_ is None or not os.path.exists(self.best_weights_path_):
            raise ValueError("No trained weights found to save. Was training successful?")
            
        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Enforce .pt extension for Ultralytics
        if not path.endswith('.pt'):
            path = path + '.pt'
            
        # Copy the best weights from the runs/ directory to the requested path
        shutil.copy2(self.best_weights_path_, path)
        print(f"[ObjectDetector] Saved YOLO weights to: {path}")
        
        # Save metadata companion file
        meta_path = path.replace('.pt', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'model_type': 'yolov8',
                'model_size': self.model_size,
                'classes': self.classes_,
                'metrics': self.metrics_,
                'imgsz': self.imgsz
            }, f)
            
    @classmethod
    def load(cls, path: str) -> 'ObjectDetector':
        """
        Load a trained ObjectDetector from a .pt file.
        
        Args:
            path: Path to the .pt weights file.
            
        Returns:
            Instantiated ObjectDetector
        """
        if not path.endswith('.pt'):
            path = path + '.pt'
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")
            
        # Instantiate empty shell
        detector = cls()
        
        # Load weights into YOLO framework
        detector.model_ = YOLO(path)
        detector.best_weights_path_ = path
        
        # Extract class names from loaded weights
        detector.classes_ = [detector.model_.names[i] for i in range(len(detector.model_.names))]
        
        # Attempt to load metadata companion if it exists
        meta_path = path.replace('.pt', '_meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                detector.metrics_ = meta.get('metrics', {})
                detector.imgsz = meta.get('imgsz', 640)
                detector.model_size = meta.get('model_size', 'n')
            except Exception as e:
                print(f"[ObjectDetector] Warning: Failed to load metadata: {e}")
                
        return detector
