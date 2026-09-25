import os
import shutil
import zipfile
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


class DetectionPreprocessor:
    """
    Handles extracting, validating, and formatting datasets specifically 
    for YOLOv8 Object Detection.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the DetectionPreprocessor.
        
        Args:
            data_dir: The target directory to write extracted/formatted files to.
        """
        self.data_dir = os.path.abspath(data_dir)
        self.images_dir = os.path.join(self.data_dir, "images")
        self.labels_dir = os.path.join(self.data_dir, "labels")
        self.yaml_path = os.path.join(self.data_dir, "data.yaml")
        
        # Ensure directories exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def process_zip(self, zip_path: str) -> Dict[str, str]:
        """
        Extracts a YOLO-formatted ZIP file and generates a data.yaml
        for PyTorch/Ultralytics training.
        
        Args:
            zip_path: Path to the uploaded .zip dataset
            
        Returns:
            Dictionary containing the path to the newly generated `data.yaml`
        """
        print(f"[DetectionPreprocessor] Extracting {zip_path} into {self.data_dir}")
        temp_extract_dir = os.path.join(self.data_dir, "temp_extract")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
            
        # Parse the extracted structure to find images and labels
        classes = self._restructure_and_find_classes(temp_extract_dir)
        
        # Clean up temp
        shutil.rmtree(temp_extract_dir)
        
        # Build the data.yaml
        self._generate_yaml(classes)
        
        return {
            "yaml_path": self.yaml_path,
            "classes": classes,
            "num_classes": len(classes)
        }
        
    def _restructure_and_find_classes(self, extract_dir: str) -> List[str]:
        """
        Navigates the unzipped folder. YOLO strictly requires parallel:
        /images
           img1.jpg
        /labels
           img1.txt
           
        Depending on how the user zipped it, we must flatten/move items.
        Also parses a classes.txt or similar if present to find class names.
        
        Returns:
            List of class names
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # Track if we found an explicit classes file
        classes_found = []
        classes_file = None
        
        # 1. Walk to find classes.txt first, and move files to images/labels dirs simultaneously
        for root, dirs, files in os.walk(extract_dir):
            # Skip if we are walking inside the destination directories to avoid circular loops
            if root == self.images_dir or root == self.labels_dir:
                continue
                
            for file in files:
                ext = Path(file).suffix.lower()
                full_path = os.path.join(root, file)
                
                # Check for classes.txt (common in YOLO exports like Roboflow)
                if file.lower() == 'classes.txt':
                    classes_file = full_path
                    continue
                    
                # If Image, move to images_dir
                if ext in image_extensions:
                    dest = os.path.join(self.images_dir, file)
                    shutil.move(full_path, dest)
                    
                # If Label (txt), move to labels_dir
                elif ext == '.txt' and file.lower() != 'classes.txt':
                    dest = os.path.join(self.labels_dir, file)
                    shutil.move(full_path, dest)
                    
        # Parse classes if a file was provided
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                classes_found = [line.strip() for line in f if line.strip()]
        
        # If no classes.txt was found, we have to infer classes dynamically or assign generic names.
        # YOLO format stores class IDs natively in the text files (e.g. `0 0.5 0.5 0.2 0.2`).
        # We must scan all moved txt files to find the maximum class ID.
        if not classes_found:
            print("[DetectionPreprocessor] No classes.txt found. Scanning label files for unique Class IDs.")
            max_class_id = -1
            
            for label_file in os.listdir(self.labels_dir):
                if not label_file.endswith('.txt'):
                    continue
                    
                with open(os.path.join(self.labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:
                            # First element in YOLO format is the class ID
                            try:
                                cls_id = int(parts[0])
                                max_class_id = max(max_class_id, cls_id)
                            except ValueError:
                                pass
                                
            if max_class_id >= 0:
                # Generate generic names: Class_0, Class_1, ...
                classes_found = [f"Class_{i}" for i in range(max_class_id + 1)]
            else:
                # Fallback if 0 labels were found
                classes_found = ["Object"]
                
        return classes_found

    def _generate_yaml(self, classes: List[str]):
        """
        Creates the data.yaml file that Ultralytics YOLO strictly requires.
        """
        yaml_data = {
            'train': self.images_dir,  # We just pass the directory to it
            'val': self.images_dir,    # For simplicity, validate on train if no separate split provided
            
            'nc': len(classes),
            'names': classes
        }
        
        with open(self.yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
            
        print(f"[DetectionPreprocessor] Generated {self.yaml_path} with {len(classes)} classes: {classes}")
