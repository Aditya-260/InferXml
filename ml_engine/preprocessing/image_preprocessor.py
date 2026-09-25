"""
Image Data Preprocessor
"""
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import zipfile
import tempfile
import shutil


class ImagePreprocessor:
    """
    Preprocessing for image data
    Handles: resizing, normalization, augmentation
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        normalization_mode: str = 'imagenet',  # 'imagenet', '0-1', '-1-1'
        color_mode: str = 'rgb'  # 'rgb', 'grayscale'
    ):
        """
        Initialize preprocessor
        
        Args:
            target_size: (width, height) to resize images to
            normalize: Whether to normalize pixel values
            normalization_mode: Normalization method
            color_mode: Color mode for images
        """
        self.target_size = target_size
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.color_mode = color_mode
        
        # ImageNet normalization constants
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
        
        # Fitted attributes
        self.classes_ = []
        self.class_to_idx_ = {}
    
    def load_image(self, path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image
        
        Args:
            path: Path to image file
            
        Returns:
            Preprocessed image array, or None if the image is corrupted
        """
        try:
            # Load image
            img = Image.open(path)
            
            # Convert color mode
            if self.color_mode == 'rgb':
                img = img.convert('RGB')
            elif self.color_mode == 'grayscale':
                img = img.convert('L')
            
            # Resize
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to array
            img_array = np.array(img, dtype=np.float32)
            
            # Add channel dimension for grayscale
            if self.color_mode == 'grayscale' and len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            
            # Normalize
            if self.normalize:
                img_array = self._normalize(img_array)
            
            return img_array
            
        except (IOError, OSError, SyntaxError) as e:
            # Catch file IO errors, corrupted images, etc.
            print(f"Warning: Skipping corrupted or unreadable image {path} - {str(e)}")
            return None
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Apply normalization"""
        if self.normalization_mode == 'imagenet':
            # Scale to 0-1 first
            img = img / 255.0
            # Apply ImageNet normalization
            img = (img - self.imagenet_mean) / self.imagenet_std
            
        elif self.normalization_mode == '0-1':
            img = img / 255.0
            
        elif self.normalization_mode == '-1-1':
            img = (img / 127.5) - 1.0
        
        return img
    
    def inverse_normalize(self, img: np.ndarray) -> np.ndarray:
        """Inverse normalization for visualization"""
        if self.normalization_mode == 'imagenet':
            img = (img * self.imagenet_std) + self.imagenet_mean
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
        elif self.normalization_mode == '0-1':
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
        elif self.normalization_mode == '-1-1':
            img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        return img
    
    def load_from_directory(
        self,
        directory: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load images from directory structure (folder per class)
        
        Directory structure:
        directory/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
                img4.jpg
        
        Args:
            directory: Root directory with class folders
            
        Returns:
            images: Array of preprocessed images (N, H, W, C)
            labels: Array of label indices
            class_names: List of class names
        """
        images = []
        labels = []
        
        # Get class directories
        class_dirs = sorted([
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ])
        
        self.classes_ = class_dirs
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(class_dirs)}
        
        # Load images
        for class_name in class_dirs:
            class_dir = os.path.join(directory, class_name)
            class_idx = self.class_to_idx_[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        img = self.load_image(filepath)
                        images.append(img)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        return np.array(images), np.array(labels), class_dirs
    
    def load_from_zip(
        self,
        zip_path: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load images from a ZIP file
        
        The ZIP should contain folders organized by class.
        
        Args:
            zip_path: Path to ZIP file
            
        Returns:
            images, labels, class_names
        """
        # Extract to temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Check if there's a single root folder
            contents = os.listdir(temp_dir)
            if len(contents) == 1 and os.path.isdir(os.path.join(temp_dir, contents[0])):
                temp_dir = os.path.join(temp_dir, contents[0])
            
            return self.load_from_directory(temp_dir)
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_data_generator(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False
    ):
        """
        Create a data generator for training
        
        Args:
            images: Array of images
            labels: Array of labels
            batch_size: Batch size
            shuffle: Whether to shuffle
            augment: Whether to apply augmentation
            
        Yields:
            Batches of (images, labels)
        """
        n_samples = len(images)
        indices = np.arange(n_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_images = images[batch_indices].copy()
                batch_labels = labels[batch_indices]
                
                if augment:
                    batch_images = self._augment_batch(batch_images)
                
                yield batch_images, batch_labels
    
    def _augment_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Apply advanced augmentation to batch
        
        Args:
            images: Batch of images
            
        Returns:
            Augmented images
        """
        import scipy.ndimage as ndimage
        augmented = []
        
        for img in images:
            # 1. Random horizontal flip
            if np.random.random() > 0.5:
                img = np.fliplr(img)
            
            # 2. Random brightness adjustment
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                img = np.clip(img * brightness, 0, 1 if self.normalize else 255)
                
            # 3. Random rotation (-15 to +15 degrees)
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                img = ndimage.rotate(img, angle, reshape=False, mode='nearest')
                
            # 4. Random zoom (0.9x to 1.1x)
            if np.random.random() > 0.5:
                zoom_factor = np.random.uniform(0.9, 1.1)
                
                # Apply zoom
                h, w = img.shape[:2]
                zh = int(np.round(h / zoom_factor))
                zw = int(np.round(w / zoom_factor))
                
                # Boundaries to crop/pad
                top = (h - zh) // 2
                left = (w - zw) // 2
                
                # Zoom in (crop)
                if zoom_factor > 1.0:
                    zoomed = img[top:top+zh, left:left+zw]
                    # Since we cropped, resize back to original shape
                    if zoomed.size > 0:
                        # Convert to uint8 for PIL, resize, then back to original dtype
                        orig_dtype = zoomed.dtype
                        if orig_dtype == np.float32 or orig_dtype == np.float64:
                            pil_img = Image.fromarray(((zoomed - zoomed.min()) / (zoomed.max() - zoomed.min() + 1e-8) * 255).astype(np.uint8))
                        else:
                            pil_img = Image.fromarray(zoomed.astype(np.uint8))
                        resized = np.array(pil_img.resize((w, h), Image.Resampling.LANCZOS)).astype(orig_dtype)
                        if orig_dtype in [np.float32, np.float64] and self.normalize:
                             img = resized / 255.0
                        else:
                             img = resized
                
                # Zoom out (pad)
                else:
                    # Pad the exact amount needed to reach h,w after scaling
                    pad_h1, pad_h2 = -top, h - zh + top
                    pad_w1, pad_w2 = -left, w - zw + left
                    zoomed = np.pad(img, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0,0)), mode='constant', constant_values=0)
                    if zoomed.size > 0:
                        orig_dtype = zoomed.dtype
                        if orig_dtype == np.float32 or orig_dtype == np.float64:
                            pil_img = Image.fromarray((np.clip(zoomed, 0, 1) * 255).astype(np.uint8) if self.normalize else np.clip(zoomed, 0, 255).astype(np.uint8))
                        else:
                            pil_img = Image.fromarray(zoomed.astype(np.uint8))
                        resized = np.array(pil_img.resize((w, h), Image.Resampling.LANCZOS)).astype(orig_dtype)
                        if orig_dtype in [np.float32, np.float64] and self.normalize:
                             img = resized / 255.0
                        else:
                             img = resized
            
            augmented.append(img)
        
        return np.array(augmented)
    
    def compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced datasets
        
        Args:
            labels: Array of class labels
            
        Returns:
            Dictionary mapping class index to computed weight
        """
        n_samples = len(labels)
        n_classes = len(self.classes_)
        
        # If no classes defined yet, fall back to unique labels
        if n_classes == 0:
            unique_labels = np.unique(labels)
            n_classes = len(unique_labels)
            
        counts = np.bincount(labels)
        
        weights = {}
        for i, count in enumerate(counts):
            if count > 0:
                # Standard balanced weight formula: n_samples / (n_classes * n_samples_of_class)
                weights[i] = n_samples / (n_classes * count)
            else:
                weights[i] = 1.0  # Fallback for empty classes
                
        return weights
        
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        return {
            'target_size': self.target_size,
            'normalize': self.normalize,
            'normalization_mode': self.normalization_mode,
            'color_mode': self.color_mode,
            'num_classes': len(self.classes_),
            'classes': self.classes_
        }


def prepare_image_for_prediction(
    image_path: str,
    preprocessor: ImagePreprocessor
) -> np.ndarray:
    """
    Prepare a single image for model prediction
    
    Args:
        image_path: Path to image
        preprocessor: Fitted preprocessor
        
    Returns:
        Image array ready for prediction (1, H, W, C)
    """
    img = preprocessor.load_image(image_path)
    return np.expand_dims(img, axis=0)
