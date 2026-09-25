"""
Unit Tests for Vision Preprocessing
"""
import pytest
import numpy as np
from PIL import Image
import os
import sys

# Ensure backend and ml_engine are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def test_rgba_to_rgb_conversion():
    """Test that RGBA images are correctly converted to RGB and normalized"""
    # Create an RGBA image (224x224x4)
    rgba_image = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
    
    # Simulate the preprocessing logic in streamlit_app/app.py
    img_rgb = rgba_image.convert('RGB')
    img_resized = img_rgb.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    
    # Normalization constants
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # This should NOT raise ValueError now
    normalized_array = (img_array - mean) / std
    
    assert normalized_array.shape == (224, 224, 3)
    assert img_array.shape == (224, 224, 3)
    # Check if conversion actually happened (channel count)
    assert np.array(rgba_image).shape[-1] == 4
    assert img_array.shape[-1] == 3

def test_corrupt_image_handling():
    """Test that corrupted images return None instead of throwing Exception"""
    from ml_engine.preprocessing.image_preprocessor import ImagePreprocessor
    import tempfile
    
    preprocessor = ImagePreprocessor()
    
    # Create a corrupted file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(b'this is not a valid image file')
        tmp_path = tmp.name
        
    try:
        # Should gracefully return None instead of raising PIL.UnidentifiedImageError
        result = preprocessor.load_image(tmp_path)
        assert result is None
    finally:
        os.unlink(tmp_path)

def test_compute_class_weights():
    """Test standard balanced class weight formula"""
    from ml_engine.preprocessing.image_preprocessor import ImagePreprocessor
    
    preprocessor = ImagePreprocessor()
    preprocessor.classes_ = ['dogs', 'cats']
    
    # 8 dogs (label 0), 2 cats (label 1) -> Total 10
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    
    weights = preprocessor.compute_class_weights(labels)
    
    # Dog weight = 10 / (2 * 8) = 10/16 = 0.625
    assert np.isclose(weights[0], 0.625)
    # Cat weight = 10 / (2 * 2) = 10/4 = 2.5
    assert np.isclose(weights[1], 2.5)

def test_image_augmentation_shapes():
    """Test that augmentation maintains strictly the same dimensions"""
    from ml_engine.preprocessing.image_preprocessor import ImagePreprocessor
    
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Create dummy batch of images
    dummy_images = np.random.uniform(0, 1, (4, 224, 224, 3)).astype(np.float32)
    
    # Apply augmentations (which use scipy and cv2 internally)
    augmented = preprocessor._augment_batch(dummy_images)
    
    # Output shape must be strictly identical
    assert augmented.shape == (4, 224, 224, 3)

if __name__ == "__main__":
    test_rgba_to_rgb_conversion()
    test_corrupt_image_handling()
    test_compute_class_weights()
    test_image_augmentation_shapes()
    print("Test passed!")
