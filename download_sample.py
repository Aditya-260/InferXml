import os
import shutil
import random
from PIL import Image, ImageDraw

OUTPUT_ZIP = "synthetic_shapes_dataset.zip"
DATASET_DIR = "synthetic_dataset"
CLASSES = ['circles', 'squares']
NUM_IMAGES_PER_CLASS = 50
IMG_SIZE = (224, 224)

def generate_synthetic_dataset():
    print(f"🎨 Generating synthetic dataset: {NUM_IMAGES_PER_CLASS} images per class...")
    
    # Clean up previous runs
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    
    # Create directories
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
        
    # Generate images
    for cls in CLASSES:
        print(f"   Generating {cls}...")
        for i in range(NUM_IMAGES_PER_CLASS):
            # Create a random background color
            bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new('RGB', IMG_SIZE, color=bg_color)
            draw = ImageDraw.Draw(img)
            
            # shape color
            shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Random position and size
            margin = 40
            x1 = random.randint(margin, IMG_SIZE[0] - margin - 50)
            y1 = random.randint(margin, IMG_SIZE[1] - margin - 50)
            size = random.randint(30, 100)
            x2 = x1 + size
            y2 = y1 + size
            
            if cls == 'circles':
                draw.ellipse([x1, y1, x2, y2], fill=shape_color, outline='black')
            elif cls == 'squares':
                draw.rectangle([x1, y1, x2, y2], fill=shape_color, outline='black')
                
            # Save image
            filename = f"{cls}_{i+1}.jpg"
            img.save(os.path.join(DATASET_DIR, cls, filename))
            
    # Zip it up
    print(f"📦 Zipping dataset into {OUTPUT_ZIP}...")
    shutil.make_archive(OUTPUT_ZIP.replace('.zip', ''), 'zip', DATASET_DIR)
    
    # Cleanup
    shutil.rmtree(DATASET_DIR)
    
    print(f"\n✅ Created '{OUTPUT_ZIP}' with {len(CLASSES) * NUM_IMAGES_PER_CLASS} images.")
    print("👉 Upload this file to InferX-ML to test image classification.")

if __name__ == "__main__":
    generate_synthetic_dataset()
