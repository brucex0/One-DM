import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

def preprocess_handwriting(input_path, output_path, target_size=64):
    """
    Preprocess handwriting image:
    1. Load the image
    2. Filter out white background
    3. Make strokes white on black background
    4. Resize while maintaining aspect ratio
    5. Center on black background
    
    Args:
        input_path: Path to input directory (can contain subfolders)
        output_path: Path to output directory (will mirror input structure)
        target_size: Target size for output images
    """
    # Convert paths to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Get all image files recursively
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(list(input_path.rglob(f'*{ext}')))
        image_files.extend(list(input_path.rglob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        # Calculate relative path from input directory
        rel_path = img_path.relative_to(input_path)
        
        # Create corresponding output directory
        output_dir = output_path / rel_path.parent
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path}")
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to separate strokes from background
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Invert to get white strokes on black background
        inverted = cv2.bitwise_not(binary)
        
        # Find bounding box of content
        coords = cv2.findNonZero(inverted)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            inverted = inverted[y:y+h, x:x+w]
        
        # Convert to PIL Image for better resizing
        pil_img = Image.fromarray(inverted)
        
        # Calculate new size while maintaining aspect ratio
        ratio = target_size / max(pil_img.size)
        new_size = tuple([int(x * ratio) for x in pil_img.size])
        resized = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new black image
        final = Image.new('L', (target_size, target_size), 0)
        
        # Paste resized image in center
        paste_x = (target_size - new_size[0]) // 2
        paste_y = (target_size - new_size[1]) // 2
        final.paste(resized, (paste_x, paste_y))
        
        # Save preprocessed image maintaining folder structure
        output_file = output_path / rel_path
        final.save(output_file)
        print(f"Processed {rel_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess handwriting samples')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing handwriting images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for preprocessed images')
    parser.add_argument('--size', type=int, default=64,
                        help='Target size for output images (default: 64)')
    
    args = parser.parse_args()
    
    preprocess_handwriting(args.input, args.output, args.size)