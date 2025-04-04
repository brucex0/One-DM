import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

def preprocess_handwriting(input_path, output_path, target_size=64):
    """
    Preprocess handwriting image:
    1. Convert to grayscale
    2. Binarize to get black text on white background
    3. Remove noise
    4. Resize while maintaining aspect ratio
    5. Center on white background
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(list(Path(input_path).glob(f'*{ext}')))
        image_files.extend(list(Path(input_path).glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path}")
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # C constant
        )
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert to get black text on white background
        denoised = cv2.bitwise_not(denoised)
        
        # Find text bounding box to crop empty space
        coords = cv2.findNonZero(cv2.bitwise_not(denoised))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            denoised = denoised[y:y+h, x:x+w]
        
        # Convert to PIL Image for better resizing
        pil_img = Image.fromarray(denoised)
        
        # Calculate new size while maintaining aspect ratio
        ratio = target_size / max(pil_img.size)
        new_size = tuple([int(x * ratio) for x in pil_img.size])
        resized = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new white image
        final = Image.new('L', (target_size, target_size), 255)
        
        # Paste resized image in center
        paste_x = (target_size - new_size[0]) // 2
        paste_y = (target_size - new_size[1]) // 2
        final.paste(resized, (paste_x, paste_y))
        
        # Save preprocessed image
        output_file = os.path.join(output_path, img_path.name)
        final.save(output_file)
        print(f"Processed {img_path.name}")

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