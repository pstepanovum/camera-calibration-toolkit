"""
Checkerboard Pattern Diagnostic Tool

This script helps diagnose why checkerboard corner detection is failing.
It will:
1. Show you one of your images
2. Try different pattern sizes to find the correct one
3. Visualize what OpenCV sees
4. Give recommendations

Run this BEFORE running stereo_calibrate.py
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import PathConfig

def test_pattern_size(image, pattern_size):
    """
    Test if a specific pattern size can be detected.
    
    Parameters
    ----------
    image : ndarray
        Input image
    pattern_size : tuple
        (width, height) of internal corners to test
        
    Returns
    -------
    tuple
        (success, corners) - success is bool, corners is array or None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(
        gray, 
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + 
        cv2.CALIB_CB_FAST_CHECK + 
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    return ret, corners


def main():
    print("=" * 70)
    print("CHECKERBOARD PATTERN DIAGNOSTIC TOOL")
    print("=" * 70)
    
    # Load a test image
    test_image_path = Path(PathConfig.STEREO_LEFT_DIR) / 'calibration_left' / 'imageL10.jpg'
    
    print(f"\nLoading test image: {test_image_path}")
    
    if not test_image_path.exists():
        print(f"ERROR: Test image not found!")
        print(f"Looking for: {test_image_path}")
        # Try to find any image
        calib_dir = Path(PathConfig.STEREO_LEFT_DIR) / 'calibration_left'
        images = list(calib_dir.glob('*.jpg')) + list(calib_dir.glob('*.png'))
        if images:
            test_image_path = images[0]
            print(f"Using instead: {test_image_path}")
        else:
            print("No images found!")
            return
    
    img = cv2.imread(str(test_image_path))
    if img is None:
        print("ERROR: Could not load image")
        return
    
    print(f"Image size: {img.shape[1]} x {img.shape[0]} pixels")
    
    # Show the image
    print("\n" + "=" * 70)
    print("STEP 1: VISUAL INSPECTION")
    print("=" * 70)
    print("\nShowing your checkerboard image...")
    print("Please COUNT the internal corners:")
    print("  - Internal corners = where 4 squares meet (NOT the outer edges)")
    print("  - Count horizontal corners (width)")
    print("  - Count vertical corners (height)")
    print("\nPress any key when done counting...")
    
    cv2.imshow('Your Checkerboard - Count Internal Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("STEP 2: PATTERN SIZE DETECTION")
    print("=" * 70)
    
    # Common checkerboard sizes to test
    common_patterns = [
        (9, 6),   # Standard small checkerboard
        (8, 6),   # Another common size
        (10, 7),  # Medium checkerboard
        (9, 7),
        (14, 10), # Your current config
        (13, 9),
        (12, 9),
        (11, 8),
        (10, 8),
        (7, 5),
        (6, 4),
        (8, 5),
    ]
    
    print("\nTrying common pattern sizes...\n")
    successful_patterns = []
    
    for pattern in common_patterns:
        ret, corners = test_pattern_size(img, pattern)
        status = "✓ SUCCESS" if ret else "✗ FAILED"
        print(f"{pattern[0]:2d} x {pattern[1]:2d} corners: {status}")
        
        if ret:
            successful_patterns.append((pattern, corners))
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if len(successful_patterns) == 0:
        print("\n❌ NO PATTERNS DETECTED!")
        print("\nPossible issues:")
        print("  1. Checkerboard is not clearly visible")
        print("  2. Image is blurry or out of focus")
        print("  3. Poor lighting or glare")
        print("  4. Checkerboard is partially occluded")
        print("  5. Not a standard black/white checkerboard pattern")
        print("\nRecommendations:")
        print("  - Check image quality (open imageL10.jpg and inspect)")
        print("  - Ensure checkerboard fills ~50-70% of frame")
        print("  - Improve lighting conditions")
        print("  - Take new calibration images")
        
    elif len(successful_patterns) == 1:
        pattern, corners = successful_patterns[0]
        print(f"\n✅ FOUND YOUR PATTERN: {pattern[0]} x {pattern[1]} corners")
        print("\nUpdate your config.py:")
        print(f"  PATTERN_SIZE = {pattern}")
        
        # Show the detection
        print("\nShowing detected corners...")
        print("Press any key to close...")
        img_vis = img.copy()
        cv2.drawChessboardCorners(img_vis, pattern, corners, True)
        cv2.imshow('Detected Pattern', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print(f"\n⚠️  MULTIPLE PATTERNS DETECTED ({len(successful_patterns)})")
        print("\nThis might mean:")
        print("  - OpenCV is finding subpatterns")
        print("  - The largest pattern is usually correct")
        print("\nDetected patterns (largest first):")
        
        # Sort by number of corners (descending)
        successful_patterns.sort(key=lambda x: x[0][0] * x[0][1], reverse=True)
        
        for i, (pattern, corners) in enumerate(successful_patterns):
            total_corners = pattern[0] * pattern[1]
            marker = "← RECOMMENDED" if i == 0 else ""
            print(f"  {pattern[0]:2d} x {pattern[1]:2d} = {total_corners:3d} corners {marker}")
        
        # Show the largest pattern
        pattern, corners = successful_patterns[0]
        print(f"\nRecommended config:")
        print(f"  PATTERN_SIZE = {pattern}")
        
        print("\nShowing the largest detected pattern...")
        print("Press any key to close...")
        img_vis = img.copy()
        cv2.drawChessboardCorners(img_vis, pattern, corners, True)
        cv2.imshow('Recommended Pattern (Largest)', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Manual input option
    print("\n" + "=" * 70)
    print("MANUAL PATTERN SIZE TEST")
    print("=" * 70)
    print("\nIf you want to test a specific pattern size, enter it now.")
    print("Format: width,height (e.g., 9,6)")
    print("Press Enter to skip.")
    
    user_input = input("Pattern size: ").strip()
    
    if user_input:
        try:
            w, h = map(int, user_input.split(','))
            pattern = (w, h)
            print(f"\nTesting pattern: {pattern}")
            
            ret, corners = test_pattern_size(img, pattern)
            
            if ret:
                print(f"✅ SUCCESS: {pattern[0]} x {pattern[1]} corners detected!")
                img_vis = img.copy()
                cv2.drawChessboardCorners(img_vis, pattern, corners, True)
                cv2.imshow('Your Pattern', img_vis)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"❌ FAILED: Could not detect {pattern[0]} x {pattern[1]} pattern")
        except:
            print("Invalid format. Skipping.")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Update config.py with the correct PATTERN_SIZE")
    print("  2. Run stereo_calibrate.py again")
    print("=" * 70)


if __name__ == "__main__":
    main()