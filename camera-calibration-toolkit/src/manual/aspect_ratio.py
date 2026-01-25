"""
Manual Calibration: Aspect Ratio Estimation

This script allows manual measurement of a square object in an image
to determine the pixel aspect ratio (fx/fy).

Theory:
- If pixels are square (aspect ratio = 1), a physical square appears square in the image
- If pixels are rectangular, a physical square appears as a rectangle
- By measuring a known square, we can determine fx/fy

Usage:
1. Take a photo of a square object positioned perpendicular to the camera
2. Run this script and click on the four corners of the square
3. Script calculates the aspect ratio from the measurements
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import PathConfig


class AspectRatioEstimator:
    """
    Estimates pixel aspect ratio from manual measurements of a square object.
    """
    
    def __init__(self):
        self.image = None
        self.image_path = None
        self.corners = []
        self.display_image = None
        
    def load_image(self, image_path):
        """
        Load the image containing the square object.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = self.image.copy()
        print(f"\nImage loaded: {os.path.basename(image_path)}")
        print(f"Resolution: {self.image.shape[1]} x {self.image.shape[0]} pixels")
        
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse clicks to record corner positions.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((x, y))
                print(f"Corner {len(self.corners)}: ({x}, {y})")
                
                # Draw the corner on the display image
                cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.display_image, str(len(self.corners)), 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
                
                # Draw lines between corners
                if len(self.corners) > 1:
                    cv2.line(self.display_image, 
                            self.corners[-2], self.corners[-1], 
                            (0, 255, 0), 2)
                
                # Close the square when 4 corners are selected
                if len(self.corners) == 4:
                    cv2.line(self.display_image, 
                            self.corners[3], self.corners[0], 
                            (0, 255, 0), 2)
                    print("\nAll 4 corners selected! Press any key to continue...")
                
                cv2.imshow('Select Square Corners', self.display_image)
    
    def select_corners_interactive(self):
        """
        Interactive GUI for selecting the four corners of the square.
        
        Instructions:
        - Click on the four corners of the square in order (clockwise or counter-clockwise)
        - Press 'r' to reset if you make a mistake
        - Press any other key when done
        """
        print("\n" + "=" * 70)
        print("INTERACTIVE CORNER SELECTION")
        print("=" * 70)
        print("\nInstructions:")
        print("  1. Click on the FOUR corners of your square object")
        print("  2. Click in order: top-left, top-right, bottom-right, bottom-left")
        print("     (or any consistent clockwise/counter-clockwise order)")
        print("  3. Press 'r' to RESET if you make a mistake")
        print("  4. Press any other key when all 4 corners are marked")
        print("\nWindow will open now...")
        
        cv2.namedWindow('Select Square Corners')
        cv2.setMouseCallback('Select Square Corners', self.mouse_callback)
        
        while True:
            cv2.imshow('Select Square Corners', self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            # Press 'r' to reset
            if key == ord('r'):
                print("\nResetting corners...")
                self.corners = []
                self.display_image = self.image.copy()
                cv2.imshow('Select Square Corners', self.display_image)
            
            # Press any other key to finish (after 4 corners selected)
            elif key != 255 and len(self.corners) == 4:
                break
            
            # Press 'q' to quit early
            elif key == ord('q'):
                print("\nQuitting...")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def calculate_aspect_ratio(self, physical_size_mm):
        """
        Calculate aspect ratio from the selected corners.
        
        Parameters
        ----------
        physical_size_mm : float
            Physical size of the square in millimeters
            
        Returns
        -------
        dict
            Dictionary containing aspect ratio and measurements
        """
        if len(self.corners) != 4:
            raise ValueError("Need exactly 4 corners to calculate aspect ratio")
        
        print("\n" + "=" * 70)
        print("CALCULATING ASPECT RATIO")
        print("=" * 70)
        
        corners = np.array(self.corners, dtype=np.float32)
        
        # Calculate the four side lengths
        side_lengths = []
        for i in range(4):
            j = (i + 1) % 4
            length = np.linalg.norm(corners[i] - corners[j])
            side_lengths.append(length)
            print(f"Side {i+1} length: {length:.2f} pixels")
        
        # Identify horizontal and vertical sides
        # Assuming corners are ordered, opposite sides should be similar
        horizontal_1 = side_lengths[0]  # Top side
        horizontal_2 = side_lengths[2]  # Bottom side
        vertical_1 = side_lengths[1]    # Right side
        vertical_2 = side_lengths[3]    # Left side
        
        # Average the opposite sides
        avg_horizontal = (horizontal_1 + horizontal_2) / 2
        avg_vertical = (vertical_1 + vertical_2) / 2
        
        print(f"\nAverage horizontal length: {avg_horizontal:.2f} pixels")
        print(f"Average vertical length: {avg_vertical:.2f} pixels")
        
        # Calculate aspect ratio
        # aspect_ratio = fx/fy = (horizontal_pixels/physical_size) / (vertical_pixels/physical_size)
        aspect_ratio = avg_horizontal / avg_vertical
        
        print(f"\nPhysical square size: {physical_size_mm} mm")
        print(f"Measured aspect ratio (fx/fy): {aspect_ratio:.6f}")
        
        # Calculate deviation from square pixels
        deviation_percent = abs(1.0 - aspect_ratio) * 100
        print(f"Deviation from square pixels: {deviation_percent:.2f}%")
        
        if abs(aspect_ratio - 1.0) < 0.01:
            print("Pixels are approximately SQUARE")
        else:
            if aspect_ratio > 1.0:
                print("Pixels are WIDER than tall")
            else:
                print("Pixels are TALLER than wide")
        
        # Calculate scale factors (pixels per mm)
        scale_horizontal = avg_horizontal / physical_size_mm
        scale_vertical = avg_vertical / physical_size_mm
        
        print(f"\nScale factor horizontal: {scale_horizontal:.4f} pixels/mm")
        print(f"Scale factor vertical: {scale_vertical:.4f} pixels/mm")
        
        results = {
            'aspect_ratio': float(aspect_ratio),
            'horizontal_pixels': float(avg_horizontal),
            'vertical_pixels': float(avg_vertical),
            'physical_size_mm': float(physical_size_mm),
            'scale_horizontal': float(scale_horizontal),
            'scale_vertical': float(scale_vertical),
            'deviation_percent': float(deviation_percent),
            'corners': [[float(x), float(y)] for x, y in self.corners]
        }
        
        return results
    
    def save_results(self, results, output_dir, comparison_data=None):
        """
        Save aspect ratio results and comparison with automated calibration.
        
        Parameters
        ----------
        results : dict
            Aspect ratio calculation results
        output_dir : str
            Directory to save results
        comparison_data : dict, optional
            Automated calibration data for comparison
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numerical results
        results_path = os.path.join(output_dir, 'aspect_ratio_manual.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")
        
        # Save marked image
        image_path = os.path.join(output_dir, 'aspect_ratio_measurement.png')
        cv2.imwrite(image_path, self.display_image)
        print(f"Marked image saved: {image_path}")
        
        # Save detailed report
        report_path = os.path.join(output_dir, 'aspect_ratio_report.txt')
        with open(report_path, 'w') as f:
            f.write("MANUAL CALIBRATION: ASPECT RATIO ESTIMATION\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("METHOD\n")
            f.write("-" * 70 + "\n")
            f.write("Manual measurement of a known square object\n")
            f.write(f"Image: {os.path.basename(self.image_path)}\n")
            f.write(f"Physical square size: {results['physical_size_mm']} mm\n\n")
            
            f.write("MEASUREMENTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Horizontal length: {results['horizontal_pixels']:.2f} pixels\n")
            f.write(f"Vertical length: {results['vertical_pixels']:.2f} pixels\n")
            f.write(f"Scale horizontal: {results['scale_horizontal']:.4f} pixels/mm\n")
            f.write(f"Scale vertical: {results['scale_vertical']:.4f} pixels/mm\n\n")
            
            f.write("RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Aspect Ratio (fx/fy): {results['aspect_ratio']:.6f}\n")
            f.write(f"Deviation from unity: {results['deviation_percent']:.2f}%\n\n")
            
            if comparison_data:
                f.write("COMPARISON WITH AUTOMATED CALIBRATION\n")
                f.write("-" * 70 + "\n")
                auto_aspect = comparison_data.get('fx', 0) / comparison_data.get('fy', 1)
                f.write(f"Automated aspect ratio: {auto_aspect:.6f}\n")
                f.write(f"Manual aspect ratio: {results['aspect_ratio']:.6f}\n")
                diff = abs(auto_aspect - results['aspect_ratio'])
                diff_percent = (diff / auto_aspect) * 100
                f.write(f"Absolute difference: {diff:.6f}\n")
                f.write(f"Relative difference: {diff_percent:.2f}%\n\n")
                
                if diff_percent < 5:
                    f.write("Agreement: EXCELLENT (< 5% difference)\n")
                elif diff_percent < 10:
                    f.write("Agreement: GOOD (< 10% difference)\n")
                else:
                    f.write("Agreement: FAIR (> 10% difference)\n")
        
        print(f"Report saved: {report_path}")


def load_automated_calibration():
    """
    Load automated calibration results for comparison.
    
    Returns
    -------
    dict or None
        Camera matrix data if available
    """
    try:
        camera_matrix_path = os.path.join(PathConfig.RESULTS_DIR, 
                                         PathConfig.CAMERA_MATRIX_FILE)
        with open(camera_matrix_path, 'r') as f:
            return json.load(f)
    except:
        return None


def main():
    """
    Main entry point for aspect ratio estimation.
    """
    print("=" * 70)
    print("MANUAL CALIBRATION: ASPECT RATIO ESTIMATION")
    print("=" * 70)
    
    # Get image path from user
    print("\nEnter the path to your image with a square object:")
    print("(Press Enter to use default: data/manual_experiments/square_image.png)")
    image_path = input("Image path: ").strip()
    
    if not image_path:
        image_path = os.path.join(PathConfig.MANUAL_EXPERIMENTS_DIR, "square_image.png")
    
    if not os.path.exists(image_path):
        print(f"\nERROR: Image not found: {image_path}")
        print("Please provide a valid image path.")
        return
    
    # Get physical size of square
    print("\nEnter the physical size of your square object (in millimeters):")
    print("(e.g., if you're using a 30mm square from the checkerboard, enter 30)")
    physical_size = input("Size (mm): ").strip()
    
    try:
        physical_size = float(physical_size)
    except ValueError:
        print("ERROR: Invalid size. Please enter a number.")
        return
    
    # Create estimator and process
    estimator = AspectRatioEstimator()
    estimator.load_image(image_path)
    
    # Interactive corner selection
    success = estimator.select_corners_interactive()
    
    if not success:
        print("\nCalibration cancelled.")
        return
    
    # Calculate aspect ratio
    results = estimator.calculate_aspect_ratio(physical_size)
    
    # Load automated calibration for comparison
    auto_calib = load_automated_calibration()
    if auto_calib:
        print("\n" + "=" * 70)
        print("COMPARISON WITH AUTOMATED CALIBRATION")
        print("=" * 70)
        auto_aspect = auto_calib['fx'] / auto_calib['fy']
        print(f"Automated aspect ratio (fx/fy): {auto_aspect:.6f}")
        print(f"Manual aspect ratio: {results['aspect_ratio']:.6f}")
        diff = abs(auto_aspect - results['aspect_ratio'])
        diff_percent = (diff / auto_aspect) * 100
        print(f"Difference: {diff:.6f} ({diff_percent:.2f}%)")
    
    # Save results
    output_dir = os.path.join(PathConfig.RESULTS_DIR, "manual_calibration")
    estimator.save_results(results, output_dir, auto_calib)
    
    print("\n" + "=" * 70)
    print("ASPECT RATIO ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()