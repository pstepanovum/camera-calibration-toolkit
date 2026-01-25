"""
Manual Calibration: Focal Length Estimation

This script estimates focal length using the projection equation:
f = (image_size × distance) / object_size

Theory:
- The projection equation relates object size, image size, distance, and focal length
- By measuring a known object at a known distance, we can solve for focal length
- We measure both horizontal and vertical to get fx and fy independently

Usage:
1. Take a photo of a known object (e.g., ruler, checkerboard) at known distance
2. Run this script and measure the object in the image
3. Script calculates fx and fy from measurements
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


class FocalLengthEstimator:
    """
    Estimates focal length using manual measurements and projection equation.
    """
    
    def __init__(self):
        self.image = None
        self.image_path = None
        self.points = []
        self.display_image = None
        
    def load_image(self, image_path):
        """
        Load the image containing the object.
        
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
        Handle mouse clicks to record measurement points.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                
                # Draw the point
                cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.display_image, str(len(self.points)), 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
                
                # Draw horizontal line after 2 points
                if len(self.points) == 2:
                    cv2.line(self.display_image, 
                            self.points[0], self.points[1], 
                            (0, 255, 0), 2)
                    cv2.putText(self.display_image, "Horizontal", 
                               ((self.points[0][0] + self.points[1][0])//2, 
                                (self.points[0][1] + self.points[1][1])//2 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print("Horizontal measurement complete. Now click for vertical measurement.")
                
                # Draw vertical line after 4 points
                if len(self.points) == 4:
                    cv2.line(self.display_image, 
                            self.points[2], self.points[3], 
                            (255, 0, 0), 2)
                    cv2.putText(self.display_image, "Vertical", 
                               ((self.points[2][0] + self.points[3][0])//2 + 10, 
                                (self.points[2][1] + self.points[3][1])//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    print("\nAll measurements complete! Press any key to continue...")
                
                cv2.imshow('Measure Object', self.display_image)
    
    def select_measurements_interactive(self):
        """
        Interactive GUI for measuring the object.
        
        Instructions:
        - Click two points for HORIZONTAL measurement
        - Then click two points for VERTICAL measurement
        """
        print("\n" + "=" * 70)
        print("INTERACTIVE OBJECT MEASUREMENT")
        print("=" * 70)
        print("\nInstructions:")
        print("  1. First, measure HORIZONTAL dimension:")
        print("     Click LEFT edge, then RIGHT edge of your object")
        print("  2. Then, measure VERTICAL dimension:")
        print("     Click TOP edge, then BOTTOM edge of your object")
        print("  3. Press 'r' to RESET if you make a mistake")
        print("  4. Press any other key when all 4 points are marked")
        print("\nWindow will open now...")
        
        cv2.namedWindow('Measure Object')
        cv2.setMouseCallback('Measure Object', self.mouse_callback)
        
        while True:
            cv2.imshow('Measure Object', self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            # Press 'r' to reset
            if key == ord('r'):
                print("\nResetting measurements...")
                self.points = []
                self.display_image = self.image.copy()
                cv2.imshow('Measure Object', self.display_image)
            
            # Press any other key to finish (after 4 points)
            elif key != 255 and len(self.points) == 4:
                break
            
            # Press 'q' to quit early
            elif key == ord('q'):
                print("\nQuitting...")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def calculate_focal_length(self, physical_width_mm, physical_height_mm, distance_mm):
        """
        Calculate focal length using projection equation.
        
        Parameters
        ----------
        physical_width_mm : float
            Physical width of the object in millimeters
        physical_height_mm : float
            Physical height of the object in millimeters
        distance_mm : float
            Distance from camera to object in millimeters
            
        Returns
        -------
        dict
            Dictionary containing focal length estimates
        """
        if len(self.points) != 4:
            raise ValueError("Need exactly 4 points (2 for horizontal, 2 for vertical)")
        
        print("\n" + "=" * 70)
        print("CALCULATING FOCAL LENGTH")
        print("=" * 70)
        
        # Calculate horizontal measurement (first 2 points)
        p1, p2 = np.array(self.points[0]), np.array(self.points[1])
        horizontal_pixels = np.linalg.norm(p2 - p1)
        
        # Calculate vertical measurement (last 2 points)
        p3, p4 = np.array(self.points[2]), np.array(self.points[3])
        vertical_pixels = np.linalg.norm(p4 - p3)
        
        print(f"\nMeasurements in image:")
        print(f"  Horizontal: {horizontal_pixels:.2f} pixels")
        print(f"  Vertical: {vertical_pixels:.2f} pixels")
        
        print(f"\nPhysical dimensions:")
        print(f"  Width: {physical_width_mm} mm")
        print(f"  Height: {physical_height_mm} mm")
        print(f"  Distance to camera: {distance_mm} mm")
        
        # Apply projection equation: f = (image_size × distance) / object_size
        fx_estimated = (horizontal_pixels * distance_mm) / physical_width_mm
        fy_estimated = (vertical_pixels * distance_mm) / physical_height_mm
        
        print(f"\nEstimated focal lengths:")
        print(f"  fx (horizontal): {fx_estimated:.2f} pixels")
        print(f"  fy (vertical): {fy_estimated:.2f} pixels")
        
        # Calculate aspect ratio
        aspect_ratio = fx_estimated / fy_estimated
        print(f"  Aspect ratio (fx/fy): {aspect_ratio:.6f}")
        
        results = {
            'fx_estimated': float(fx_estimated),
            'fy_estimated': float(fy_estimated),
            'aspect_ratio': float(aspect_ratio),
            'horizontal_pixels': float(horizontal_pixels),
            'vertical_pixels': float(vertical_pixels),
            'physical_width_mm': float(physical_width_mm),
            'physical_height_mm': float(physical_height_mm),
            'distance_mm': float(distance_mm),
            'measurement_points': [[float(x), float(y)] for x, y in self.points]
        }
        
        return results
    
    def save_results(self, results, output_dir, comparison_data=None):
        """
        Save focal length results and comparison.
        
        Parameters
        ----------
        results : dict
            Focal length calculation results
        output_dir : str
            Directory to save results
        comparison_data : dict, optional
            Automated calibration data for comparison
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numerical results
        results_path = os.path.join(output_dir, 'focal_length_manual.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")
        
        # Save marked image
        image_path = os.path.join(output_dir, 'focal_length_measurement.png')
        cv2.imwrite(image_path, self.display_image)
        print(f"Marked image saved: {image_path}")
        
        # Save detailed report
        report_path = os.path.join(output_dir, 'focal_length_report.txt')
        with open(report_path, 'w') as f:
            f.write("MANUAL CALIBRATION: FOCAL LENGTH ESTIMATION\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("METHOD\n")
            f.write("-" * 70 + "\n")
            f.write("Projection equation: f = (image_size × distance) / object_size\n")
            f.write(f"Image: {os.path.basename(self.image_path)}\n\n")
            
            f.write("MEASUREMENTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Physical width: {results['physical_width_mm']} mm\n")
            f.write(f"Physical height: {results['physical_height_mm']} mm\n")
            f.write(f"Distance to camera: {results['distance_mm']} mm\n")
            f.write(f"Horizontal in image: {results['horizontal_pixels']:.2f} pixels\n")
            f.write(f"Vertical in image: {results['vertical_pixels']:.2f} pixels\n\n")
            
            f.write("RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"fx (horizontal focal length): {results['fx_estimated']:.2f} pixels\n")
            f.write(f"fy (vertical focal length): {results['fy_estimated']:.2f} pixels\n")
            f.write(f"Aspect ratio (fx/fy): {results['aspect_ratio']:.6f}\n\n")
            
            if comparison_data:
                f.write("COMPARISON WITH AUTOMATED CALIBRATION\n")
                f.write("-" * 70 + "\n")
                fx_auto = comparison_data.get('fx', 0)
                fy_auto = comparison_data.get('fy', 0)
                
                f.write(f"Automated fx: {fx_auto:.2f} pixels\n")
                f.write(f"Manual fx: {results['fx_estimated']:.2f} pixels\n")
                fx_diff = abs(fx_auto - results['fx_estimated'])
                fx_diff_percent = (fx_diff / fx_auto) * 100
                f.write(f"fx difference: {fx_diff:.2f} pixels ({fx_diff_percent:.2f}%)\n\n")
                
                f.write(f"Automated fy: {fy_auto:.2f} pixels\n")
                f.write(f"Manual fy: {results['fy_estimated']:.2f} pixels\n")
                fy_diff = abs(fy_auto - results['fy_estimated'])
                fy_diff_percent = (fy_diff / fy_auto) * 100
                f.write(f"fy difference: {fy_diff:.2f} pixels ({fy_diff_percent:.2f}%)\n\n")
                
                avg_diff_percent = (fx_diff_percent + fy_diff_percent) / 2
                if avg_diff_percent < 5:
                    f.write("Agreement: EXCELLENT (< 5% difference)\n")
                elif avg_diff_percent < 10:
                    f.write("Agreement: GOOD (< 10% difference)\n")
                elif avg_diff_percent < 20:
                    f.write("Agreement: FAIR (< 20% difference)\n")
                else:
                    f.write("Agreement: POOR (> 20% difference)\n")
                    f.write("\nPossible causes of large difference:\n")
                    f.write("- Inaccurate distance measurement\n")
                    f.write("- Measurement errors in image\n")
                    f.write("- Object size measurement error\n")
                    f.write("- Lens distortion effects\n")
        
        print(f"Report saved: {report_path}")


def load_automated_calibration():
    """
    Load automated calibration results for comparison.
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
    Main entry point for focal length estimation.
    """
    print("=" * 70)
    print("MANUAL CALIBRATION: FOCAL LENGTH ESTIMATION")
    print("=" * 70)
    
    # Get image path
    print("\nEnter the path to your image with a known object:")
    print("(Press Enter to use default: data/manual_experiments/focal_length_image.jpg)")
    image_path = input("Image path: ").strip()
    
    if not image_path:
        image_path = os.path.join(PathConfig.MANUAL_EXPERIMENTS_DIR, 
                                 "focal_length_image.jpg")
    
    if not os.path.exists(image_path):
        print(f"\nERROR: Image not found: {image_path}")
        print("Please provide a valid image path.")
        return
    
    # Get physical dimensions
    print("\nEnter the PHYSICAL WIDTH of your object (in millimeters):")
    print("(e.g., if measuring a ruler across 100mm, enter 100)")
    width_str = input("Width (mm): ").strip()
    
    print("\nEnter the PHYSICAL HEIGHT of your object (in millimeters):")
    height_str = input("Height (mm): ").strip()
    
    print("\nEnter the DISTANCE from camera to object (in millimeters):")
    print("(Measure from camera lens to the object - be as accurate as possible!)")
    distance_str = input("Distance (mm): ").strip()
    
    try:
        width = float(width_str)
        height = float(height_str)
        distance = float(distance_str)
    except ValueError:
        print("ERROR: Invalid input. Please enter numbers.")
        return
    
    # Create estimator and process
    estimator = FocalLengthEstimator()
    estimator.load_image(image_path)
    
    # Interactive measurement
    success = estimator.select_measurements_interactive()
    
    if not success:
        print("\nCalibration cancelled.")
        return
    
    # Calculate focal length
    results = estimator.calculate_focal_length(width, height, distance)
    
    # Load automated calibration for comparison
    auto_calib = load_automated_calibration()
    if auto_calib:
        print("\n" + "=" * 70)
        print("COMPARISON WITH AUTOMATED CALIBRATION")
        print("=" * 70)
        fx_auto = auto_calib['fx']
        fy_auto = auto_calib['fy']
        print(f"Automated fx: {fx_auto:.2f} pixels")
        print(f"Manual fx: {results['fx_estimated']:.2f} pixels")
        fx_diff_percent = abs(fx_auto - results['fx_estimated']) / fx_auto * 100
        print(f"fx difference: {fx_diff_percent:.2f}%")
        
        print(f"\nAutomated fy: {fy_auto:.2f} pixels")
        print(f"Manual fy: {results['fy_estimated']:.2f} pixels")
        fy_diff_percent = abs(fy_auto - results['fy_estimated']) / fy_auto * 100
        print(f"fy difference: {fy_diff_percent:.2f}%")
    
    # Save results
    output_dir = os.path.join(PathConfig.RESULTS_DIR, "manual_calibration")
    estimator.save_results(results, output_dir, auto_calib)
    
    print("\n" + "=" * 70)
    print("FOCAL LENGTH ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()