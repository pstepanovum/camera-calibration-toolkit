"""
Automated camera calibration using checkerboard pattern detection.

This module implements the Zhang calibration method using OpenCV.
"""

import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import CheckerboardConfig, CalibrationConfig, PathConfig
from utils.visualization import generate_all_visualizations


class CameraCalibrator:
    """
    Handles automated camera calibration using checkerboard patterns.
    """
    
    def __init__(self, images_dir, pattern_size=None, square_size=None):
        """
        Initialize the calibrator.
        
        Parameters
        ----------
        images_dir : str
            Directory containing checkerboard images
        pattern_size : tuple, optional
            (width, height) of internal corners
        square_size : float, optional
            Physical size of each square in mm
        """
        self.images_dir = images_dir
        self.pattern_size = pattern_size or CheckerboardConfig.PATTERN_SIZE
        self.square_size = square_size or CheckerboardConfig.SQUARE_SIZE
        
        # Storage for calibration data
        self.image_paths = []
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane
        self.image_size = None
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None
        
        print(f"Calibrator initialized:")
        print(f"  Pattern size: {self.pattern_size[0]}x{self.pattern_size[1]} corners")
        print(f"  Square size: {self.square_size} mm")
        
    def load_images(self):
        """
        Load all calibration images from the directory.
        
        Returns
        -------
        int
            Number of images loaded
        """
        print(f"\nLoading images from: {self.images_dir}")
        
        # Get all image files
        for ext in CalibrationConfig.IMAGE_EXTENSIONS:
            self.image_paths.extend(
                glob.glob(os.path.join(self.images_dir, f"*{ext}"))
            )
        
        self.image_paths.sort()  # Sort for consistent ordering
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        print(f"Found {len(self.image_paths)} images")
        
        if len(self.image_paths) < CalibrationConfig.MIN_IMAGES:
            print(f"WARNING: Less than {CalibrationConfig.MIN_IMAGES} images. "
                  f"Calibration may be less accurate.")
        
        return len(self.image_paths)
    
    def prepare_object_points(self):
        """
        Prepare 3D object points for the checkerboard pattern.
        
        Creates a grid of 3D coordinates where the checkerboard lies on the Z=0 plane.
        Each point represents an internal corner of the checkerboard.
        
        Returns
        -------
        ndarray
            Array of 3D object points, shape (num_corners, 3)
        """
        # Create coordinate grid
        # Example: for 10x6 pattern, creates points like:
        # (0,0,0), (30,0,0), (60,0,0), ..., (270,0,0),
        # (0,30,0), (30,30,0), ..., (270,150,0)
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 
                                0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        return objp
    
    def detect_corners(self, visualize=True, output_dir=None):
        """
        Detect checkerboard corners in all images.
        
        Parameters
        ----------
        visualize : bool
            Whether to save images with detected corners
        output_dir : str, optional
            Directory to save visualization images
            
        Returns
        -------
        int
            Number of images with successfully detected corners
        """
        print("\nDetecting checkerboard corners...")
        
        if visualize and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Prepare object points template
        objp = self.prepare_object_points()
        
        # Termination criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        successful_detections = 0
        
        for i, img_path in enumerate(self.image_paths):
            print(f"Processing image {i+1}/{len(self.image_paths)}: {os.path.basename(img_path)}")
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ERROR: Could not read image")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Store image size (should be same for all images)
            if self.image_size is None:
                self.image_size = gray.shape[::-1]  # (width, height)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Refine corner locations to sub-pixel accuracy
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                
                # Store the object points and image points
                self.object_points.append(objp)
                self.image_points.append(corners_refined)
                
                successful_detections += 1
                print(f"  SUCCESS: Detected all {len(corners_refined)} corners")
                
                # Visualize if requested
                if visualize and output_dir:
                    img_vis = img.copy()
                    cv2.drawChessboardCorners(img_vis, self.pattern_size, 
                                             corners_refined, ret)
                    output_path = os.path.join(
                        output_dir, 
                        f"corners_{i:02d}_{os.path.basename(img_path)}"
                    )
                    cv2.imwrite(output_path, img_vis)
            else:
                print(f"  FAILED: Could not detect corners")
        
        print(f"\nCorner detection complete:")
        print(f"  Successful: {successful_detections}/{len(self.image_paths)}")
        print(f"  Failed: {len(self.image_paths) - successful_detections}/{len(self.image_paths)}")
        
        if successful_detections < CalibrationConfig.MIN_IMAGES:
            raise ValueError(
                f"Only {successful_detections} images with detected corners. "
                f"Need at least {CalibrationConfig.MIN_IMAGES}."
            )
        
        return successful_detections
    
    def calibrate(self):
        """
        Perform camera calibration using detected corners.
        
        Uses cv2.calibrateCamera to compute:
        - Camera intrinsic matrix (fx, fy, cx, cy)
        - Distortion coefficients (k1, k2, p1, p2, k3)
        - Rotation and translation vectors for each image
        
        Returns
        -------
        float
            Mean reprojection error in pixels
        """
        print("\n" + "=" * 70)
        print("COMPUTING CAMERA CALIBRATION")
        print("=" * 70)
        
        if len(self.object_points) == 0:
            raise ValueError("No corner points detected. Run detect_corners() first.")
        
        print(f"\nCalibrating with {len(self.object_points)} images...")
        print(f"Image size: {self.image_size[0]} x {self.image_size[1]} pixels")
        
        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None
        )
        
        # Store results
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.reprojection_error = ret
        
        # Print results
        print("\nCamera Matrix:")
        print(f"  fx: {camera_matrix[0, 0]:.2f} pixels")
        print(f"  fy: {camera_matrix[1, 1]:.2f} pixels")
        print(f"  cx: {camera_matrix[0, 2]:.2f} pixels")
        print(f"  cy: {camera_matrix[1, 2]:.2f} pixels")
        
        print(f"\nReprojection Error: {ret:.4f} pixels")
        
        return ret
    
    def save_calibration(self, output_dir):
        """
        Save calibration results to JSON files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save calibration results
        """
        if self.camera_matrix is None:
            raise ValueError("Camera not calibrated. Run calibrate() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving calibration results to: {output_dir}")
        
        # Save camera matrix
        camera_matrix_dict = {
            'fx': float(self.camera_matrix[0, 0]),
            'fy': float(self.camera_matrix[1, 1]),
            'cx': float(self.camera_matrix[0, 2]),
            'cy': float(self.camera_matrix[1, 2]),
            'skew': float(self.camera_matrix[0, 1]),
            'matrix': self.camera_matrix.tolist()
        }
        
        camera_matrix_path = os.path.join(output_dir, PathConfig.CAMERA_MATRIX_FILE)
        with open(camera_matrix_path, 'w') as f:
            json.dump(camera_matrix_dict, f, indent=2)
        print(f"  Camera matrix saved: {camera_matrix_path}")
        
        # Save distortion coefficients
        dist_coeffs_dict = {
            'k1': float(self.dist_coeffs[0, 0]),
            'k2': float(self.dist_coeffs[0, 1]),
            'p1': float(self.dist_coeffs[0, 2]),
            'p2': float(self.dist_coeffs[0, 3]),
            'k3': float(self.dist_coeffs[0, 4]),
            'coefficients': self.dist_coeffs.tolist()
        }
        
        dist_coeffs_path = os.path.join(output_dir, PathConfig.DISTORTION_COEFFS_FILE)
        with open(dist_coeffs_path, 'w') as f:
            json.dump(dist_coeffs_dict, f, indent=2)
        print(f"  Distortion coefficients saved: {dist_coeffs_path}")
        
        # Save detailed report
        report_path = os.path.join(output_dir, PathConfig.CALIBRATION_REPORT_FILE)
        with open(report_path, 'w') as f:
            f.write("CAMERA CALIBRATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Pattern size: {self.pattern_size[0]} x {self.pattern_size[1]} corners\n")
            f.write(f"Square size: {self.square_size} mm\n")
            f.write(f"Number of images: {len(self.object_points)}\n")
            f.write(f"Image resolution: {self.image_size[0]} x {self.image_size[1]} pixels\n\n")
            
            f.write("INTRINSIC PARAMETERS\n")
            f.write("-" * 70 + "\n")
            f.write(f"fx: {self.camera_matrix[0, 0]:.6f} pixels\n")
            f.write(f"fy: {self.camera_matrix[1, 1]:.6f} pixels\n")
            f.write(f"cx: {self.camera_matrix[0, 2]:.6f} pixels\n")
            f.write(f"cy: {self.camera_matrix[1, 2]:.6f} pixels\n\n")
            
            f.write("DISTORTION COEFFICIENTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"k1: {self.dist_coeffs[0, 0]:.6f}\n")
            f.write(f"k2: {self.dist_coeffs[0, 1]:.6f}\n")
            f.write(f"p1: {self.dist_coeffs[0, 2]:.6f}\n")
            f.write(f"p2: {self.dist_coeffs[0, 3]:.6f}\n")
            f.write(f"k3: {self.dist_coeffs[0, 4]:.6f}\n\n")
            
            f.write(f"Mean reprojection error: {self.reprojection_error:.6f} pixels\n")
        
        print(f"  Calibration report saved: {report_path}")
        print("\nCalibration results saved successfully!")


def main():
    """
    Main entry point for automated calibration.
    """
    print("=" * 70)
    print("AUTOMATED CAMERA CALIBRATION")
    print("=" * 70)
    
    # TODO: Add argument parsing for command-line usage
    # For now, using default paths from config
    
    images_dir = PathConfig.CHECKERBOARD_IMAGES_DIR
    results_dir = PathConfig.RESULTS_DIR
    
    # Create calibrator instance
    calibrator = CameraCalibrator(images_dir)
    
    # Step 1: Load images
    calibrator.load_images()
    
    # Step 2: Detect corners
    corners_output = os.path.join(results_dir, "detected_corners")
    successful = calibrator.detect_corners(
        visualize=True, 
        output_dir=corners_output
    )
    
    # Step 3: Perform calibration
    calibrator.calibrate()
    
    # Step 4: Save results
    calibrator.save_calibration(results_dir)
    
    # Step 5: Generate visualizations
    generate_all_visualizations(calibrator, results_dir)
    
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()