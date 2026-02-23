"""
Stereo Camera Calibration

This module performs stereo calibration to find the geometric relationship
between two cameras (rotation R and translation T between cameras).

Workflow:
1. Load paired checkerboard images from left and right cameras
2. Detect corners in both image sets
3. Perform stereo calibration using cv2.stereoCalibrate
4. Compute fundamental and essential matrices
5. Save calibration parameters for later use

Output:
- R: Rotation matrix between cameras
- T: Translation vector between cameras (baseline direction)
- E: Essential matrix
- F: Fundamental matrix
- Refined camera matrices and distortion coefficients
"""

import re

import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))  # Go up to src/ directory
from config import CheckerboardConfig, CalibrationConfig, PathConfig, StereoConfig


class StereoCalibrator:
    """
    Handles stereo camera calibration using paired checkerboard images.
    """
    
    def __init__(self, left_images_dir, right_images_dir, pattern_size=None, square_size=None):
        """
        Initialize the stereo calibrator.
        
        Parameters
        ----------
        left_images_dir : str
            Directory containing left camera checkerboard images
        right_images_dir : str
            Directory containing right camera checkerboard images
        pattern_size : tuple, optional
            (width, height) of internal corners
        square_size : float, optional
            Physical size of each square in mm
        """
        self.left_images_dir = left_images_dir
        self.right_images_dir = right_images_dir
        self.pattern_size = pattern_size or CheckerboardConfig.PATTERN_SIZE
        self.square_size = square_size or CheckerboardConfig.SQUARE_SIZE
        
        # Storage for calibration data
        self.left_image_paths = []
        self.right_image_paths = []
        self.object_points = []  # 3D points in real world space (same for both cameras)
        self.left_image_points = []   # 2D points in left camera
        self.right_image_points = []  # 2D points in right camera
        self.image_size = None
        
        # Stereo calibration results
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        self.reprojection_error = None
        
        print(f"Stereo Calibrator initialized:")
        print(f"  Pattern size: {self.pattern_size[0]}x{self.pattern_size[1]} corners")
        print(f"  Square size: {self.square_size} mm")
        
    def load_image_pairs(self):
        """
        Load paired images from left and right directories.
        
        Returns
        -------
        int
            Number of image pairs loaded
        """
        print(f"\nLoading image pairs...")
        print(f"  Left dir:  {self.left_images_dir}")
        print(f"  Right dir: {self.right_images_dir}")
        
        # Get all image files from both directories
        left_images = []
        right_images = []
        
        for ext in CalibrationConfig.IMAGE_EXTENSIONS:
            left_images.extend(glob.glob(os.path.join(self.left_images_dir, f"*{ext}")))
            right_images.extend(glob.glob(os.path.join(self.right_images_dir, f"*{ext}")))
        
        left_images.sort()
        right_images.sort()
        
        # Match pairs by filename
        left_basenames = [os.path.basename(p) for p in left_images]
        right_basenames = [os.path.basename(p) for p in right_images]
        
        for left_path, left_name in zip(left_images, left_basenames):
            name_no_ext, ext = os.path.splitext(left_name)
            
            # Replace leading 'L' or 'left' (case-insensitive) before the number
            right_stem = re.sub(r'^L(?=\d)', 'R', name_no_ext)
            right_stem = re.sub(r'^[Ll]eft', 'right', right_stem)
            
            # Try both same-case and original extension
            for candidate_ext in [ext, ext.lower(), ext.upper()]:
                right_name = right_stem + candidate_ext
                if right_name in right_basenames:
                    right_path = right_images[right_basenames.index(right_name)]
                    self.left_image_paths.append(left_path)
                    self.right_image_paths.append(right_path)
                    break
            else:
                print(f"  WARNING: No matching right image for {left_name}")
        
        if len(self.left_image_paths) == 0:
            raise ValueError("No matching image pairs found!")
        
        print(f"\nFound {len(self.left_image_paths)} image pairs")
        
        if len(self.left_image_paths) < CalibrationConfig.MIN_IMAGES:
            print(f"WARNING: Less than {CalibrationConfig.MIN_IMAGES} pairs. "
                  f"Calibration may be less accurate.")
        
        return len(self.left_image_paths)
    
    def prepare_object_points(self):
        """
        Prepare 3D object points for the checkerboard pattern.
        
        Returns
        -------
        ndarray
            Array of 3D object points
        """
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 
                                0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def detect_corners_stereo(self, visualize=True, output_dir=None):
        """
        Detect checkerboard corners in paired left and right images.
        
        Parameters
        ----------
        visualize : bool
            Whether to save images with detected corners
        output_dir : str, optional
            Directory to save visualization images
            
        Returns
        -------
        int
            Number of image pairs with successfully detected corners
        """
        print("\nDetecting checkerboard corners in stereo pairs...")
        
        if visualize and output_dir:
            os.makedirs(os.path.join(output_dir, 'left'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'right'), exist_ok=True)
        
        # Prepare object points template
        objp = self.prepare_object_points()
        
        # Termination criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        successful_pairs = 0
        
        for i, (left_path, right_path) in enumerate(zip(self.left_image_paths, 
                                                         self.right_image_paths)):
            print(f"\nProcessing pair {i+1}/{len(self.left_image_paths)}:")
            print(f"  Left:  {os.path.basename(left_path)}")
            print(f"  Right: {os.path.basename(right_path)}")
            
            # Read left image
            img_left = cv2.imread(left_path)
            if img_left is None:
                print(f"  ERROR: Could not read left image")
                continue
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            
            # Read right image
            img_right = cv2.imread(right_path)
            if img_right is None:
                print(f"  ERROR: Could not read right image")
                continue
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            
            # Store image size
            if self.image_size is None:
                self.image_size = gray_left.shape[::-1]
            
            # Find corners in left image
            ret_left, corners_left = cv2.findChessboardCorners(
                gray_left, 
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Find corners in right image
            ret_right, corners_right = cv2.findChessboardCorners(
                gray_right, 
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Only accept if both succeeded
            if ret_left and ret_right:
                # Refine corner locations
                corners_left_refined = cv2.cornerSubPix(
                    gray_left, corners_left, (11, 11), (-1, -1), criteria
                )
                corners_right_refined = cv2.cornerSubPix(
                    gray_right, corners_right, (11, 11), (-1, -1), criteria
                )
                
                # Store the points
                self.object_points.append(objp)
                self.left_image_points.append(corners_left_refined)
                self.right_image_points.append(corners_right_refined)
                
                successful_pairs += 1
                print(f"  ✓ SUCCESS: Detected corners in both images")
                
                # Visualize if requested
                if visualize and output_dir:
                    # Left image
                    img_left_vis = img_left.copy()
                    cv2.drawChessboardCorners(img_left_vis, self.pattern_size, 
                                             corners_left_refined, ret_left)
                    left_output = os.path.join(output_dir, 'left',
                                              f"corners_{i:02d}_{os.path.basename(left_path)}")
                    cv2.imwrite(left_output, img_left_vis)
                    
                    # Right image
                    img_right_vis = img_right.copy()
                    cv2.drawChessboardCorners(img_right_vis, self.pattern_size, 
                                             corners_right_refined, ret_right)
                    right_output = os.path.join(output_dir, 'right',
                                               f"corners_{i:02d}_{os.path.basename(right_path)}")
                    cv2.imwrite(right_output, img_right_vis)
            else:
                if not ret_left:
                    print(f"  ✗ FAILED: Could not detect corners in LEFT image")
                if not ret_right:
                    print(f"  ✗ FAILED: Could not detect corners in RIGHT image")
        
        print(f"\n{'='*70}")
        print(f"Corner detection complete:")
        print(f"  Successful pairs: {successful_pairs}/{len(self.left_image_paths)}")
        print(f"  Failed pairs: {len(self.left_image_paths) - successful_pairs}")
        print(f"{'='*70}")
        
        if successful_pairs < CalibrationConfig.MIN_IMAGES:
            raise ValueError(
                f"Only {successful_pairs} successful pairs. "
                f"Need at least {CalibrationConfig.MIN_IMAGES}."
            )
        
        return successful_pairs
    
    def calibrate_stereo(self):
        """
        Perform stereo calibration using detected corners.
        
        Computes:
        - Individual camera intrinsics (K_left, K_right, dist_left, dist_right)
        - Rotation matrix R between cameras
        - Translation vector T between cameras (baseline)
        - Essential matrix E and Fundamental matrix F
        
        Returns
        -------
        float
            Mean reprojection error in pixels
        """
        print("\n" + "=" * 70)
        print("COMPUTING STEREO CALIBRATION")
        print("=" * 70)
        
        if len(self.object_points) == 0:
            raise ValueError("No corner points detected. Run detect_corners_stereo() first.")
        
        print(f"\nCalibrating with {len(self.object_points)} image pairs...")
        print(f"Image size: {self.image_size[0]} x {self.image_size[1]} pixels")
        
        # Perform stereo calibration
        # This will refine the camera matrices and compute R, T
        flags = StereoConfig.STEREO_CALIB_FLAGS
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            self.object_points,
            self.left_image_points,
            self.right_image_points,
            None, None,  # Will compute camera matrices
            None, None,  # Will compute distortion coefficients
            self.image_size,
            criteria=criteria,
            flags=flags
        )
        
        # Store results
        self.camera_matrix_left = K_left
        self.dist_coeffs_left = dist_left
        self.camera_matrix_right = K_right
        self.dist_coeffs_right = dist_right
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        self.reprojection_error = ret
        
        # Calculate baseline (magnitude of translation vector)
        baseline = np.linalg.norm(T)
        
        # Print results
        print("\n" + "=" * 70)
        print("STEREO CALIBRATION RESULTS")
        print("=" * 70)
        
        print("\nLeft Camera Matrix:")
        print(f"  fx: {K_left[0, 0]:.2f} pixels")
        print(f"  fy: {K_left[1, 1]:.2f} pixels")
        print(f"  cx: {K_left[0, 2]:.2f} pixels")
        print(f"  cy: {K_left[1, 2]:.2f} pixels")
        
        print("\nRight Camera Matrix:")
        print(f"  fx: {K_right[0, 0]:.2f} pixels")
        print(f"  fy: {K_right[1, 1]:.2f} pixels")
        print(f"  cx: {K_right[0, 2]:.2f} pixels")
        print(f"  cy: {K_right[1, 2]:.2f} pixels")
        
        print(f"\nBaseline (distance between cameras): {baseline:.2f} mm")
        print(f"\nTranslation Vector T:")
        print(f"  Tx: {T[0, 0]:.2f} mm")
        print(f"  Ty: {T[1, 0]:.2f} mm")
        print(f"  Tz: {T[2, 0]:.2f} mm")
        
        print(f"\nRotation between cameras (Rodrigues):")
        angles = cv2.Rodrigues(R)[0].flatten() * 180 / np.pi
        print(f"  Rx: {angles[0]:.2f}°")
        print(f"  Ry: {angles[1]:.2f}°")
        print(f"  Rz: {angles[2]:.2f}°")
        
        print(f"\nReprojection Error: {ret:.4f} pixels")
        
        print("\n" + "=" * 70)
        
        return ret
    
    def save_calibration(self, output_dir):
        """
        Save stereo calibration results to JSON files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save calibration results
        """
        if self.R is None:
            raise ValueError("Stereo calibration not performed. Run calibrate_stereo() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving stereo calibration results to: {output_dir}")
        
        # Calculate baseline
        baseline = float(np.linalg.norm(self.T))
        
        # Save complete stereo parameters
        stereo_params = {
            'left_camera': {
                'fx': float(self.camera_matrix_left[0, 0]),
                'fy': float(self.camera_matrix_left[1, 1]),
                'cx': float(self.camera_matrix_left[0, 2]),
                'cy': float(self.camera_matrix_left[1, 2]),
                'k1': float(self.dist_coeffs_left[0, 0]),
                'k2': float(self.dist_coeffs_left[0, 1]),
                'p1': float(self.dist_coeffs_left[0, 2]),
                'p2': float(self.dist_coeffs_left[0, 3]),
                'k3': float(self.dist_coeffs_left[0, 4]),
                'camera_matrix': self.camera_matrix_left.tolist(),
                'distortion_coefficients': self.dist_coeffs_left.tolist()
            },
            'right_camera': {
                'fx': float(self.camera_matrix_right[0, 0]),
                'fy': float(self.camera_matrix_right[1, 1]),
                'cx': float(self.camera_matrix_right[0, 2]),
                'cy': float(self.camera_matrix_right[1, 2]),
                'k1': float(self.dist_coeffs_right[0, 0]),
                'k2': float(self.dist_coeffs_right[0, 1]),
                'p1': float(self.dist_coeffs_right[0, 2]),
                'p2': float(self.dist_coeffs_right[0, 3]),
                'k3': float(self.dist_coeffs_right[0, 4]),
                'camera_matrix': self.camera_matrix_right.tolist(),
                'distortion_coefficients': self.dist_coeffs_right.tolist()
            },
            'stereo_parameters': {
                'rotation_matrix': self.R.tolist(),
                'translation_vector': self.T.tolist(),
                'essential_matrix': self.E.tolist(),
                'fundamental_matrix': self.F.tolist(),
                'baseline_mm': baseline,
                'reprojection_error': float(self.reprojection_error)
            },
            'image_size': {
                'width': int(self.image_size[0]),
                'height': int(self.image_size[1])
            }
        }
        
        stereo_params_path = os.path.join(output_dir, PathConfig.STEREO_PARAMS_FILE)
        with open(stereo_params_path, 'w') as f:
            json.dump(stereo_params, f, indent=2)
        print(f"  ✓ Stereo parameters saved: {stereo_params_path}")
        
        # Save detailed report
        report_path = os.path.join(output_dir, PathConfig.STEREO_REPORT_FILE)
        with open(report_path, 'w') as f:
            f.write("STEREO CAMERA CALIBRATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Pattern size: {self.pattern_size[0]} x {self.pattern_size[1]} corners\n")
            f.write(f"Square size: {self.square_size} mm\n")
            f.write(f"Number of image pairs: {len(self.object_points)}\n")
            f.write(f"Image resolution: {self.image_size[0]} x {self.image_size[1]} pixels\n\n")
            
            f.write("LEFT CAMERA INTRINSICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"fx: {self.camera_matrix_left[0, 0]:.6f} pixels\n")
            f.write(f"fy: {self.camera_matrix_left[1, 1]:.6f} pixels\n")
            f.write(f"cx: {self.camera_matrix_left[0, 2]:.6f} pixels\n")
            f.write(f"cy: {self.camera_matrix_left[1, 2]:.6f} pixels\n\n")
            
            f.write("RIGHT CAMERA INTRINSICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"fx: {self.camera_matrix_right[0, 0]:.6f} pixels\n")
            f.write(f"fy: {self.camera_matrix_right[1, 1]:.6f} pixels\n")
            f.write(f"cx: {self.camera_matrix_right[0, 2]:.6f} pixels\n")
            f.write(f"cy: {self.camera_matrix_right[1, 2]:.6f} pixels\n\n")
            
            f.write("STEREO GEOMETRY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Baseline: {baseline:.6f} mm\n")
            f.write(f"Translation vector T:\n")
            f.write(f"  Tx: {self.T[0, 0]:.6f} mm\n")
            f.write(f"  Ty: {self.T[1, 0]:.6f} mm\n")
            f.write(f"  Tz: {self.T[2, 0]:.6f} mm\n\n")
            
            angles = cv2.Rodrigues(self.R)[0].flatten() * 180 / np.pi
            f.write(f"Rotation angles:\n")
            f.write(f"  Rx: {angles[0]:.6f} degrees\n")
            f.write(f"  Ry: {angles[1]:.6f} degrees\n")
            f.write(f"  Rz: {angles[2]:.6f} degrees\n\n")
            
            f.write(f"Mean reprojection error: {self.reprojection_error:.6f} pixels\n")
        
        print(f"  ✓ Calibration report saved: {report_path}")
        print("\n✓ Stereo calibration results saved successfully!")


def main():
    """
    Main entry point for stereo calibration.
    """
    print("=" * 70)
    print("STEREO CAMERA CALIBRATION")
    print("=" * 70)
    
    # Use stereo calibration paths
    left_calibration_dir = os.path.join(PathConfig.STEREO_LEFT_DIR, 'calibration_left')
    right_calibration_dir = os.path.join(PathConfig.STEREO_RIGHT_DIR, 'calibration_right')
    results_dir = PathConfig.STEREO_RESULTS_DIR
    
    # Check if directories exist
    if not os.path.exists(left_calibration_dir):
        print(f"\nERROR: Left calibration directory not found:")
        print(f"  {left_calibration_dir}")
        print("\nPlease ensure your stereo images are organized as:")
        print("  data/stereo_images/left/calibration_left/")
        print("  data/stereo_images/right/calibration_right/")
        return
    
    if not os.path.exists(right_calibration_dir):
        print(f"\nERROR: Right calibration directory not found:")
        print(f"  {right_calibration_dir}")
        return
    
    # Create calibrator instance
    calibrator = StereoCalibrator(left_calibration_dir, right_calibration_dir)
    
    # Step 1: Load image pairs
    calibrator.load_image_pairs()
    
    # Step 2: Detect corners in both cameras
    corners_output = os.path.join(results_dir, "detected_corners_stereo")
    successful = calibrator.detect_corners_stereo(
        visualize=True, 
        output_dir=corners_output
    )
    
    # Step 3: Perform stereo calibration
    calibrator.calibrate_stereo()
    
    # Step 4: Save results
    calibrator.save_calibration(results_dir)
    
    print("\n" + "=" * 70)
    print("STEREO CALIBRATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the stereo calibration results")
    print("  2. Run stereo_rectify.py to prepare images for depth estimation")
    print("=" * 70)


if __name__ == "__main__":
    main()