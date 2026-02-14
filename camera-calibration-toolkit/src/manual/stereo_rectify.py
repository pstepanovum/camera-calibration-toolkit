"""
Stereo Rectification

This module performs stereo rectification to align left and right images
so that epipolar lines are horizontal and matching points are on the same row.

Theory:
- Rectification transforms both images to a common image plane
- After rectification, corresponding points have the same y-coordinate
- This simplifies stereo matching to 1D search along scanlines
- The Q matrix converts disparity to 3D coordinates

Workflow:
1. Load stereo calibration parameters (R, T, K_left, K_right)
2. Compute rectification transforms using cv2.stereoRectify
3. Generate remapping matrices for efficient rectification
4. Optionally rectify scene images for visualization
5. Save rectification parameters for depth estimation

Output:
- R1, R2: Rectification rotation matrices for left and right cameras
- P1, P2: Projection matrices in rectified coordinate system
- Q: Disparity-to-depth mapping matrix (4x4)
- Remapping matrices for fast image rectification
"""

import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PathConfig, CalibrationConfig


class StereoRectifier:
    """
    Handles stereo image rectification for depth estimation.
    """
    
    def __init__(self, stereo_params_path):
        """
        Initialize the stereo rectifier.
        
        Parameters
        ----------
        stereo_params_path : str
            Path to stereo calibration parameters JSON file
        """
        self.stereo_params_path = stereo_params_path
        
        # Load stereo calibration parameters
        print(f"Loading stereo calibration from: {stereo_params_path}")
        with open(stereo_params_path, 'r') as f:
            params = json.load(f)
        
        # Extract camera parameters
        self.K_left = np.array(params['left_camera']['camera_matrix'])
        self.dist_left = np.array(params['left_camera']['distortion_coefficients'])
        self.K_right = np.array(params['right_camera']['camera_matrix'])
        self.dist_right = np.array(params['right_camera']['distortion_coefficients'])
        
        # Extract stereo geometry
        self.R = np.array(params['stereo_parameters']['rotation_matrix'])
        self.T = np.array(params['stereo_parameters']['translation_vector'])
        self.baseline = params['stereo_parameters']['baseline_mm']
        
        # Image size
        self.image_size = (params['image_size']['width'], 
                          params['image_size']['height'])
        
        # Rectification results (to be computed)
        self.R1 = None  # Rectification rotation for left camera
        self.R2 = None  # Rectification rotation for right camera
        self.P1 = None  # Projection matrix for left camera
        self.P2 = None  # Projection matrix for right camera
        self.Q = None   # Disparity-to-depth mapping matrix
        self.roi_left = None   # Region of interest in rectified left image
        self.roi_right = None  # Region of interest in rectified right image
        
        # Remapping matrices for fast rectification
        self.map_left_x = None
        self.map_left_y = None
        self.map_right_x = None
        self.map_right_y = None
        
        print("Stereo parameters loaded successfully!")
        print(f"  Image size: {self.image_size[0]} x {self.image_size[1]}")
        print(f"  Baseline: {self.baseline:.2f} mm")
        
    def compute_rectification(self, alpha=1.0):
        """
        Compute rectification transforms.
        
        Parameters
        ----------
        alpha : float, optional
            Free scaling parameter (0.0 - 1.0)
            - 0.0: Rectified images contain only valid pixels
            - 1.0: All source image pixels are retained (some black borders)
            - 0.5-0.8: Good compromise for most applications
            
        Returns
        -------
        dict
            Dictionary containing rectification parameters
        """
        print("\n" + "=" * 70)
        print("COMPUTING STEREO RECTIFICATION")
        print("=" * 70)
        
        print(f"\nParameters:")
        print(f"  Alpha (scaling): {alpha:.2f}")
        print(f"    0.0 = crop to valid pixels only")
        print(f"    1.0 = retain all source pixels")
        
        # Compute rectification transforms
        self.R1, self.R2, self.P1, self.P2, self.Q, roi_left, roi_right = cv2.stereoRectify(
            self.K_left,
            self.dist_left,
            self.K_right,
            self.dist_right,
            self.image_size,
            self.R,
            self.T,
            alpha=alpha,
            flags=cv2.CALIB_ZERO_DISPARITY,  # Principal points at same position
            newImageSize=self.image_size
        )
        
        self.roi_left = roi_left
        self.roi_right = roi_right
        
        # Compute remapping matrices for fast rectification
        print("\nComputing undistortion and rectification maps...")
        
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.K_left,
            self.dist_left,
            self.R1,
            self.P1,
            self.image_size,
            cv2.CV_32FC1
        )
        
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.K_right,
            self.dist_right,
            self.R2,
            self.P2,
            self.image_size,
            cv2.CV_32FC1
        )
        
        print("\n" + "=" * 70)
        print("RECTIFICATION RESULTS")
        print("=" * 70)
        
        print("\nNew camera matrix (both cameras, rectified):")
        print(f"  fx: {self.P1[0, 0]:.2f} pixels")
        print(f"  fy: {self.P1[1, 1]:.2f} pixels")
        print(f"  cx: {self.P1[0, 2]:.2f} pixels")
        print(f"  cy: {self.P1[1, 2]:.2f} pixels")
        print(f"  baseline: {-self.P2[0, 3] / self.P2[0, 0]:.2f} mm")
        
        print(f"\nValid pixel regions (x, y, width, height):")
        print(f"  Left ROI:  {roi_left}")
        print(f"  Right ROI: {roi_right}")
        
        print("\n✓ Rectification computed successfully!")
        
        return {
            'R1': self.R1,
            'R2': self.R2,
            'P1': self.P1,
            'P2': self.P2,
            'Q': self.Q,
            'roi_left': roi_left,
            'roi_right': roi_right
        }
    
    def rectify_images(self, left_image_path, right_image_path, 
                      output_dir=None, show_epipolar_lines=True):
        """
        Rectify a pair of stereo images.
        
        Parameters
        ----------
        left_image_path : str
            Path to left image
        right_image_path : str
            Path to right image
        output_dir : str, optional
            Directory to save rectified images
        show_epipolar_lines : bool
            Whether to draw epipolar lines for verification
            
        Returns
        -------
        tuple
            (rectified_left, rectified_right) as numpy arrays
        """
        if self.map_left_x is None:
            raise ValueError("Rectification not computed. Run compute_rectification() first.")
        
        print(f"\nRectifying image pair:")
        print(f"  Left:  {os.path.basename(left_image_path)}")
        print(f"  Right: {os.path.basename(right_image_path)}")
        
        # Load images
        img_left = cv2.imread(left_image_path)
        img_right = cv2.imread(right_image_path)
        
        if img_left is None or img_right is None:
            raise ValueError("Could not load images")
        
        # Apply rectification
        rectified_left = cv2.remap(
            img_left, 
            self.map_left_x, 
            self.map_left_y, 
            cv2.INTER_LINEAR
        )
        
        rectified_right = cv2.remap(
            img_right, 
            self.map_right_x, 
            self.map_right_y, 
            cv2.INTER_LINEAR
        )
        
        # Draw epipolar lines if requested
        if show_epipolar_lines:
            vis_left = rectified_left.copy()
            vis_right = rectified_right.copy()
            
            # Draw horizontal lines every 20 pixels
            for y in range(0, self.image_size[1], 20):
                color = (0, 255, 0) if y % 100 == 0 else (0, 255, 255)
                thickness = 2 if y % 100 == 0 else 1
                cv2.line(vis_left, (0, y), (self.image_size[0], y), color, thickness)
                cv2.line(vis_right, (0, y), (self.image_size[0], y), color, thickness)
            
            # Save visualization
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save individual rectified images
                left_name = os.path.basename(left_image_path)
                right_name = os.path.basename(right_image_path)
                
                left_rect_path = os.path.join(output_dir, f"rectified_{left_name}")
                right_rect_path = os.path.join(output_dir, f"rectified_{right_name}")
                
                cv2.imwrite(left_rect_path, rectified_left)
                cv2.imwrite(right_rect_path, rectified_right)
                
                # Save visualization with epipolar lines
                left_epi_path = os.path.join(output_dir, f"epipolar_{left_name}")
                right_epi_path = os.path.join(output_dir, f"epipolar_{right_name}")
                
                cv2.imwrite(left_epi_path, vis_left)
                cv2.imwrite(right_epi_path, vis_right)
                
                # Save side-by-side comparison
                combined = np.hstack([vis_left, vis_right])
                combined_path = os.path.join(output_dir, 
                                            f"stereo_pair_{left_name}")
                cv2.imwrite(combined_path, combined)
                
                print(f"  ✓ Saved rectified images to: {output_dir}")
        
        return rectified_left, rectified_right
    
    def rectify_all_scenes(self, scenes_to_process=None, output_base_dir=None):
        """
        Rectify all scene images in the stereo dataset.
        
        Parameters
        ----------
        scenes_to_process : list, optional
            List of scene folder names to process (e.g., ['scene_1', 'scene_2'])
            If None, will auto-detect all scene folders
        output_base_dir : str, optional
            Base directory for saving rectified scenes
        """
        if self.map_left_x is None:
            raise ValueError("Rectification not computed. Run compute_rectification() first.")
        
        print("\n" + "=" * 70)
        print("RECTIFYING SCENE IMAGES")
        print("=" * 70)
        
        # Auto-detect scene folders if not specified
        if scenes_to_process is None:
            left_base = PathConfig.STEREO_LEFT_DIR
            scenes_to_process = []
            
            # Find all directories starting with 'scene'
            for item in os.listdir(left_base):
                item_path = os.path.join(left_base, item)
                if os.path.isdir(item_path) and item.startswith('scene'):
                    scenes_to_process.append(item)
            
            scenes_to_process.sort()
            print(f"\nAuto-detected scenes: {scenes_to_process}")
        
        if not scenes_to_process:
            print("\nNo scenes found to process!")
            return
        
        # Process each scene
        for scene_name in scenes_to_process:
            print(f"\n{'─' * 70}")
            print(f"Processing: {scene_name}")
            print(f"{'─' * 70}")
            
            # Get scene directories
            left_scene_dir = os.path.join(PathConfig.STEREO_LEFT_DIR, 
                                         f"{scene_name}_left")
            right_scene_dir = os.path.join(PathConfig.STEREO_RIGHT_DIR, 
                                          f"{scene_name}_right")
            
            if not os.path.exists(left_scene_dir):
                print(f"  WARNING: Left scene not found: {left_scene_dir}")
                continue
            
            if not os.path.exists(right_scene_dir):
                print(f"  WARNING: Right scene not found: {right_scene_dir}")
                continue
            
            # Get all images
            left_images = []
            right_images = []
            
            for ext in CalibrationConfig.IMAGE_EXTENSIONS:
                left_images.extend(glob.glob(os.path.join(left_scene_dir, f"*{ext}")))
                right_images.extend(glob.glob(os.path.join(right_scene_dir, f"*{ext}")))
            
            left_images.sort()
            right_images.sort()
            
            # Match pairs
            for left_path in left_images:
                left_name = os.path.basename(left_path)
                right_name = left_name.replace('L', 'R').replace('left', 'right')
                
                right_path = None
                for rp in right_images:
                    if os.path.basename(rp) == right_name:
                        right_path = rp
                        break
                
                if right_path is None:
                    print(f"  WARNING: No matching right image for {left_name}")
                    continue
                
                # Set output directory
                if output_base_dir:
                    output_dir = os.path.join(output_base_dir, scene_name)
                else:
                    output_dir = os.path.join(PathConfig.STEREO_RESULTS_DIR, 
                                             'rectified_scenes', scene_name)
                
                # Rectify the pair
                self.rectify_images(left_path, right_path, output_dir, 
                                   show_epipolar_lines=True)
        
        print("\n" + "=" * 70)
        print("✓ ALL SCENES RECTIFIED")
        print("=" * 70)
    
    def save_rectification_params(self, output_dir):
        """
        Save rectification parameters to file.
        
        Parameters
        ----------
        output_dir : str
            Directory to save rectification parameters
        """
        if self.Q is None:
            raise ValueError("Rectification not computed. Run compute_rectification() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving rectification parameters to: {output_dir}")
        
        rectify_params = {
            'rectification_transforms': {
                'R1': self.R1.tolist(),
                'R2': self.R2.tolist(),
                'P1': self.P1.tolist(),
                'P2': self.P2.tolist(),
                'Q': self.Q.tolist()
            },
            'roi': {
                'left': list(self.roi_left),
                'right': list(self.roi_right)
            },
            'rectified_camera_params': {
                'fx': float(self.P1[0, 0]),
                'fy': float(self.P1[1, 1]),
                'cx': float(self.P1[0, 2]),
                'cy': float(self.P1[1, 2]),
                'baseline_mm': float(-self.P2[0, 3] / self.P2[0, 0])
            },
            'image_size': {
                'width': self.image_size[0],
                'height': self.image_size[1]
            }
        }
        
        rectify_path = os.path.join(output_dir, PathConfig.STEREO_RECTIFY_FILE)
        with open(rectify_path, 'w') as f:
            json.dump(rectify_params, f, indent=2)
        
        print(f"  ✓ Rectification parameters saved: {rectify_path}")
        
        # Also save the remapping matrices as numpy arrays for faster loading
        maps_path = os.path.join(output_dir, 'rectification_maps.npz')
        np.savez(maps_path,
                map_left_x=self.map_left_x,
                map_left_y=self.map_left_y,
                map_right_x=self.map_right_x,
                map_right_y=self.map_right_y)
        
        print(f"  ✓ Remapping matrices saved: {maps_path}")
        print("\n✓ Rectification parameters saved successfully!")


def main():
    """
    Main entry point for stereo rectification.
    """
    print("=" * 70)
    print("STEREO RECTIFICATION")
    print("=" * 70)
    
    # Load stereo calibration
    stereo_params_path = os.path.join(PathConfig.STEREO_RESULTS_DIR, 
                                     PathConfig.STEREO_PARAMS_FILE)
    
    if not os.path.exists(stereo_params_path):
        print(f"\nERROR: Stereo calibration file not found:")
        print(f"  {stereo_params_path}")
        print("\nPlease run stereo_calibrate.py first!")
        return
    
    # Create rectifier
    rectifier = StereoRectifier(stereo_params_path)
    
    # Compute rectification
    # alpha=0.8 is a good compromise: keeps most pixels while minimizing black borders
    rectifier.compute_rectification(alpha=0.8)
    
    # Save rectification parameters
    rectifier.save_rectification_params(PathConfig.STEREO_RESULTS_DIR)
    
    # Rectify all scene images
    print("\n" + "=" * 70)
    print("Do you want to rectify all scene images? (y/n)")
    print("This will process all folders starting with 'scene' in:")
    print(f"  {PathConfig.STEREO_LEFT_DIR}")
    print(f"  {PathConfig.STEREO_RIGHT_DIR}")
    response = input("Rectify scenes? [y/n]: ").strip().lower()
    
    if response == 'y':
        rectifier.rectify_all_scenes()
    else:
        print("\nSkipping scene rectification.")
        print("You can rectify individual images later using the StereoRectifier class.")
    
    print("\n" + "=" * 70)
    print("STEREO RECTIFICATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the rectified images with epipolar lines")
    print("  2. Verify that corresponding points are on the same horizontal line")
    print("  3. Use the rectified images for disparity map calculation")
    print("=" * 70)


if __name__ == "__main__":
    main()