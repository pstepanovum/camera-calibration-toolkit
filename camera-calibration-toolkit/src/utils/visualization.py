"""
Visualization utilities for camera calibration results.

Provides functions for plotting reprojection errors, 3D checkerboard positions,
and other calibration quality metrics.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_3d_checkerboard_positions(rvecs, tvecs, pattern_size, square_size, save_path=None):
    """
    Plot 3D positions of the checkerboard in camera coordinate system.
    
    Parameters
    ----------
    rvecs : list of ndarray
        Rotation vectors for each image
    tvecs : list of ndarray
        Translation vectors for each image
    pattern_size : tuple
        (width, height) of checkerboard internal corners
    square_size : float
        Physical size of each square in mm
    save_path : str, optional
        Path to save the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate checkerboard corner positions in checkerboard coordinate system
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Color map for different images
    colors = plt.cm.rainbow(np.linspace(0, 1, len(rvecs)))
    
    # Transform and plot each checkerboard position
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Transform checkerboard points to camera coordinates
        points_cam = (R @ objp.T).T + tvec.T
        
        # Plot the checkerboard points
        ax.scatter(points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
                  c=[colors[i]], marker='o', s=20, alpha=0.6,
                  label=f'Image {i+1}')
    
    # Plot camera at origin
    ax.scatter([0], [0], [0], c='red', marker='^', s=200, 
              label='Camera', edgecolors='black', linewidths=2)
    
    # Set labels and title
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title('3D Checkerboard Positions in Camera Coordinate System', 
                fontsize=14, fontweight='bold')
    
    # Add legend (only show first few to avoid clutter)
    if len(rvecs) <= 15:
        ax.legend(loc='upper left', fontsize=8)
    else:
        # Show legend for camera only if too many images
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[-1]], [labels[-1]], loc='upper left', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    max_range = np.array([points_cam[:, 0].max()-points_cam[:, 0].min(),
                         points_cam[:, 1].max()-points_cam[:, 1].min(),
                         points_cam[:, 2].max()-points_cam[:, 2].min()]).max() / 2.0
    
    mid_x = (points_cam[:, 0].max()+points_cam[:, 0].min()) * 0.5
    mid_y = (points_cam[:, 1].max()+points_cam[:, 1].min()) * 0.5
    mid_z = (points_cam[:, 2].max()+points_cam[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D position plot saved: {save_path}")
    
    plt.close()


def plot_reprojection_errors(object_points, image_points, camera_matrix, 
                             dist_coeffs, rvecs, tvecs, image_paths, save_path=None):
    """
    Plot reprojection errors for each image.
    
    Parameters
    ----------
    object_points : list of ndarray
        3D object points for each image
    image_points : list of ndarray
        2D image points for each image
    camera_matrix : ndarray
        Camera intrinsic matrix
    dist_coeffs : ndarray
        Distortion coefficients
    rvecs : list of ndarray
        Rotation vectors
    tvecs : list of ndarray
        Translation vectors
    image_paths : list of str
        Paths to calibration images
    save_path : str, optional
        Path to save the plot
    """
    errors_per_image = []
    
    for i in range(len(object_points)):
        # Project 3D points to 2D
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i],
                                         camera_matrix, dist_coeffs)
        
        # Calculate error
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors_per_image.append(error)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    image_numbers = range(1, len(errors_per_image) + 1)
    bars = ax.bar(image_numbers, errors_per_image, color='steelblue', edgecolor='black')
    
    # Color bars based on error magnitude
    mean_error = np.mean(errors_per_image)
    for i, bar in enumerate(bars):
        if errors_per_image[i] > mean_error * 1.5:
            bar.set_color('coral')
        elif errors_per_image[i] < mean_error * 0.5:
            bar.set_color('lightgreen')
    
    # Add mean line
    ax.axhline(y=mean_error, color='red', linestyle='--', linewidth=2,
              label=f'Mean Error: {mean_error:.4f} pixels')
    
    # Labels and title
    ax.set_xlabel('Image Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reprojection Error (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Image Reprojection Errors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (num, error) in enumerate(zip(image_numbers, errors_per_image)):
        ax.text(num, error + max(errors_per_image) * 0.02, f'{error:.3f}',
               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reprojection error plot saved: {save_path}")
    
    plt.close()


def plot_error_distribution(object_points, image_points, camera_matrix,
                            dist_coeffs, rvecs, tvecs, save_path=None):
    """
    Plot distribution of reprojection errors.
    
    Parameters
    ----------
    object_points : list of ndarray
        3D object points for each image
    image_points : list of ndarray
        2D image points for each image
    camera_matrix : ndarray
        Camera intrinsic matrix
    dist_coeffs : ndarray
        Distortion coefficients
    rvecs : list of ndarray
        Rotation vectors
    tvecs : list of ndarray
        Translation vectors
    save_path : str, optional
        Path to save the plot
    """
    all_errors = []
    
    for i in range(len(object_points)):
        # Project 3D points to 2D
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i],
                                         camera_matrix, dist_coeffs)
        
        # Calculate per-point errors
        point_errors = np.linalg.norm(image_points[i] - imgpoints2, axis=2).ravel()
        all_errors.extend(point_errors)
    
    all_errors = np.array(all_errors)
    
    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(all_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_errors):.4f} pixels')
    ax1.axvline(np.median(all_errors), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(all_errors):.4f} pixels')
    ax1.set_xlabel('Reprojection Error (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2.boxplot(all_errors, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', edgecolor='black'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))
    ax2.set_ylabel('Reprojection Error (pixels)', fontsize=12, fontweight='bold')
    ax2.set_title('Error Statistics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Min: {np.min(all_errors):.4f}\n'
    stats_text += f'Max: {np.max(all_errors):.4f}\n'
    stats_text += f'Std: {np.std(all_errors):.4f}'
    ax2.text(1.15, np.median(all_errors), stats_text,
            transform=ax2.get_yaxis_transform(),
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved: {save_path}")
    
    plt.close()


def create_calibration_summary(camera_matrix, dist_coeffs, reprojection_error,
                               image_size, num_images, save_path=None):
    """
    Create a summary visualization of calibration results.
    
    Parameters
    ----------
    camera_matrix : ndarray
        Camera intrinsic matrix
    dist_coeffs : ndarray
        Distortion coefficients
    reprojection_error : float
        Mean reprojection error
    image_size : tuple
        (width, height) of images
    num_images : int
        Number of images used
    save_path : str, optional
        Path to save the plot
    """
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Camera Calibration Summary', fontsize=16, fontweight='bold')
    
    # Remove axes
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Create text summary
    summary_text = "CALIBRATION RESULTS\n"
    summary_text += "=" * 60 + "\n\n"
    
    summary_text += "Intrinsic Parameters:\n"
    summary_text += "-" * 60 + "\n"
    summary_text += f"fx (focal length x):     {camera_matrix[0, 0]:>10.2f} pixels\n"
    summary_text += f"fy (focal length y):     {camera_matrix[1, 1]:>10.2f} pixels\n"
    summary_text += f"cx (principal point x):  {camera_matrix[0, 2]:>10.2f} pixels\n"
    summary_text += f"cy (principal point y):  {camera_matrix[1, 2]:>10.2f} pixels\n"
    summary_text += f"skew:                    {camera_matrix[0, 1]:>10.6f}\n\n"
    
    aspect_ratio = camera_matrix[0, 0] / camera_matrix[1, 1]
    summary_text += f"Pixel Aspect Ratio:      {aspect_ratio:>10.6f}\n\n"
    
    summary_text += "Distortion Coefficients:\n"
    summary_text += "-" * 60 + "\n"
    summary_text += f"k1 (radial):            {dist_coeffs[0, 0]:>10.6f}\n"
    summary_text += f"k2 (radial):            {dist_coeffs[0, 1]:>10.6f}\n"
    summary_text += f"p1 (tangential):        {dist_coeffs[0, 2]:>10.6f}\n"
    summary_text += f"p2 (tangential):        {dist_coeffs[0, 3]:>10.6f}\n"
    summary_text += f"k3 (radial):            {dist_coeffs[0, 4]:>10.6f}\n\n"
    
    summary_text += "Quality Metrics:\n"
    summary_text += "-" * 60 + "\n"
    summary_text += f"Mean Reprojection Error: {reprojection_error:>10.4f} pixels\n"
    error_percent = (reprojection_error / max(image_size)) * 100
    summary_text += f"Error % of resolution:   {error_percent:>10.4f}%\n"
    summary_text += f"Image resolution:        {image_size[0]} x {image_size[1]} pixels\n"
    summary_text += f"Number of images:        {num_images:>10d}\n\n"
    
    if reprojection_error < 0.5:
        quality = "EXCELLENT"
    elif reprojection_error < 1.0:
        quality = "VERY GOOD"
    elif reprojection_error < 2.0:
        quality = "GOOD"
    else:
        quality = "ACCEPTABLE"
    
    summary_text += f"Calibration Quality:     {quality}\n"
    
    # Display text
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontfamily='monospace', fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration summary saved: {save_path}")
    
    plt.close()


def generate_all_visualizations(calibrator, output_dir):
    """
    Generate all calibration visualizations.
    
    Parameters
    ----------
    calibrator : CameraCalibrator
        Calibrator object with completed calibration
    output_dir : str
        Directory to save visualizations
    """
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 3D checkerboard positions
    plot_3d_checkerboard_positions(
        calibrator.rvecs,
        calibrator.tvecs,
        calibrator.pattern_size,
        calibrator.square_size,
        save_path=os.path.join(vis_dir, "3d_checkerboard_positions.png")
    )
    
    # Reprojection errors per image
    plot_reprojection_errors(
        calibrator.object_points,
        calibrator.image_points,
        calibrator.camera_matrix,
        calibrator.dist_coeffs,
        calibrator.rvecs,
        calibrator.tvecs,
        calibrator.image_paths,
        save_path=os.path.join(vis_dir, "reprojection_errors_per_image.png")
    )
    
    # Error distribution
    plot_error_distribution(
        calibrator.object_points,
        calibrator.image_points,
        calibrator.camera_matrix,
        calibrator.dist_coeffs,
        calibrator.rvecs,
        calibrator.tvecs,
        save_path=os.path.join(vis_dir, "error_distribution.png")
    )
    
    # Calibration summary
    create_calibration_summary(
        calibrator.camera_matrix,
        calibrator.dist_coeffs,
        calibrator.reprojection_error,
        calibrator.image_size,
        len(calibrator.object_points),
        save_path=os.path.join(vis_dir, "calibration_summary.png")
    )
    
    print(f"\n✓ All visualizations saved to: {vis_dir}")