"""
Configuration parameters for camera calibration.

This module contains all configurable parameters for both automated
and manual calibration procedures.
"""


class CheckerboardConfig:
    """
    Checkerboard pattern configuration for automated calibration.
    """
    
    # Number of internal corners (intersections) in the checkerboard
    PATTERN_SIZE = (11, 7)  # (width, height) - internal corners
    
    # Physical size of each square in millimeters
    SQUARE_SIZE = 30.0  # 30mm per square
    
    # Termination criteria for corner refinement
    CRITERIA = {
        'type': 'cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER',
        'max_iter': 30,
        'epsilon': 0.001
    }


class CalibrationConfig:
    """
    General calibration configuration.
    """
    
    # Acceptable reprojection error threshold (in pixels)
    # Target: 0.1-0.2% of image resolution
    MAX_REPROJECTION_ERROR = 2.0  # pixels
    
    # Minimum number of images required for calibration
    MIN_IMAGES = 10
    
    # Image file extensions to consider
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


class PathConfig:
    """
    File path configuration.
    """
    
    # Default paths (relative to project root)
    DATA_DIR = 'data'
    CHECKERBOARD_IMAGES_DIR = 'data/checkerboard_images'
    MANUAL_EXPERIMENTS_DIR = 'data/manual_experiments'
    RESULTS_DIR = 'results'
    
    # Output file names
    CAMERA_MATRIX_FILE = 'camera_matrix.json'
    DISTORTION_COEFFS_FILE = 'distortion_coeffs.json'
    CALIBRATION_REPORT_FILE = 'calibration_report.txt'