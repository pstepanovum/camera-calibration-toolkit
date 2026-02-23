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
    # 15, 11
    PATTERN_SIZE = (14, 10)  # (width, height) - internal corners
    
    # Physical size of each square in millimeters
    SQUARE_SIZE = 25  # 10mm per square
    
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
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff',
                    '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF')


class StereoConfig:
    """
    Stereo vision configuration.
    """
    
    # Stereo calibration flags
    # Use CALIB_FIX_INTRINSIC if you want to use pre-calibrated individual cameras
    STEREO_CALIB_FLAGS = 0  # 0 = refine everything, or cv2.CALIB_FIX_INTRINSIC
    
    # Stereo matching parameters for disparity calculation
    STEREO_MATCHER = 'SGBM'  # 'BM' or 'SGBM' (Semi-Global Block Matching is better)
    
    # Disparity range parameters
    MIN_DISPARITY = 0
    NUM_DISPARITIES = 16 * 6  # Must be divisible by 16, typical range: 64-128
    BLOCK_SIZE = 11  # Block size for matching (odd number, 5-21 typical)
    
    # SGBM-specific parameters
    SGBM_P1 = 8 * 3 * 11**2  # Penalty for small disparity changes
    SGBM_P2 = 32 * 3 * 11**2  # Penalty for large disparity changes
    SGBM_DISP_MAX_DIFF = 1
    SGBM_UNIQUENESS_RATIO = 10
    SGBM_SPECKLE_WINDOW_SIZE = 100
    SGBM_SPECKLE_RANGE = 32
    SGBM_MODE = 'SGBM_MODE_SGBM_3WAY'  # or 'SGBM_MODE_HH' for better quality
    
    # Post-processing
    WLS_FILTER_ENABLE = True  # Use weighted least squares filter for refinement
    WLS_LAMBDA = 8000.0
    WLS_SIGMA = 1.5


class PathConfig:
    """
    File path configuration.
    """
    
    # Default paths (relative to project root)
    DATA_DIR = 'data'
    CHECKERBOARD_IMAGES_DIR = 'data/checkerboard_images'
    MANUAL_EXPERIMENTS_DIR = 'data/manual_experiments'
    RESULTS_DIR = 'results'
    
    # Stereo paths
    STEREO_DATA_DIR = 'data/stereo_images'
    STEREO_LEFT_DIR = 'data/stereo_images/left'
    STEREO_RIGHT_DIR = 'data/stereo_images/right'
    STEREO_RESULTS_DIR = 'results/stereo'
    
    # Output file names - Monocular
    CAMERA_MATRIX_FILE = 'camera_matrix.json'
    DISTORTION_COEFFS_FILE = 'distortion_coeffs.json'
    CALIBRATION_REPORT_FILE = 'calibration_report.txt'
    
    # Output file names - Stereo
    STEREO_PARAMS_FILE = 'stereo_params.json'
    STEREO_RECTIFY_FILE = 'stereo_rectify.json'
    STEREO_REPORT_FILE = 'stereo_calibration_report.txt'