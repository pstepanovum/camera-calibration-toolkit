# Camera Calibration Toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive toolkit for camera intrinsic calibration implementing both automated and manual calibration methodologies. Developed as part of ECE 738 Computer Vision coursework at the University of Miami, Spring 2026.

## Overview

This project implements two complementary approaches to camera calibration:

1. **Automated Calibration**: Utilizes OpenCV's calibration framework with checkerboard patterns to determine intrinsic parameters and lens distortion coefficients through optimization-based methods.

2. **Manual Calibration**: Employs custom-designed experiments to independently estimate camera parameters including focal length, aspect ratio, and principal point using fundamental projection relationships.

## Features

- Automated checkerboard-based calibration with reprojection error analysis
- Manual calibration experiments for independent parameter estimation
- Comprehensive visualization of calibration results
- Distortion correction and undistortion utilities
- Detailed uncertainty quantification for calibration parameters
- Export calibration data in standard formats

## Project Structure

```
camera-calibration-toolkit/
├── src/
│   ├── automated/          # Automated calibration implementation
│   │   └── calibrate.py
│   ├── manual/             # Manual calibration experiments
│   │   ├── aspect_ratio.py
│   │   └── focal_length.py
│   └── utils/              # Shared utilities and visualization tools
├── data/
│   ├── checkerboard_images/    # Input images for automated calibration
│   └── manual_experiments/     # Data for manual calibration
├── results/                # Output calibration parameters and visualizations
├── tests/                  # Unit tests
├── docs/                   # Documentation and technical report
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/camera-calibration-toolkit.git
cd camera-calibration-toolkit
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Automated Calibration

Place checkerboard images in `data/checkerboard_images/` and run:

```bash
python src/automated/calibrate.py
python src/automated/calibrate.py --input data/checkerboard_images/ --output results/
```

### Manual Calibration

Execute individual calibration experiments:

```bash
# Aspect ratio estimation
python src/manual/aspect_ratio.py

# Focal length estimation
python src/manual/focal_length.py

# Principal point estimation
python src/manual/principal_point.py
```

## Calibration Quality Metrics

The toolkit reports the following quality metrics:

- **Reprojection Error**: Mean pixel error between detected and reprojected corners
- **Target Accuracy**: 0.1-0.2% of image resolution (1-2 pixels per 1000 pixels)
- **Parameter Uncertainty**: Standard deviation for each calibrated parameter

## Methodology

### Automated Calibration

Implements the Zhang method using multiple views of a planar checkerboard pattern. The calibration process:

1. Detects checkerboard corners in each image
2. Estimates initial intrinsic parameters
3. Refines parameters through non-linear optimization
4. Computes lens distortion coefficients

### Manual Calibration

**Aspect Ratio**: Measures differential scaling between x and y axes using known square patterns.

**Focal Length**: Applies projection equations with measured object dimensions and distances.

**Principal Point**: Determines optical center using geometric properties and vanishing points.

## Results

Calibration results are saved in the `results/` directory:

- `camera_matrix.json`: Intrinsic parameters
- `distortion_coeffs.json`: Lens distortion coefficients
- `calibration_report.txt`: Detailed calibration statistics
- Visualization plots of reprojection errors and undistorted images

## Contributing

This is an academic project. Contributions, suggestions, and discussions are welcome through issues and pull requests.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References

- Zhang, Z. (2000). A flexible new technique for camera calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- OpenCV Documentation: Camera Calibration and 3D Reconstruction
- Hartley, R., & Zisserman, A. (2004). Multiple View Geometry in Computer Vision.

## Contact

For questions or discussions regarding this project, please open an issue on GitHub.

## Acknowledgments

- Course: ECE 738 Computer Vision
- Institution: University of Miami
- Semester: Spring 2026
