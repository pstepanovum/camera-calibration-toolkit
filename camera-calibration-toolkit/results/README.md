# Results Directory

This directory stores calibration outputs and analysis results.

## Contents

After running calibration, you will find:

- `camera_matrix.json`: Intrinsic camera parameters
- `distortion_coeffs.json`: Lens distortion coefficients
- `calibration_report.txt`: Detailed calibration statistics
- `reprojection_errors.png`: Visualization of calibration errors
- `undistorted_samples/`: Example undistorted images
- `manual_results/`: Results from manual calibration experiments

## Interpreting Results

### Camera Matrix
```
[[fx,  s, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

- fx, fy: Focal lengths in pixels (x and y directions)
- cx, cy: Principal point coordinates (optical center)
- s: Skew coefficient (typically zero)

### Distortion Coefficients
- k1, k2, k3: Radial distortion
- p1, p2: Tangential distortion

### Quality Metrics
- Mean reprojection error should be < 0.2% of image resolution
- Lower values indicate better calibration
