"""
Manual Calibration: Principal Point Estimation

This script estimates the principal point (optical center) using vanishing points
from images of parallel lines.

Theory:
- Parallel lines in 3D appear to converge to a vanishing point in the image
- Multiple vanishing points from different sets of parallel lines can be used
- The principal point is related to the pattern of vanishing points
- For a calibrated camera, vanishing points of orthogonal directions provide constraints

Usage:
1. Take a photo containing parallel lines (building edges, tiles, etc.)
2. Run this script and mark several sets of parallel lines
3. Script calculates vanishing points and estimates the principal point
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


class PrincipalPointEstimator:
    """
    Estimates principal point (optical center) using vanishing point analysis.
    """
    
    def __init__(self):
        self.image = None
        self.image_path = None
        self.line_sets = []  # List of line sets, each set has parallel lines
        self.current_lines = []  # Current set of lines being drawn
        self.display_image = None
        self.vanishing_points = []
        self.colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]
        
    def load_image(self, image_path):
        """
        Load the image containing parallel lines.
        
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
        Handle mouse clicks to record line endpoints.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_lines.append((x, y))
            
            # Draw point
            color = self.colors[len(self.line_sets) % len(self.colors)]
            cv2.circle(self.display_image, (x, y), 3, color, -1)
            
            # If we have 2 points, draw the line
            if len(self.current_lines) % 2 == 0:
                pt1 = self.current_lines[-2]
                pt2 = self.current_lines[-1]
                cv2.line(self.display_image, pt1, pt2, color, 2)
                line_num = len(self.current_lines) // 2
                print(f"  Line {line_num}: ({pt1[0]}, {pt1[1]}) -> ({pt2[0]}, {pt2[1]})")
            
            cv2.imshow('Mark Parallel Lines', self.display_image)
    
    def select_parallel_lines_interactive(self):
        """
        Interactive GUI for selecting multiple sets of parallel lines.
        
        Instructions:
        - For each set of parallel lines (e.g., horizontal edges of a building):
          * Click START and END points of each parallel line
          * Mark at least 2 parallel lines per set
          * Press SPACE to finish the current set and start a new set
        - Press ENTER when all sets are complete
        """
        print("\n" + "=" * 70)
        print("INTERACTIVE PARALLEL LINE SELECTION")
        print("=" * 70)
        print("\nInstructions:")
        print("  1. Identify PARALLEL lines in the image (e.g., building edges)")
        print("  2. For each line, click START point, then END point")
        print("  3. Mark at least 2-3 parallel lines")
        print("  4. Press SPACE when done with this set of parallel lines")
        print("  5. Repeat for 2-3 different sets of parallel lines")
        print("     (e.g., horizontal lines, vertical lines, diagonal lines)")
        print("  6. Press ENTER when all sets are marked")
        print("  7. Press 'r' to RESET current set")
        print("  8. Press 'q' to QUIT")
        print("\nTIP: The more sets of parallel lines, the better the estimate!")
        print("\nWindow will open now...")
        
        cv2.namedWindow('Mark Parallel Lines')
        cv2.setMouseCallback('Mark Parallel Lines', self.mouse_callback)
        
        set_number = 1
        print(f"\n--- SET {set_number}: Mark parallel lines ---")
        print("(Click start and end points for each line, press SPACE when done with this set)")
        
        while True:
            cv2.imshow('Mark Parallel Lines', self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            # Press 'r' to reset current set
            if key == ord('r'):
                print(f"\nResetting SET {set_number}...")
                # Redraw image with previous sets only
                self.display_image = self.image.copy()
                for i, line_set in enumerate(self.line_sets):
                    color = self.colors[i % len(self.colors)]
                    for line in line_set:
                        cv2.line(self.display_image, line[0], line[1], color, 2)
                self.current_lines = []
                cv2.imshow('Mark Parallel Lines', self.display_image)
            
            # Press SPACE to finish current set and start new set
            elif key == ord(' '):
                if len(self.current_lines) >= 4:  # At least 2 lines (4 points)
                    # Convert points to line segments
                    lines = []
                    for i in range(0, len(self.current_lines), 2):
                        if i + 1 < len(self.current_lines):
                            lines.append((self.current_lines[i], self.current_lines[i+1]))
                    
                    self.line_sets.append(lines)
                    print(f"✓ SET {set_number} complete: {len(lines)} parallel lines marked")
                    
                    self.current_lines = []
                    set_number += 1
                    print(f"\n--- SET {set_number}: Mark another set of parallel lines ---")
                    print("(Or press ENTER if you're done)")
                else:
                    print("Need at least 2 lines (4 points) per set!")
            
            # Press ENTER to finish all sets
            elif key == 13:  # Enter key
                # Save current set if any
                if len(self.current_lines) >= 4:
                    lines = []
                    for i in range(0, len(self.current_lines), 2):
                        if i + 1 < len(self.current_lines):
                            lines.append((self.current_lines[i], self.current_lines[i+1]))
                    self.line_sets.append(lines)
                    print(f"✓ SET {set_number} complete: {len(lines)} parallel lines marked")
                
                if len(self.line_sets) >= 2:
                    print(f"\n✓ Total: {len(self.line_sets)} sets of parallel lines marked")
                    break
                else:
                    print("\nNeed at least 2 sets of parallel lines! Continue marking...")
            
            # Press 'q' to quit
            elif key == ord('q'):
                print("\nQuitting...")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def compute_vanishing_point(self, lines):
        """
        Compute vanishing point from a set of parallel lines.
        
        Parameters
        ----------
        lines : list of tuples
            List of line segments, each as ((x1,y1), (x2,y2))
            
        Returns
        -------
        tuple or None
            (x, y) coordinates of vanishing point, or None if computation fails
        """
        if len(lines) < 2:
            return None
        
        # Convert lines to homogeneous form (ax + by + c = 0)
        line_equations = []
        for (pt1, pt2) in lines:
            x1, y1 = pt1
            x2, y2 = pt2
            
            # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x1
            
            line_equations.append([a, b, c])
        
        # Find vanishing point as intersection of lines
        # Solve using least squares: minimize sum of distances to all lines
        A = []
        b = []
        for eq in line_equations:
            a, b_coef, c = eq
            A.append([a, b_coef])
            b.append(-c)
        
        A = np.array(A)
        b = np.array(b)
        
        try:
            # Least squares solution
            vp, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return tuple(vp)
        except:
            return None
    
    def calculate_principal_point(self):
        """
        Calculate principal point from vanishing points.
        
        Returns
        -------
        dict
            Dictionary containing principal point estimate and related data
        """
        if len(self.line_sets) < 2:
            raise ValueError("Need at least 2 sets of parallel lines")
        
        print("\n" + "=" * 70)
        print("CALCULATING VANISHING POINTS")
        print("=" * 70)
        
        # Compute vanishing point for each set
        for i, line_set in enumerate(self.line_sets):
            vp = self.compute_vanishing_point(line_set)
            if vp is not None:
                self.vanishing_points.append(vp)
                print(f"Set {i+1}: Vanishing point at ({vp[0]:.1f}, {vp[1]:.1f})")
                
                # Draw vanishing point on image
                color = self.colors[i % len(self.colors)]
                x, y = int(vp[0]), int(vp[1])
                
                # Only draw if within reasonable bounds
                if -5000 < x < self.image.shape[1] + 5000 and -5000 < y < self.image.shape[0] + 5000:
                    cv2.circle(self.display_image, (x, y), 8, color, -1)
                    cv2.circle(self.display_image, (x, y), 10, (255, 255, 255), 2)
                    cv2.putText(self.display_image, f"VP{i+1}", 
                               (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, color, 2)
        
        if len(self.vanishing_points) < 2:
            raise ValueError("Could not compute enough vanishing points")
        
        print("\n" + "=" * 70)
        print("ESTIMATING PRINCIPAL POINT")
        print("=" * 70)
        
        # Method 1: Average of vanishing points (simple approximation)
        vp_array = np.array(self.vanishing_points)
        pp_average = np.mean(vp_array, axis=0)
        
        # Method 2: Geometric mean (better for some configurations)
        # For orthogonal vanishing points, principal point lies on certain geometric loci
        
        # Method 3: Use image center as reference
        image_center = np.array([self.image.shape[1] / 2, self.image.shape[0] / 2])
        
        # Weighted estimate: combine vanishing point average with image center
        # This is more robust than using vanishing points alone
        alpha = 0.7  # Weight for vanishing points
        pp_weighted = alpha * pp_average + (1 - alpha) * image_center
        
        print(f"\nVanishing point average: ({pp_average[0]:.1f}, {pp_average[1]:.1f})")
        print(f"Image center: ({image_center[0]:.1f}, {image_center[1]:.1f})")
        print(f"Weighted estimate: ({pp_weighted[0]:.1f}, {pp_weighted[1]:.1f})")
        
        # Calculate offset from image center
        offset = pp_weighted - image_center
        offset_percent_x = (offset[0] / image_center[0]) * 100
        offset_percent_y = (offset[1] / image_center[1]) * 100
        
        print(f"\nOffset from image center:")
        print(f"  X: {offset[0]:.1f} pixels ({offset_percent_x:.2f}%)")
        print(f"  Y: {offset[1]:.1f} pixels ({offset_percent_y:.2f}%)")
        
        # Draw principal point on image
        pp_x, pp_y = int(pp_weighted[0]), int(pp_weighted[1])
        cv2.drawMarker(self.display_image, (pp_x, pp_y), (0, 255, 255), 
                      cv2.MARKER_CROSS, 30, 3)
        cv2.circle(self.display_image, (pp_x, pp_y), 5, (0, 255, 255), -1)
        cv2.putText(self.display_image, "Principal Point", 
                   (pp_x + 20, pp_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        # Also draw image center for comparison
        center_x, center_y = int(image_center[0]), int(image_center[1])
        cv2.drawMarker(self.display_image, (center_x, center_y), (128, 128, 128), 
                      cv2.MARKER_DIAMOND, 20, 2)
        cv2.putText(self.display_image, "Image Center", 
                   (center_x + 20, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (128, 128, 128), 2)
        
        results = {
            'cx_estimated': float(pp_weighted[0]),
            'cy_estimated': float(pp_weighted[1]),
            'cx_image_center': float(image_center[0]),
            'cy_image_center': float(image_center[1]),
            'offset_x_pixels': float(offset[0]),
            'offset_y_pixels': float(offset[1]),
            'offset_x_percent': float(offset_percent_x),
            'offset_y_percent': float(offset_percent_y),
            'vanishing_points': [[float(x), float(y)] for x, y in self.vanishing_points],
            'num_line_sets': len(self.line_sets),
            'num_vanishing_points': len(self.vanishing_points)
        }
        
        return results
    
    def save_results(self, results, output_dir, comparison_data=None):
        """
        Save principal point results and comparison.
        
        Parameters
        ----------
        results : dict
            Principal point calculation results
        output_dir : str
            Directory to save results
        comparison_data : dict, optional
            Automated calibration data for comparison
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numerical results
        results_path = os.path.join(output_dir, 'principal_point_manual.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")
        
        # Save marked image
        image_path = os.path.join(output_dir, 'principal_point_measurement.png')
        cv2.imwrite(image_path, self.display_image)
        print(f"Marked image saved: {image_path}")
        
        # Display the result
        cv2.imshow('Principal Point Estimate', self.display_image)
        print("\nPress any key to close the visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save detailed report
        report_path = os.path.join(output_dir, 'principal_point_report.txt')
        with open(report_path, 'w') as f:
            f.write("MANUAL CALIBRATION: PRINCIPAL POINT ESTIMATION\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("METHOD\n")
            f.write("-" * 70 + "\n")
            f.write("Vanishing point analysis from parallel lines\n")
            f.write(f"Image: {os.path.basename(self.image_path)}\n")
            f.write(f"Number of line sets: {results['num_line_sets']}\n")
            f.write(f"Number of vanishing points: {results['num_vanishing_points']}\n\n")
            
            f.write("VANISHING POINTS\n")
            f.write("-" * 70 + "\n")
            for i, vp in enumerate(results['vanishing_points']):
                f.write(f"VP {i+1}: ({vp[0]:.1f}, {vp[1]:.1f})\n")
            f.write("\n")
            
            f.write("RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Estimated principal point (cx, cy): ({results['cx_estimated']:.1f}, {results['cy_estimated']:.1f})\n")
            f.write(f"Image center: ({results['cx_image_center']:.1f}, {results['cy_image_center']:.1f})\n\n")
            
            f.write("OFFSET FROM IMAGE CENTER\n")
            f.write("-" * 70 + "\n")
            f.write(f"X offset: {results['offset_x_pixels']:.1f} pixels ({results['offset_x_percent']:.2f}%)\n")
            f.write(f"Y offset: {results['offset_y_pixels']:.1f} pixels ({results['offset_y_percent']:.2f}%)\n\n")
            
            # Interpretation
            if abs(results['offset_x_percent']) < 5 and abs(results['offset_y_percent']) < 5:
                f.write("Principal point is very close to image center (< 5% offset)\n")
                f.write("This is typical for most cameras.\n")
            elif abs(results['offset_x_percent']) < 10 and abs(results['offset_y_percent']) < 10:
                f.write("Principal point has moderate offset from center (5-10%)\n")
            else:
                f.write("Principal point has significant offset from center (> 10%)\n")
                f.write("This may indicate lens misalignment or specialized optics.\n")
            f.write("\n")
            
            if comparison_data:
                f.write("COMPARISON WITH AUTOMATED CALIBRATION\n")
                f.write("-" * 70 + "\n")
                cx_auto = comparison_data.get('cx', 0)
                cy_auto = comparison_data.get('cy', 0)
                
                f.write(f"Automated cx: {cx_auto:.1f} pixels\n")
                f.write(f"Manual cx: {results['cx_estimated']:.1f} pixels\n")
                cx_diff = abs(cx_auto - results['cx_estimated'])
                cx_diff_percent = (cx_diff / cx_auto) * 100 if cx_auto != 0 else 0
                f.write(f"cx difference: {cx_diff:.1f} pixels ({cx_diff_percent:.2f}%)\n\n")
                
                f.write(f"Automated cy: {cy_auto:.1f} pixels\n")
                f.write(f"Manual cy: {results['cy_estimated']:.1f} pixels\n")
                cy_diff = abs(cy_auto - results['cy_estimated'])
                cy_diff_percent = (cy_diff / cy_auto) * 100 if cy_auto != 0 else 0
                f.write(f"cy difference: {cy_diff:.1f} pixels ({cy_diff_percent:.2f}%)\n\n")
                
                avg_diff_percent = (cx_diff_percent + cy_diff_percent) / 2
                if avg_diff_percent < 5:
                    f.write("Agreement: EXCELLENT (< 5% difference)\n")
                elif avg_diff_percent < 10:
                    f.write("Agreement: GOOD (< 10% difference)\n")
                elif avg_diff_percent < 20:
                    f.write("Agreement: FAIR (< 20% difference)\n")
                else:
                    f.write("Agreement: POOR (> 20% difference)\n")
                    f.write("\nNote: Principal point estimation from vanishing points\n")
                    f.write("is inherently less accurate than checkerboard calibration.\n")
                    f.write("Differences of 10-20% are common and acceptable.\n")
        
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
    Main entry point for principal point estimation.
    """
    print("=" * 70)
    print("MANUAL CALIBRATION: PRINCIPAL POINT ESTIMATION")
    print("=" * 70)
    
    # Get image path
    print("\nEnter the path to your image with parallel lines:")
    print("(Good examples: building facades, hallways, tiled floors, railroad tracks)")
    print("(Press Enter to use default: data/manual_experiments/principal_point_image.png)")
    image_path = input("Image path: ").strip()
    
    if not image_path:
        image_path = os.path.join(PathConfig.MANUAL_EXPERIMENTS_DIR, 
                                 "principal_point_image.png")
    
    if not os.path.exists(image_path):
        print(f"\nERROR: Image not found: {image_path}")
        print("Please provide a valid image path.")
        return
    
    # Create estimator and process
    estimator = PrincipalPointEstimator()
    estimator.load_image(image_path)
    
    # Interactive line selection
    success = estimator.select_parallel_lines_interactive()
    
    if not success:
        print("\nCalibration cancelled.")
        return
    
    # Calculate principal point
    try:
        results = estimator.calculate_principal_point()
    except ValueError as e:
        print(f"\nERROR: {e}")
        return
    
    # Load automated calibration for comparison
    auto_calib = load_automated_calibration()
    if auto_calib:
        print("\n" + "=" * 70)
        print("COMPARISON WITH AUTOMATED CALIBRATION")
        print("=" * 70)
        cx_auto = auto_calib['cx']
        cy_auto = auto_calib['cy']
        print(f"Automated (cx, cy): ({cx_auto:.1f}, {cy_auto:.1f})")
        print(f"Manual (cx, cy): ({results['cx_estimated']:.1f}, {results['cy_estimated']:.1f})")
        cx_diff = abs(cx_auto - results['cx_estimated'])
        cy_diff = abs(cy_auto - results['cy_estimated'])
        print(f"Difference: ({cx_diff:.1f}, {cy_diff:.1f}) pixels")
    
    # Save results
    output_dir = os.path.join(PathConfig.RESULTS_DIR, "manual_calibration")
    estimator.save_results(results, output_dir, auto_calib)
    
    print("\n" + "=" * 70)
    print("PRINCIPAL POINT ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()