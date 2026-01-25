# Manual Calibration: Aspect Ratio Experiment

## Overview

This experiment measures the pixel aspect ratio (fx/fy) by manually measuring a known square object in an image.

## File Location

**Script:** `src/manual/aspect_ratio.py`

## Theory

**Pixel Aspect Ratio = fx/fy**

- If pixels are **square** (aspect ratio = 1.0), a physical square appears as a square in the image
- If pixels are **rectangular**, a physical square appears distorted
- By measuring a known square, we can calculate fx/fy independently

From your automated calibration:
- fx = 722.06 pixels
- fy = 732.73 pixels
- **Aspect ratio = 0.9854** (pixels are slightly taller than wide)

## Preparation

### Step 1: Prepare Your Square Object

You need a square object with known dimensions. Options:

**Option A: Use your checkerboard**
- Each square is 30mm × 30mm
- Take a photo of the board perpendicular to camera
- You'll measure one of the squares

**Option B: Create a paper square**
- Cut a precise square from paper (e.g., 50mm × 50mm)
- Tape it flat to a surface
- Take a photo perpendicular to camera

**Option C: Use any square object**
- Measure its size precisely with a ruler
- Photograph it straight-on

### Step 2: Take the Photo

**Important:**
- Position the square **perpendicular** to the camera (not tilted)
- Use good lighting
- Keep the camera steady
- Fill a reasonable portion of the frame with the square
- Same camera settings as your calibration images

### Step 3: Save the Image

Save your image to:
```
data/manual_experiments/square_image.jpg
```

Or use any path - you'll specify it when running the script.

## How to Run

### Command

From the project root directory:

```bash
python src/manual/aspect_ratio.py
```

### Interactive Prompts

**1. Image Path:**
```
Enter the path to your image with a square object:
(Press Enter to use default: data/manual_experiments/square_image.jpg)
Image path: 
```

Enter your image path, or press Enter for default.

**2. Physical Size:**
```
Enter the physical size of your square object (in millimeters):
(e.g., if you're using a 30mm square from the checkerboard, enter 30)
Size (mm): 
```

Enter the size of your square (e.g., `30` if using checkerboard square).

### Corner Selection

A window will open showing your image.

**Instructions:**
1. Click on the **four corners** of your square in order
2. Order: top-left → top-right → bottom-right → bottom-left (or any consistent order)
3. Red dots and numbers will appear as you click
4. Green lines connect the corners

**Controls:**
- Click to mark corners
- Press **'r'** to reset if you make a mistake
- Press **any other key** when all 4 corners are marked
- Press **'q'** to quit

### Example

```
Corner 1: (245, 189)
Corner 2: (567, 195)
Corner 3: (561, 423)
Corner 4: (239, 417)

All 4 corners selected! Press any key to continue...
```

## Output

### Console Output

```
CALCULATING ASPECT RATIO
======================================================================
Side 1 length: 322.06 pixels
Side 2 length: 228.27 pixels
Side 3 length: 322.00 pixels
Side 4 length: 228.00 pixels

Average horizontal length: 322.03 pixels
Average vertical length: 228.14 pixels

Physical square size: 30.0 mm
Measured aspect ratio (fx/fy): 1.4115

COMPARISON WITH AUTOMATED CALIBRATION
======================================================================
Automated aspect ratio (fx/fy): 0.985400
Manual aspect ratio: 1.411500
Difference: 0.426100 (43.24%)
```

### Saved Files

In `results/manual_calibration/`:

1. **`aspect_ratio_manual.json`** - Numerical results
2. **`aspect_ratio_measurement.png`** - Your image with marked corners
3. **`aspect_ratio_report.txt`** - Detailed report with comparison

## Interpreting Results

### Good Agreement

If manual and automated aspect ratios are within **5%**:
- ✓ Excellent validation of automated calibration
- Both methods agree on pixel shape

### Moderate Agreement (5-10%)

- Measurement errors in manual method
- Possible slight tilt in your square image
- Still acceptable validation

### Poor Agreement (>10%)

Possible causes:
- Square object was tilted (not perpendicular)
- Inaccurate corner clicking
- Wrong physical size entered
- Image distortion at edges

**Solutions:**
- Retake photo ensuring square is perpendicular
- Be more precise with corner clicking
- Use a larger square for better accuracy
- Take photo near image center (less distortion)

## Expected Results

Based on your automated calibration:
- **Expected aspect ratio:** ~0.985
- **Pixels are:** Slightly taller than wide (1.5% deviation)

Your manual measurement should be close to this value if:
- Square is truly perpendicular
- Corners are clicked accurately
- Physical size is measured correctly

## Tips for Best Results

1. **Perpendicular is Critical:** Use a tripod or ensure camera is level
2. **Center the Square:** Less lens distortion near image center
3. **Larger Square:** Easier to measure accurately
4. **Precise Clicking:** Zoom in mentally, click exact corners
5. **Good Lighting:** Easier to see corners clearly

## Troubleshooting

**Error: "Image not found"**
- Check the file path
- Make sure image is in the correct directory

**Window doesn't appear**
- Check if OpenCV GUI is working
- Try running from terminal, not IDE

**Results don't match automated**
- Most likely: square wasn't perpendicular
- Try retaking the photo
- Consider using a checkerboard square (you know it's flat)

## Next Steps

After completing this experiment:
1. Review the comparison with automated calibration
2. If results agree well, proceed to focal length experiment
3. If results differ significantly, retake measurements
4. Document your findings for the IEEE report
