# CSC 8830 Module 4 - Human Boundary Detection

This project compares traditional OpenCV human boundary detection with SAM2 segmentation using RGB and thermal images.

## Methods

### RGB Image
- GrabCut segmentation
- Contour detection
- Human boundary extraction

### Thermal Image
- HSV color thresholding
- Morphological operations
- Contour detection
- Human boundary extraction

## SAM2 Comparison

- RGB SAM2 IoU Score: 0.7646
- Thermal SAM2 IoU Score: 0.3021

The RGB image produced a stronger SAM2 segmentation result than the thermal image.

## Web Application

The results are displayed using a Streamlit web application.

## Files

- `app.py` - Streamlit web application
- `rgb_boundary.py` - RGB boundary detection
- `thermal_boundary.py` - Thermal boundary detection
- `rgb_images/` - RGB input image
- `thermal_images/` - Thermal input image
- `results/` - OpenCV and SAM2 results
