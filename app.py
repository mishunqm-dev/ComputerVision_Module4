import streamlit as st
from PIL import Image, ImageOps
from pathlib import Path

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="CSC 8830 Module 4",
    layout="wide"
)

st.title("CSC 8830 - Module 4: Human Boundary Detection")

st.write(
    "This project compares traditional OpenCV human boundary detection "
    "with SAM2 segmentation for RGB and thermal images."
)

# ---------------------------------------------------------
# FIND RGB IMAGE AUTOMATICALLY
# ---------------------------------------------------------

rgb_folder = Path("rgb_images")

rgb_files = (
    list(rgb_folder.glob("*.jpg"))
    + list(rgb_folder.glob("*.jpeg"))
    + list(rgb_folder.glob("*.png"))
)

if len(rgb_files) == 0:
    st.error("No RGB image was found inside the rgb_images folder.")
    st.stop()

rgb_path = rgb_files[0]

# Load RGB image and correct orientation metadata
rgb_original = Image.open(rgb_path)
rgb_original = ImageOps.exif_transpose(rgb_original)

# ---------------------------------------------------------
# RGB HUMAN BOUNDARY DETECTION
# ---------------------------------------------------------

st.header("RGB Human Boundary Detection")

rgb_boundary = Image.open("results/rgb_person_boundary.jpg")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original RGB Image")
    st.image(rgb_original, width="stretch")

with col2:
    st.subheader("OpenCV Boundary Result")
    st.image(rgb_boundary, width="stretch")

st.write(
    "The RGB boundary was detected using traditional OpenCV image processing. "
    "GrabCut segmentation and contour detection were used to isolate the human "
    "subject without using machine learning or deep learning."
)

# ---------------------------------------------------------
# THERMAL HUMAN BOUNDARY DETECTION
# ---------------------------------------------------------

st.header("Thermal Human Boundary Detection")

thermal_original = Image.open("thermal_images/thermal_person.jpg")
thermal_boundary = Image.open("results/thermal_person_boundary.jpg")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Original Thermal Image")
    st.image(thermal_original, width="stretch")

with col4:
    st.subheader("OpenCV Thermal Boundary Result")
    st.image(thermal_boundary, width="stretch")

st.write(
    "The thermal human boundary was detected using traditional OpenCV processing. "
    "Warm thermal colors were isolated using HSV thresholding, followed by "
    "morphological operations and contour detection to identify the standing person."
)

# ---------------------------------------------------------
# SAM2 COMPARISON
# ---------------------------------------------------------
st.header("SAM2 Comparison")

sam2_rgb = Image.open("results/sam2_rgb_result.png")
sam2_thermal = Image.open("results/sam2_thermal_result.png")

col5, col6 = st.columns(2)

with col5:
    st.subheader("SAM2 RGB Result")
    st.image(sam2_rgb, width="stretch")
    st.metric("SAM2 IoU Score", "0.7646")
    st.write(
        "The SAM2 result produced a good match on the RGB image. "
        "The person was segmented more cleanly than with the traditional "
        "OpenCV boundary method."
    )

with col6:
    st.subheader("SAM2 Thermal Result")
    st.image(sam2_thermal, width="stretch")
    st.metric("SAM2 IoU Score", "0.3021")
    st.write(
        "The SAM2 result produced a lower match on the thermal image. "
        "Thermal color patterns and the initial mask made segmentation "
        "more difficult than with the RGB image."
    )
    st.subheader("Comparison Summary")

st.write(
    "The RGB image achieved a higher SAM2 IoU score than the thermal image. "
    "Traditional OpenCV methods successfully detected the human boundary in both "
    "cases, while SAM2 performed substantially better on the RGB image than on "
    "the thermal image."
)
