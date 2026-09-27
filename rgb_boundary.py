import cv2
import os

# Path to the RGB image
image_path = "rgb_images/rgb_person.jpg"

# Load the image
image = cv2.imread(image_path)

# Check if the image loaded correctly
if image is None:
    print("ERROR: RGB image could not be loaded.")
else:
    print("RGB image loaded successfully!")
    print("Image size:", image.shape)
    import numpy as np

# Create a copy of the original image
output = image.copy()

# Create an empty mask for GrabCut
mask = np.zeros(image.shape[:2], np.uint8)

# GrabCut models
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Rectangle around the main person
# Format: (x, y, width, height)
rect = (850, 1600, 900, 1800)

# Run GrabCut segmentation
cv2.grabCut(
    image,
    mask,
    rect,
    bgdModel,
    fgdModel,
    5,
    cv2.GC_INIT_WITH_RECT
)

# Convert GrabCut mask into a binary foreground mask
person_mask = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
    255,
    0
).astype("uint8")

# Clean the mask
kernel = np.ones((5, 5), np.uint8)
person_mask = cv2.morphologyEx(
    person_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)

# Find contours
contours, _ = cv2.findContours(
    person_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Draw the largest contour as the human boundary
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(
        output,
        [largest_contour],
        -1,
        (0, 255, 0),
        8
    )

# Save results
cv2.imwrite("results/rgb_person_mask.jpg", person_mask)
cv2.imwrite("results/rgb_person_boundary.jpg", output)

print("RGB human boundary detection completed!")
print("Results saved in the results folder.")