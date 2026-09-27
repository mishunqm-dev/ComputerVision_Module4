import cv2
import numpy as np

# Path to thermal image
image_path = "thermal_images/thermal_person.jpg"

# Load thermal image
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Thermal image could not be loaded.")
else:
    print("Thermal image loaded successfully!")
    print("Image size:", image.shape)
   # ---------------------------------------------------------
# THERMAL HUMAN BOUNDARY DETECTION
# ---------------------------------------------------------

# Make a copy for drawing the final boundary
output = image.copy()

# Select the region containing the standing person
x1, y1 = 145, 10
x2, y2 = 275, 285

person_region = image[y1:y2, x1:x2]

# Convert the person region to HSV
hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)

# Detect warm thermal colors
lower_warm = np.array([0, 70, 70])
upper_warm = np.array([45, 255, 255])

mask = cv2.inRange(hsv, lower_warm, upper_warm)

# Clean small gaps and noise
kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

# Find contours inside the selected person region
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Move contour coordinates back to the original image
    largest_contour[:, :, 0] += x1
    largest_contour[:, :, 1] += y1

    # Draw green boundary
    cv2.drawContours(
        output,
        [largest_contour],
        -1,
        (0, 255, 0),
        3
    )

    print("Standing thermal person boundary detected successfully!")

else:
    print("No thermal person contour detected.")

# Save results
cv2.imwrite("results/thermal_person_mask.jpg", mask)
cv2.imwrite("results/thermal_person_boundary.jpg", output)

print("Thermal boundary results saved successfully!")