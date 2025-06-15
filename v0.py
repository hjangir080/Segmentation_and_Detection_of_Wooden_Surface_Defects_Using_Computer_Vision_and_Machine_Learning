import cv2
import numpy as np

# Load the image
image = cv2.imread('data/Images/Images_1/99100020.jpg')

# Resize the image to a smaller size for easier viewing
resize_scale = 0.5  # Scale factor (e.g., 0.5 for half the original size)
width = int(image.shape[1] * resize_scale)
height = int(image.shape[0] * resize_scale)
resized_image = cv2.resize(image, (width, height))

# Crop unnecessary black areas from the left and right edges
crop_left = int(0.1 * width)  # 10% from the left
crop_right = int(0.1 * width)  # 10% from the right
cropped_image = resized_image[:, crop_left:width - crop_right]

# Convert cropped image to grayscale
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply global thresholding
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Morphological operations to close small gaps
kernel = np.ones((7, 7), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the cropped image to draw contours
output = cropped_image.copy()

# Minimum area threshold for detecting larger defects
min_area = 1000  # Set this to a value suitable for your images

# List to hold bounding box information
bounding_boxes = []

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > min_area:
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(cnt)

        # Normalize the coordinates
        x_center = (x + w / 2) / cropped_image.shape[1]
        y_center = (y + h / 2) / cropped_image.shape[0]
        width_norm = w / cropped_image.shape[1]
        height_norm = h / cropped_image.shape[0]

        # Append the bounding box in the required format (class 0 assumed)
        bounding_boxes.append(f"0 {x_center} {y_center} {width_norm} {height_norm}")

        # Draw the bounding box on the cropped image
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Print the bounding boxes
for box in bounding_boxes:
    print(box)

# Show the output
cv2.imshow('Cropped Image with Bounding Boxes', output)
cv2.imshow('Thresholded Image', thresh)
cv2.imshow('Morphological Operations', morph)
cv2.waitKey(0)
cv2.destroyAllWindows()