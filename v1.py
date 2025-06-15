import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Load the image
image_path = 'data/Images/Images_1/99300003.jpg'
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Calculate 5% of the width for cropping
crop_width = int(width * 0.08)

# Crop 5% from both left and right sides
cropped_image = image[:, crop_width:width - crop_width]
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Step 1: Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 10)

# Step 2: Global Thresholding
_, global_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Step 3: Otsu’s Thresholding
_, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
# Step 4: Combine the results of both thresholding methods (Intersection)
combined_thresh = cv2.bitwise_and(global_thresh, otsu_thresh)
combined_thresh = cv2.bitwise_and(combined_thresh, adaptive_thresh)

# Step 5: Morphological Operations on the combined result
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
morphed = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)


# Step 6: Contour Detection on the refined result
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 7: Filter contours based on size to avoid small noise detections
min_contour_area = 500  # Adjust based on your requirements
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

# Step 8: Draw filtered contours on the cropped image for visualization
filtered_contours_img = cropped_image.copy()
cv2.drawContours(filtered_contours_img, filtered_contours, -1, (0, 255, 0), 2)

# Step 9: Apply Local Binary Patterns (LBP)
lbp_image = local_binary_pattern(gray, P=8, R=1, method="uniform")

# Step 10: Draw bounding boxes and label detected defects on the cropped image
result_image = cropped_image.copy()
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    defect_region = gray[y:y + h, x:x + w]
    lbp = local_binary_pattern(defect_region, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist = hist.astype("float") / (hist.sum() + 1e-6)  # Normalize histogram

    # Placeholder defect classification based on histogram analysis
    if hist[1] > 0.3:
        defect_type = "Dead Knot"
        color = (0, 0, 255)
    elif hist[2] > 0.3:
        defect_type = "Live Knot"
        color = (0, 255, 0)
    else:
        defect_type = "Knot with Crack"
        color = (0, 255, 255)

    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(result_image, defect_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display all processing steps
steps = [
    ("Cropped Image", gray),
    ("Gaussian Blurred", blurred),
    ("Global Thresholding", global_thresh),
    ("Otsu Thresholding", otsu_thresh),
    ("Combined Threshold", combined_thresh),
    ("Morphological Closing", morphed),
    ("Filtered Contours", filtered_contours_img),
    ("LBP Image", lbp_image),
    ("Final Result with Defects", result_image)
]

# Display results step-by-step
plt.figure(figsize=(18, 12))
for i, (title, img) in enumerate(steps, 1):
    plt.subplot(3, 3, i)
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()