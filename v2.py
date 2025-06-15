import cv2
import numpy as np
import matplotlib.pyplot as plt
# from skimage.feature import local_binary_pattern

# Load the image
image_path = 'wood_test.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
height, width = image.shape[:2]

# Calculate 8% of the width for cropping
crop_width = int(width * 0.08)

# Crop 8% from both left and right sides
cropped_image = image[:, crop_width:width - crop_width]
gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

# Step 1: HSV Color Masking to isolate areas within color range
hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
median_color = np.median(hsv_image.reshape(-1, 3), axis=0)

# Set lower and upper bounds for masking
lower_bound = np.clip(median_color - 70, 0, 255)
upper_bound = np.clip(median_color + 70, 0, 255)

# Create and invert binary mask
binary_mask = cv2.inRange(hsv_image, lower_bound.astype(int), upper_bound.astype(int))
binary_output = cv2.bitwise_not(binary_mask)

# Combine binary mask with the grayscale image
gray = cv2.bitwise_and(gray, gray, mask=binary_output)

# Step 2: Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 10)

# Step 3: Global Thresholding
_, global_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Step 4: Otsu’s Thresholding
_, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 5: Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Step 6: Combine the results of all thresholding methods (Intersection)
combined_thresh = cv2.bitwise_and(global_thresh, otsu_thresh)
combined_thresh = cv2.bitwise_and(combined_thresh, adaptive_thresh)

# Step 7: Morphological Operations on the combined result
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
morphed = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)

# Step 8: Contour Detection on the refined result
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 9: Filter contours based on size to avoid small noise detections
min_contour_area = 500  # Adjust based on your requirements
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

# Step 10: Draw filtered contours on the cropped image for visualization
filtered_contours_img = cropped_image.copy()
cv2.drawContours(filtered_contours_img, filtered_contours, -1, (0, 255, 0), 2)

# Step 11: Apply Local Binary Patterns (LBP)
# lbp_image = local_binary_pattern(gray, P=8, R=1, method="uniform")

# # Step 12: Draw bounding boxes and label detected defects on the cropped image
result_image = cropped_image.copy()
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    defect_region = gray[y:y + h, x:x + w]
    # lbp = local_binary_pattern(defect_region, P=8, R=1, method="uniform")
#     hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
#     hist = hist.astype("float") / (hist.sum() + 1e-6)  # Normalize histogram

    # #  defect classification based on histogram analysis
    # if hist[1] > 0.3:
    #     defect_type = "Dead Knot"
    #     color = (0, 0, 255)
    # elif hist[2] > 0.3:
    #     defect_type = "Live Knot"
    #     color = (0, 255, 0)
    # else:
    #     defect_type = "Knot with Crack"
    #     color = (0, 255, 255)

    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0,255,0), 2)
    # cv2.putText(result_image, defect_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display all processing steps
steps = [
    ("Cropped Image", gray),
    ("Binary Mask (Inverted)", binary_output),
    ("Gaussian Blurred", blurred),
    ("Global Thresholding", global_thresh),
    ("Otsu Thresholding", otsu_thresh),
    ("Combined Threshold", combined_thresh),
    ("Morphological Closing", morphed),
    ("Filtered Contours", filtered_contours_img),
    # ("LBP Image", lbp_image),
    ("Final Result with Defects", result_image)
]

# Display results step-by-step
plt.figure(figsize=(18, 12))
for i, (title, img) in enumerate(steps, 1):
    plt.subplot(3, 4, i)
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()