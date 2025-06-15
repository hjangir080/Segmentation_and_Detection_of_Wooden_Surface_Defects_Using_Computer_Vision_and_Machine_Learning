import cv2
import numpy as np
import pandas as pd
# from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# Defect label mapping for output readability
defect_labels = {
    0: "Quartzity",
    1: "Live_Knot",
    2: "Marrow",
    3: "Resin",
    4: "Dead_Knot",
    5: "Knot_with_Crack",
    6: "Knot_Missing",
    7: "Crack"
}

# Load the data from CSV file
data = pd.read_csv('wood_defects_annotations.csv')

print(data['class'].value_counts())
import matplotlib.pyplot as plt

data['class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Defect Class')
plt.ylabel('Count')
plt.show()

# Drop the filename column as it is not needed for classification
data = data.drop(columns=['filename'])




# Separate features (bounding box coordinates) and target (class)
X = data[['x_center', 'y_center', 'width', 'height']]
y = data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Initialization and Hyperparameter Tuning
models = {
    "Random Forest": GridSearchCV(RandomForestClassifier(random_state=42),
                                  {'n_estimators': [100, 150], 'max_depth': [10, 20]}, cv=3),
    "SVM": GridSearchCV(SVC(random_state=42),
                        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, cv=3),
    "KNN": GridSearchCV(KNeighborsClassifier(),
                        {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}, cv=3),
    "Decision Tree": GridSearchCV(DecisionTreeClassifier(random_state=42),
                                  {'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}, cv=3)
}

# Train models and display classification reports
best_estimators = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    best_estimators[model_name] = model.best_estimator_
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[defect_labels[i] for i in range(len(defect_labels))]))
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")

folder_path = 'cv_images'

# Get all image file paths in the specified folder
image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.png'))]

# Process each image in the folder
for image_path in image_paths:
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        continue

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image at {image_path}")
        continue

    # Convert color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Crop 8% from both sides
    crop_width = int(width * 0.08)
    cropped_image = image[:, crop_width:width - crop_width]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

    # HSV Masking
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    median_color = np.median(hsv_image.reshape(-1, 3), axis=0)
    lower_bound = np.clip(median_color - 70, 0, 255)
    upper_bound = np.clip(median_color + 70, 0, 255)
    binary_mask = cv2.inRange(hsv_image, lower_bound.astype(int), upper_bound.astype(int))
    binary_output = cv2.bitwise_not(binary_mask)
    gray = cv2.bitwise_and(gray, gray, mask=binary_output)

    # Blurring and Thresholding
    blurred = cv2.GaussianBlur(gray, (7, 7), 10)
    _, global_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    combined_thresh = cv2.bitwise_and(global_thresh, otsu_thresh)
    combined_thresh = cv2.bitwise_and(combined_thresh, adaptive_thresh)

    # Morphological Operations and Contour Detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    morphed = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 500
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Convert to normalized center coordinates
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            bbox_width = w / width
            bbox_height = h / height

            # Prepare input data for classification
            feature_vector = [[x_center, y_center, bbox_width, bbox_height]]

            # Predict defect type using each model
            predictions = {model_name: best_estimators[model_name].predict(feature_vector)[0] for model_name in best_estimators}

            # Display detected defect and prediction for each model
            for i, (model_name, pred_label) in enumerate(predictions.items()):
                defect = defect_labels[pred_label]
                cv2.putText(cropped_image, f"{model_name}: {defect}", (x, y - 15 - i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255 - i * 60, i * 60), 1)

            # Draw bounding box around the detected defect
            cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(cropped_image)
    plt.title(f"Detected Defects in {image_path}")
    plt.axis('off')
    plt.show()