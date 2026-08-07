import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ------------------------
# Load Dataset
# -------------------------
digits = load_digits()

images = digits.images
labels = digits.target

# ------------------------
# Image Preprocessing
# ------------------------
processed_images = []

for img in images:
    #Convert to uint8
    img = (img * 16).astype(np.uint8)

    #Resize image
    img = cv2.resize(img, (32, 32))

    #Normalize
    img = img / 255.0

    processed_images.append(img)

processed_images = np.array(processed_images)

# ------------------------
# Prepare Data
# ------------------------
x = processed_images.reshape(len(processed_images), -1)
y = labels

# ------------------------
# Train/Test Split 
# ------------------------
x_train, x_test, y_train, y_test =train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------
# Train SVM Model
# ------------------------
model = SVC(kernel="linear")

model.fit(x_train, y_train)

# ------------------------
# Predictions
# ------------------------
Predictions = model.predict(x_test)

# ------------------------
# Accuracy
# ------------------------
accuracy = accuracy_score(y_test, Predictions)

print("Model Accuracy:", accuracy)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, Predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")


# -------------------------
# Show Predictions
# -------------------------
plt.figure(figsize=(12,6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i].reshape(32,32),cmap="gray")

    plt.title(f"P:{Predictions[i]}\nA:{y_test[i]}")

    plt.axis("off")

plt.tight_layout()

plt.show()