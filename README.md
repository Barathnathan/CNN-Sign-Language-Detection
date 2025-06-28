# 🤖 Sign Language Alphabet Detection Using CNN

This project uses a **Convolutional Neural Network (CNN)** model to detect **American Sign Language (ASL)** alphabets from **A to Z**, including **J** and **Z**, based on uploaded hand sign images.

💼 This project was developed for **self-learning**, not as part of an academic submission.

---

## 📌 Features

- Detects all ASL alphabets from A to Z (including J and Z)
- Works with uploaded hand sign images (no webcam required)
- Trained on a folder-based ASL dataset (custom or from Kaggle)
- Uses TensorFlow/Keras for CNN implementation
- Image preprocessing using OpenCV

---

## 🛠️ Technologies Used

- Python
- TensorFlow & Keras
- NumPy
- OpenCV
- Google Colab / Jupyter Notebook

---

## 🧠 Model Architecture

- **Input Layer:** Grayscale hand sign image
- **Conv2D → ReLU → MaxPooling** (x2)
- **Flatten → Dense → Dropout**
- **Output Layer:** 26 softmax neurons (A to Z)



---

## 🧪 Dataset Description

- Folder-based dataset (one folder per letter: `A/`, `B/`, ..., `Z/`)
- Each folder contains multiple labeled images of hand signs
- Dataset includes **J** and **Z**
- Source: *Private or public dataset such as ASL Alphabet Kaggle dataset*


---

## 🔧 How to Use

### 1. Upload a Hand Sign Image
```python
from google.colab import files
uploaded = files.upload()
model.save("asl_model_az_29.h5")
print("✅ Model saved as 'asl_model_az_29.h5'")

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Load and preprocess image
img = cv2.imread(img_path)
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Load the model
model = load_model("asl_model_az_29.h5")

# Predict
prediction = model.predict(img)
predicted_index = np.argmax(prediction)

# Get label names
label_map = train_gen.class_indices
index_to_label = {v: k for k, v in label_map.items()}
predicted_label = index_to_label[predicted_index]

# Show result
plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

print("✅ Predicted letter or sign:", predicted_label)



By using this code you can upload your hand sign and the model will process and predict the output
