import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

IMAGE_SIZE = 224

# Load the saved model
saved_model = load_model('plant_disease_detection.h5')

# Load class labels
with open('class_labels_updated.json', 'r') as f:
    class_labels = json.load(f)

def load_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(model, image_path):
    image = load_image(image_path)
    prediction = model.predict(image)
    class_index = np.argmax(prediction, axis=1)[0]
    class_label = str(class_index)
    return class_label

# Test the model on a new image
image_path = "E:/Machine_Learning/PlantVillage-Dataset-master/data_distribution_for_SVM/test/28/1a07b2ea-b97c-4ee5-bc55-cc91756c45c4.JPG"
prediction = predict(saved_model, image_path)
disease_name = class_labels[prediction]
print("Predicted disease:", disease_name)
