import os
import numpy as np
import cv2
import tensorflow as tf
import random
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 38

# Load the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'E:\\Machine_Learning\\PlantVillage-Dataset-master\\data_distribution_for_SVM\\train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'E:\\Machine_Learning\\PlantVillage-Dataset-master\\data_distribution_for_SVM\\test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build the model using MobileNetV2
base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model with custom learning rate
learning_rate = 1e-5
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=EPOCHS, verbose=1)

# Evaluate the model on test dataset
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

# Save the model
model.save('plant_disease_detection.h5')


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

if __name__ == "__main__":
    # Load class labels
    with open('class_labels_updated.json', 'r') as f:
        class_labels = json.load(f)

    # Test the model with an image
    test_directory = 'E:\\Machine_Learning\\PlantVillage-Dataset-master\\data_distribution_for_SVM\\test'
    test_folder = random.choice(os.listdir(test_directory))
    test_image_folder = os.path.join(test_directory, test_folder)
    test_image_name = random.choice(os.listdir(test_image_folder))
    test_image_path = os.path.join(test_image_folder, test_image_name)
    prediction = predict(model, test_image_path)
    disease_name = class_labels[prediction]
    print("Predicted disease:", disease_name)
