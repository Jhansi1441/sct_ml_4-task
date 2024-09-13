import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Define dataset path and parameters
dataset_directory = r'C:\Users\mukka\Downloads\Telegram Desktop\train'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Function to check for corrupted images
def check_images(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify the image is not corrupted
            except (IOError, SyntaxError) as e:
                print(f'Bad file: {file_path}')

# Check dataset directory for corrupted images
check_images(dataset_directory)

# Data augmentation and rescaling
data_gen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15  # Split dataset for training and validation
)

# Train data generator
train_data = data_gen.flow_from_directory(
    dataset_directory,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_data = data_gen.flow_from_directory(
    dataset_directory,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Visualize some images from the training set
def visualize_sample_images(train_data):
    images, labels = next(train_data)  # Fetch a batch of images and labels
    plt.figure(figsize=(10, 10))
    for i in range(9):  # Display the first 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f'Label: {np.argmax(labels[i])}')
        plt.axis('off')
    plt.show()

# Call the function to visualize images
visualize_sample_images(train_data)

# Check data loading
print(f"Found {train_data.samples} training images across {train_data.num_classes} classes.")
print(f"Found {validation_data.samples} validation images across {validation_data.num_classes} classes.")

# Model definition using MobileNetV2
input_layer = Input(shape=(224, 224, 3))
base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_layer)
base_model.trainable = False  # Freeze the base model

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')  # Adjust output based on the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Define TensorBoard and ModelCheckpoint callbacks
tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)
checkpoint_callback = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train the model with callbacks
training_history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20,
    verbose=1,
    callbacks=[tensorboard_callback, checkpoint_callback]  # Add callbacks here
)

# Save the final trained model
model.save('hand_gesture_recognition_model.keras')

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(validation_data)
print(f"Validation Accuracy: {val_acc:.2f}")
