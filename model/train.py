import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator
import matplotlib.pyplot as plt  # For visualizing images


# Ensure TensorFlow uses CPU if no GPU is available
tf.config.set_visible_devices([], 'GPU')

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Build the model
model = models.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(10, activation='softmax')
])



# Set up ImageDataGenerator for image augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,    # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,  # Randomly shift images vertically by up to 20%
    shear_range=0.2,       # Shear transformation for images
    zoom_range=0.2,        # Random zoom
    fill_mode='nearest'    # Fill any empty pixels created by transformations
)

# Fit the data generator to your training data
train_datagen.fit(x_train)

# Generate a batch of augmented images
augmented_images = train_datagen.flow(x_train, y_train, batch_size=10)

# Get the first batch of augmented images
x_augmented, y_augmented = next(augmented_images)

# Plot 10 augmented images
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)  # Create a 2x5 grid for the images
    plt.imshow(x_augmented[i].reshape(28, 28), cmap='gray')
    plt.axis('off')  # Turn off axis for better visual appearance
    plt.title(f"Label: {y_augmented[i]}")
plt.tight_layout()
plt.show()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1)

model.save(f'model/mnist_model.h5')