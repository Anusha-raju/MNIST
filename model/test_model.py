import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time
import numpy as np
import psutil
import os
# Ensure TensorFlow uses CPU if no GPU is available
tf.config.set_visible_devices([], 'GPU')

# Load the model
model = tf.keras.models.load_model('model/mnist_model.h5')

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Run tests
def test_model():
    # Check input shape
    assert model.input_shape == (None, 28, 28, 1), "Input shape is incorrect"
    
    # Check output shape
    assert model.output_shape == (None, 10), "Output shape is incorrect"
    
    # Check number of parameters
    num_params = model.count_params()
    print("number of params is : ",num_params)
    assert num_params < 25000, "Model has more than 25000 parameters"
    
    # Evaluate accuracy
    test_loss, accuracy = model.evaluate(x_test, y_test)
    assert accuracy > 0.95, "Model accuracy is below 95%"

    # Check model evaluation on training data
    _, train_accuracy = model.evaluate(x_train, y_train)
    assert train_accuracy > 0.95, f"Model accuracy on training data is too low: {train_accuracy}"

   

    # Check if test loss is reasonably low (not overfitting)
    assert test_loss < 0.5, f"Test loss is too high: {test_loss}"

    # Check for overfitting by comparing train and test accuracies
    assert abs(train_accuracy - accuracy) < 0.1, f"Large gap between train and test accuracy: Train accuracy = {train_accuracy}, Test accuracy = {accuracy}"

    # Check for model output consistency (same input should produce same output)
    predictions_1 = model.predict(x_test[:10])
    predictions_2 = model.predict(x_test[:10])
    assert np.array_equal(np.argmax(predictions_1, axis=1), np.argmax(predictions_2, axis=1)), "Model predictions are not consistent"

    

    # Measure prediction time for 10 samples
    start_time = time.time()
    model.predict(x_test[:10])
    end_time = time.time()
    prediction_time = end_time - start_time
    assert prediction_time < 1, f"Prediction time is too high: {prediction_time} seconds"




    # Get the current memory usage of the model
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    assert memory_usage < 1000, f"Model memory usage is too high: {memory_usage} MB"



test_model()