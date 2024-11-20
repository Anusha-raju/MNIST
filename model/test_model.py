import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Ensure TensorFlow uses CPU if no GPU is available
tf.config.set_visible_devices([], 'GPU')

# Load the model
model = tf.keras.models.load_model('/MNIST/model/mnist_model.h5')

# Load MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
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
    _, accuracy = model.evaluate(x_test, y_test)
    assert accuracy > 0.95, "Model accuracy is below 95%"

test_model()