# MNIST DNN CI/CD Pipeline
[![CI/CD Pipeline](https://github.com/Anusha-raju/MNIST/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Anusha-raju/MNIST/actions/workflows/ci-cd.yml)

This project implements a basic Continuous Integration/Continuous Deployment (CI/CD) pipeline for a machine learning project using a Deep Neural Network (DNN) to classify handwritten digits from the MNIST dataset. The pipeline includes automated testing for the model, validation checks, and a simple deployment process.

## Project Structure


ml_project/

│

├── .github/

│ └── workflows/

│ └── ci-cd.yml

│

├── model/

│ ├── train.py

│ ├── test_model.py

│

├── requirements.txt

└── README.md



## Requirements

- Python 3.8 or higher
- TensorFlow
- NumPy
- Matplotlib

You can install the required packages using the following command:
```
pip install -r requirements.txt
```

## Dataset: MNIST
The MNIST (Modified National Institute of Standards and Technology) dataset is a collection of 28x28 grayscale images of handwritten digits (0-9). The dataset contains 60,000 training images and 10,000 test images.

## Image Augmentation Techniques
In this project, we use TensorFlow's ImageDataGenerator to apply a set of augmentation transformations to the training images. The following transformations are applied randomly to each image during the training process:

1. ***Rotation***: The image is randomly rotated within a range of ±20 degrees.

2. ***Width and Height Shift***: The image is randomly shifted horizontally (left or right) or vertically (up or down) by up to 20% of the total width/height.

3. ***Shear***: The image undergoes a shear transformation by up to 20%, causing it to be tilted along the X or Y axis.

4. ***Zoom***: The image is randomly zoomed in or out by up to 20%.

5. ***Fill Mode***: After applying any of the above transformations, any empty pixels (i.e., those created by shifting, rotating, or zooming) are filled using the nearest pixel value.


## Model Training

The model is defined in `train.py`, which includes the following steps:

1. Load the MNIST dataset.
2. Preprocess the data (reshape and normalize).
3. Image Augmentation (Rotation,Width and Height Shifts, Shear, Zoom, Fill Mode)
3. Build a 3-layer DNN with convolutional and fully connected layers.
4. Compile the model.
5. Train the model for 1 epoch.
6. Save the model.

### Running the Training Script

To train the model, run the following command:

```
python model/train.py
```


## Model Testing

The testing script `test_model.py` performs the following checks:

1. Validates the model's input shape to ensure it matches (28, 28, 1).
2. Validates the model's output shape to ensure it matches (10,) for digit classification.
3. Checks that the model has fewer than 25,000 parameters.
4. Evaluates the model's accuracy on the test set, ensuring it exceeds 95%.
5. Verifies the model's accuracy on the training set is above 95%.
6. Ensures the test loss is below 0.5 to avoid overfitting.
7. Compares the training and test accuracies to check for overfitting.
8. Confirms that model predictions are consistent for the same input.
9. Measures prediction time for 10 samples, ensuring it’s under 1 second.
10. Monitors the model's memory usage, ensuring it’s below 1 GB.

### Running the Testing Script

To test the model, run the following command:
```
python model/test_model.py
```


## CI/CD Pipeline

The CI/CD pipeline is configured using GitHub Actions. The workflow is defined in `.github/workflows/ci-cd.yml` and includes the following steps:

1. Checkout the code.
2. Set up Python.
3. Install dependencies.
4. Run the training script.
5. Run the testing script.

### Triggering the CI/CD Pipeline

The pipeline is triggered on every push to the `main` branch. You can monitor the status of the pipeline in the "Actions" tab of your GitHub repository.

## .gitignore

The project includes a `.gitignore` file to exclude unnecessary files from version control, such as:

- Python cache files
- Virtual environment directories
- TensorFlow model files

## Local Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd ml_project
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training script**:
   ```bash
   python model/train.py
   ```

5. **Run the tests**:
   ```bash
   python model/test_model.py
   ```

## Acknowledgments

- The MNIST dataset is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- TensorFlow and Keras for providing the deep learning framework.
