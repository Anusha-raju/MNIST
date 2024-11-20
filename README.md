# MNIST DNN CI/CD Pipeline
![Build Status](https://github.com/Anusha-raju/MNIST/actions/workflows/ci.yml/badge.svg)

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

│ └── requirements.txt

│

├── requirements.txt

└── README.md



## Requirements

- Python 3.8 or higher
- TensorFlow
- NumPy

You can install the required packages using the following command:
```
pip install -r requirements.txt
```



## Model Training

The model is defined in `train.py`, which includes the following steps:

1. Load the MNIST dataset.
2. Preprocess the data (reshape and normalize).
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

1. Validates the input shape of the model.
2. Validates the output shape of the model.
3. Checks that the model has fewer than 100,000 parameters.
4. Evaluates the model's accuracy and ensures it is above 80%.

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