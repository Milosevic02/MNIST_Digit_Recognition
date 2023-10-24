# MNIST Digit Recognition using TensorFlow

This project demonstrates the use of TensorFlow to build and train a neural network for recognizing handwritten digits from the MNIST dataset. The goal of the project is to correctly classify images of handwritten digits into their respective classes (0 to 9).

## Dataset

The MNIST dataset contains 28x28 grayscale images of handwritten digits and their corresponding labels. The data is divided into a training set and a test set, where the training set is used to train the model, and the test set is used to evaluate its performance.

## Model Architecture

The neural network used for digit recognition consists of the following layers:

1. Input Layer: Flattens the 28x28 image into a 1D vector.
2. Hidden Layer: A fully connected layer with 128 units and ReLU activation function.
3. Dropout Layer: Regularization technique to reduce overfitting (20% dropout rate).
4. Output Layer: A fully connected layer with 10 units (for 10 classes) and Softmax activation function.

## Model Training

The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. It is then trained on the training data with 10 epochs.

## Evaluation

The performance of the trained model is evaluated on the test set, and the confusion matrix is plotted to visualize the classification results. Additionally, a randomly misclassified digit is displayed along with its true and predicted labels.

## Model Accuracy

The model achieves an impressive accuracy of approximately 98% on both the training and test datasets, showcasing its strong performance in recognizing handwritten digits from the MNIST dataset.

## Requirements

To run this code, you need to have the following dependencies installed:

- TensorFlow (version 2.x)
- NumPy
- Matplotlib
- Scikit-learn

You can install these packages using `pip`:

```
pip install tensorflow numpy matplotlib scikit-learn
```


 
