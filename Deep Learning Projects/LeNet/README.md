## Building a LeNet-5 Model for MNIST Handwritten Digit Classification

### Project Overview

This project implements a convolutional neural network (CNN) called LeNet-5 to classify handwritten digits from the MNIST dataset. LeNet-5, a pioneering CNN architecture, achieves high accuracy despite its relatively simple design. My implementation is based on modern practices. Some modern optimizers and activations were not available at that time. 

### Data

- **Dataset**: MNIST dataset, consisting of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels.
- **Preprocessing**:
  - Images are resized to 32x32 pixels and normalized to a range of 0 to 1.
  - Labels are one-hot encoded for multi-class classification.

### Model Architecture

#### LeNet-5 Architecture

- **Convolutional Layers**:
  - Two convolutional layers with tanh activation functions extract spatial features.
  - Max pooling layers for downsampling and reducing the computational cost.

- **Fully Connected Layers**:
  - A fully connected layer with tanh activation processes the extracted features.
  - Another fully connected layer with softmax activation outputs probabilities for each digit class (0-9).

### Training

- **Model Compilation**:
  - Optimizer: Adam optimizer.
  - Loss Function: Sparse categorical cross-entropy.
  - Metric: Accuracy.

- **Training Setup**:
  - Early stopping with a patience of 5 epochs to prevent overfitting.
  - Trained on the training data for 20 epochs with a batch size 32.

### Evaluation

- **Performance**:
  - Final test accuracy: ~98.85%.

### Deviations from Original LeNet-5

- The implementation deviates in:
  - Connection scheme after the first downsampling layer. The original model used a non-complete connection scheme after the first downsampling layer, which is not implemented here.
  - The final layer in the original architecture employed a radial basis function (RBF) activation and a different loss function. This project uses softmax activation and sparse categorical cross-entropy loss.

### Conclusion

This project successfully builds and trains a LeNet-5 model for MNIST handwritten digit classification, showcasing the effectiveness of this classic CNN architecture.

### Source
The source for the original LeNet-5 paper is:
- Title: [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- Authors: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
