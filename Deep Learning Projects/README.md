## Project Description: Implementation of ResNet-20 on CIFAR-10 Dataset
### Objective
The goal of this project is to implement and train a ResNet-20 (Residual Network) model on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 50,000 training images and 10,000 test images. The ResNet architecture is chosen for its ability to improve the training of deep neural networks by addressing the vanishing gradient problem through residual learning.

### Model Architecture
ResNet-20 is designed using the following components:

- Input Layer: Accepts images of shape 32x32x3.
- ResNet Layers: Composed of multiple residual blocks, each containing convolutional layers, batch normalization, and ReLU activation functions.
- Convolutional Layers: Uses Conv2D with filters initialized using He normal initialization and regularization applied through L2 regularization.
- Batch Normalization: Applied after each convolutional layer to stabilize and accelerate training.
Activation Function: ReLU is applied after batch normalization.
- Average Pooling: Applied before the fully connected layer to reduce the spatial dimensions.
Fully Connected Layer: Uses Dense layer with softmax activation to output class probabilities.

### Training Process
Data Preprocessing
- Normalization: Pixel values of the images are scaled to the range [0, 1].
- One-hot Encoding: Class labels are one-hot encoded to facilitate training.
- ImageDataGenerator: Utilized to perform data augmentation, including horizontal flips and shifts in width and height. This helps in improving the generalization of the model.

Learning Rate Scheduler
- A custom learning rate schedule is defined to adjust the learning rate dynamically during training.
Starts at 1e-3 and decreases progressively based on the number of epochs.

Callbacks
- Model Checkpoint: Saves the best model based on validation accuracy.
- Learning Rate Scheduler: Adjusts learning rate during training.
- ReduceLROnPlateau: Reduces learning rate when the model stops improving.

### Model Compilation and Training
- Loss Function: Categorical cross-entropy.
- Optimizer: Adam optimizer with an initial learning rate set by the scheduler.
- Metrics: Model performance is evaluated using accuracy.

### Training Configuration
- depth 20
- Batch Size: 128
- Epochs: 200
- Steps per Epoch: Calculated as the ceiling value of the total training samples divided by the batch size.

### Evaluation
The final model achieved a accuracy of 90.65% on the CIFAR-10 test set, which is consistent with the expected performance of ResNet-20 on this dataset.

## Code Explanations
Based on: https://arxiv.org/abs/1512.03385 (Deep Residual Learning for Image Recognition). The key idea of the paper is to address the degradation problem in deep networks by introducing residual learning. The main innovation of ResNet is the use of residual blocks, which allow the network to learn residual functions with reference to the input layer. This helps in mitigating the vanishing gradient problem by allowing gradients to flow through the network directly via shortcut connections or skip connections. In the code, this is implemented using the add function to combine the output of the convolutional layers with the input: `x = add([x, y])`

Although the given implementation is a simplified version of ResNet (ResNet-20), more complex versions of ResNet use bottleneck blocks to reduce the number of parameters and computation. Batch normalization is extensively used after each convolutional layer, as suggested in the paper, to improve convergence and reduce training time


### Core Components
**Residual Block (`resnet_layer` function)**: The resnet_layer function creates a single layer used in the residual blocks:

1. Convolution Layer (Conv2D): Applies convolution with specified filters, kernel size, strides, and regularization.
2. Batch Normalization (BatchNormalization): Normalizes the activations to improve training speed and stability.
3. Activation Function (Activation): Applies ReLU activation to introduce non-linearity.
4. The function supports two configurations: applying the convolution first (conv_first=True) or applying batch normalization and activation before convolution(as mentioned in ResNet v2 paper).

**ResNet-20 Architecture (resnet_v1 function)**
The resnet_v1 function builds the ResNet-20 architecture using residual blocks:
1. Input Layer (Input): Defines the input shape for the model.
2. Initial Convolution: Applies a single convolution layer to the input.
3. Residual Blocks: Stacks multiple residual blocks, grouped into three main stages with increasing filter sizes and downsampling applied between stages.
4. Average Pooling (AveragePooling2D): Reduces the spatial dimensions.
5. Fully Connected Layer (Dense): Applies a dense layer with softmax activation for classification.

#### Explanation for stages or stacks: 

```Python
num_filters *= 2
y = resnet_layer(inputs=x, num_filters=num_filters, strides=2)
y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=2, activation=None, batch_normalization=False)
x = add([x, y])
x = Activation("relu")(x)
```
In the above code, or the second stage number of filters is doubled. This step typically happens when moving to a new stage of the network, where the spatial dimensions of the feature maps are reduced, and the number of channels (filters) is increased to capture more abstract representations. The `strides=2` parameter in the first convolution layer reduces the spatial dimensions of the feature maps by half (downsampling). Two convolution layers are applied sequentially, with batch normalization and activation functions in between (except the last activation). A `1x1` convolution with `strides=2` is applied to the shortcut connection to match the dimensions of the residual path `y`. This step ensures that the addition operation can be performed correctly since both `x` and `y` must have the same shape. The shortcut connection and the residual path are added together then ReLU activation is applied to the result. Same explanation for the next stage. 

```Python
for res_block in range(1, num_res_blocks):
    y = resnet_layer(inputs=x, num_filters=num_filters)
    y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
    x = add([x, y])
    x = Activation('relu')(x)
```
In the second part of the code, two convolution layers are applied sequentially, with batch normalization and activation functions in between (except the last activation) for each remaining `num_res_blocks`. Note that `strides=1` by default, so no downsampling occurs. The output of the residual path `y` is added to the input `x` (which in this case is the output of the previous residual block). ReLU activation is then applied to the result.

The first part handles the transition between stages by downsampling the feature maps and increasing the number of filters. This is done to capture more complex and abstract features as the network goes deeper. The second part represents standard residual blocks where the spatial dimensions remain unchanged, and the input is simply added to the residual function of two convolution layers. The same goes for the next stage!

## Sources
The implementation is based on the original ResNet paper by He et al.:
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770-778). Available: arXiv:1512.03385.

## Results and Conclusion
The ResNet-20 model effectively classified images from the CIFAR-10 dataset, achieving a final test accuracy of 90.65%. This demonstrates the strength of residual learning in training deep networks, which can achieve high performance on complex tasks. The model's ability to handle the vanishing gradient problem and its design for feature reuse through residual connections make it a powerful architecture for image recognition tasks. This project showcases the successful implementation and training of a deep residual network, providing insights into the practical application of cutting-edge neural network architectures.





