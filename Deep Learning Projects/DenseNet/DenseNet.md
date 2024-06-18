## Implementation of DenseNet-BC/B/C on CIFAR-10 Dataset
#### Introduction
This project involves the implementation of the DenseNet architecture to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. DenseNet, or Dense Convolutional Network, is a type of convolutional neural network where each layer within a dense block is connected to every other layer in a feed-forward fashion. 
#### Dataset Preparation
The CIFAR-10 dataset was loaded using Keras' built-in functionality. The dataset was then split into training, validation, and test sets, with the training set further split to create a validation set based on the paper description: 

```Python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

x_train = x_train.astype("float32") / 255
x_test = x_test.astype('float32') / 255
x_val = x_val.astype("float32") / 255

mean = np.mean(x_train, axis=(0, 1, 2))
std = np.std(x_train, axis=(0, 1, 2))

x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
x_test = (x_test - mean) / std

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)
```

A few normalization steps are taken from the paper and data was prepared without data augementation (using dropout as suggested).  Data was normalized using the channel means and standard deviations, along with pushing the pixel values between 0 and 1.  The labels were one-hot encoded. 
#### Description of DenseNet Components with Reference to the Original Paper
The DenseNet (Dense Convolutional Network) architecture connects each layer to every other layer in a feed-forward fashion. DenseNet was introduced in the paper ["Densely Connected Convolutional Networks"](https://arxiv.org/abs/1608.06993) by Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. This connectivity pattern ensures maximum information flow between layers and mitigates the vanishing gradient problem, enabling the creation of very deep networks. Below is my detailed explanation of the key components of DenseNet as implemented in the provided code. 
##### Architecture Design: 
- **Input Layer**: This is where the network first receives the input data, such as images. It prepares the data for processing through subsequent layers.
- **Dense Blocks**: DenseNet is structured around dense blocks, each containing multiple layers. In these blocks, each layer receives feature maps from all preceding layers as input. This dense connectivity enhances feature reuse and promotes strong gradient flow during training.
- **Transition Layers**: Positioned between dense blocks, transition layers serve two main purposes. They reduce the spatial dimensions of feature maps through operations like batch normalization, 1×1 convolutions, and average pooling. They also adjust the number of feature maps, controlling model complexity while maintaining information richness.
- **Classifier**: At the end of DenseNet, a global average pooling layer aggregates spatial information from the feature maps into a single vector. This vector is then fed into a classifier, typically a dense layer with softmax activation for classification tasks, which outputs probabilities for different classes.
##### Input Layer

```Python
def input_layer(input_shape, bottleneck=True, compression=True, k=12):
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Activation('relu')(x)
    if bottleneck and compression:
        x = layers.Conv2D(2 * k, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    else:
        x = layers.Conv2D(16, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    return inputs, x
```

**Explanation**:
- **Input**: Accepts the shape of the input images.
- **Batch Normalization**: Standardizes the inputs to the layer.
- **Activation**: Applies ReLU activation to introduce non-linearity.
- **Convolution**:
    - **Bottleneck and Compression**: If both are enabled, a convolutional layer with `2 * k` filters is applied. This is consistent with the DenseNet paper, where bottleneck layers (1x1 convolutions) are used to reduce the number of input feature maps.
    - **Without Bottleneck or Compression**: Applies a standard convolutional layer with 16 filters.
##### Dense Unit
```Python
def dense_unit(x, k, bottleneck=True, data_augmentation=True, dropout_rate=0.2):
    if bottleneck:
        y = layers.BatchNormalization()(x)
        y = layers.Activation("relu")(y)
        y = layers.Conv2D(4 * k, kernel_size=1, padding="same", kernel_initializer='he_normal')(y)  # 1x1 Convolution
        if not data_augmentation:
            y = layers.Dropout(dropout_rate)(y)
    else:
        y = x

    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters=k, kernel_size=3, padding="same", kernel_initializer='he_normal')(y)
    if not data_augmentation:
        y = layers.Dropout(dropout_rate)(y)
    x = layers.concatenate([x, y])
    return x
```

**Explanation**:
- **Bottleneck**:
    - **Batch Normalization and ReLU**: Standardizes and applies non-linearity.
    - **1x1 Convolution**: Reduces the number of feature maps (bottleneck layer), decreasing computation.
    - **Dropout**: Optionally applies dropout for regularization.
- **Main Convolution**:
    - **Batch Normalization and ReLU**: Standardizes and applies non-linearity.
    - **3x3 Convolution**: Applies the main convolution with `k` filters.
    - **Dropout**: Optionally applies dropout.
- **Concatenation**: Concatenates the input `x` with the output `y` to maintain dense connectivity.
###### Why Bottleneck Layers?

The concept of bottleneck layers in DenseNet is crucial for improving computational efficiency. 
- **Purpose**: Bottleneck layers are introduced to reduce the number of input feature maps before a convolution operation. This reduction helps in decreasing the computational cost without significantly impacting the model's performance.
- **Implementation**: In the DenseNet architecture, a 1×1 convolution is used as a bottleneck layer before each 3×3 convolution. This 1×1 convolution reduces the dimensionality of the feature maps, making the subsequent 3×3 convolution less computationally expensive.
- **DenseNet-B**: When the DenseNet model includes bottleneck layers, it is referred to as DenseNet-B. In this variant, each 1×1 convolution produces `4k` feature maps, where `k` is the growth rate. The sequence of operations in a DenseNet-B layer is BN (Batch Normalization) -> ReLU -> Conv(1×1) -> BN -> ReLU -> Conv(3×3).

>*a 1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency. We find this design especially effective for DenseNet and we refer to our network with such a bottleneck layer, i.e., to the BN-ReLU-Conv(1× 1)-BN-ReLU-Conv(3×3) as DenseNet-B. In our experiments, we let each 1×1 convolution produce 4k feature-maps.* *- DenseNet paper*
##### Dense  Block
```Python
def dense_block(x, k, num_layers, bottleneck=True, data_augmentation=True, dropout_rate=0.2):
    for _ in range(num_layers):
        x = dense_unit(x, k, bottleneck, data_augmentation, dropout_rate)
    return x
```

**Explanation**:
- **Loop through Layers**: Applies `num_layers` dense units sequentially.
- **Dense Connectivity**: Each dense unit adds its output to the existing feature maps, ensuring that all layers are directly connected.
#### Transition Layers
```Python
def transition_layers(x, compression=True, compression_factor=0.5, data_augmentation=True, dropout_rate=0.2):
    if compression and (compression_factor <= 0 or compression_factor > 1):
        raise ValueError("Allowed Compression Factor: (0 < θ ≤ 1)")
    if compression:
        num_filters = int(x.shape[-1] * compression_factor)
    else:
        num_filters = x.shape[-1]
    y = layers.BatchNormalization()(x)
    y = layers.Conv2D(filters=num_filters, kernel_size=1, padding="same", kernel_initializer='he_normal')(y)
    if not data_augmentation:
        y = layers.Dropout(dropout_rate)(y)
    x = layers.AveragePooling2D()(y)
    return x
```

**Explanation**:
- **Compression**: Reduces the number of feature maps by a factor of `compression_factor`. According to the DenseNet paper, this helps reduce model complexity and improved model compactness.
- **Batch Normalization and ReLU**: Standardizes and applies non-linearity.
- **1x1 Convolution**: Adjusts the number of filters.
- **Dropout**: Optionally applies dropout.
- **Average Pooling**: Reduces the spatial dimensions of the feature maps, downsampling the resolution. 
###### What Is Compression?

Compression is another technique used in DenseNet to improve model compactness and efficiency.

- **Purpose**: Compression reduces the number of feature maps at the transition layers between dense blocks. This reduction decreases the model's size and computational requirements.
- **Implementation**: The number of feature maps after a dense block is reduced by a factor called the compression factor `θ`. 
- **DenseNet-C**: When the DenseNet model employs compression (θ < 1), it is referred to as DenseNet-C. In the provided implementation, `θ` is set to 0.5.

> *We refer the DenseNet with θ <1 as DenseNet-C, and we set θ = 0.5 in our experiment. When both the bottleneck and transition layers with θ < 1 are used, we refer to our model as DenseNet-BC. When θ = 1, the number of feature-maps across transition layers remains unchanged.* *
> *- DenseNet paper*

#### Implementation Details Of `DenseNet` Function
- **Configuration**:
    - **Number of Blocks**: DenseNet typically consists of three dense blocks.
    - **Layers per Block**: For a model with depth `L` and `B` blocks, each block contains `(L - 4) / (2 * B)` layers.
    - **Growth Rate `k`**: Determines the number of filters added per dense layer.
    - **Compression Factor `θ`**: Reduces the number of feature maps in transition layers.
    - **Model Variants**: DenseNet-B (with bottleneck), DenseNet-C (with compression), and DenseNet-BC (with both).
- **Architecture**:
    - **Input Layer**: A convolutional layer with `16` filters (or `2 * k` for DenseNet-BC) is applied to the input images.
    - **Dense Blocks**: Each block applies a series of dense units, where each unit consists of batch normalization, ReLU activation, and convolution operations.
    - **Transition Layers**: Between dense blocks, 1x1 convolution and 2x2 average pooling are used to downsample the feature maps.
    - **Global Average Pooling**: Applied at the end of the last dense block.
    - **Softmax Classifier**: Produces the final class predictions.
#### Training the Model
The model was trained using a learning rate schedule and callbacks for checkpointing, learning rate reduction, and learning rate scheduling. Training took place according to the original paper without any data augmentation or with dropout set to 0.2. : 

```Python
batch_size = 64 
epochs = 300
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate= 1e-1, momentum=0.9, nesterov=True, weight_decay=1e-4), metrics=['acc'])
```

#### Model Evaluation
**Due to limitations in Google Colab's GPU resources**, the session had to be terminated. However, the best model checkpoint was saved and later evaluated on the test dataset, achieving the following results:
##### Results
- **Test loss**: 0.5097
- **Test accuracy**: 85.72%
- **Error rate**: 14.28%
#### Outcome
The implementation of DenseNet on the CIFAR-10 dataset demonstrated the effectiveness of the architecture in achieving a high accuracy on image classification tasks. Despite the constraints of the computing environment, the use of model checkpointing ensured that the best model was saved and could be evaluated accurately. The final model achieved an impressive test accuracy of 85.72%, with an error rate of 14.28% even at epoch 25 out of 300 which shows it would improve further. 

