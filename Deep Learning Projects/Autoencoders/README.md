## Autoencoders

A full tutorial on Autoencoders can be found here on my website. [Autoencoders In Depth](https://neuraldemy.com/autoencoders-fundamentals-of-encoders-and-decoders/#denoising-autoencoders)

Autoencoders are a type of artificial neural network used for unsupervised learning of efficient codings. They are designed to learn a compressed representation (encoding) of input data, and subsequently, to reconstruct the original input from this encoding. This process forces the network to capture the most important features of the data.

### Structure of Autoencoders

An autoencoder consists of two main parts:

1. **Encoder**: This part of the network compresses the input into a latent-space representation. The encoder function $\( h = f(x) \)$ maps the input $\( x \)$ to a latent representation $\( h \)$.

2. **Decoder**: This part of the network reconstructs the input from the latent representation. The decoder function $\( \hat{x} = g(h) \)$ maps the latent representation $\( h \)$ back to the input space $\( \hat{x} \)$.

The overall structure can be represented as:
$\[ x \rightarrow \text{Encoder} \rightarrow h \rightarrow \text{Decoder} \rightarrow \hat{x} \]$

### Loss Function

The training of an autoencoder involves minimizing the reconstruction error, which measures how well the reconstructed input $\( \hat{x} \)$ matches the original input $\( x \)$. Common loss functions include:

- **Mean Squared Error (MSE)**: $\( L(x, \hat{x}) = \| x - \hat{x} \|^2 \)$
- **Binary Cross-Entropy**: Typically used for binary data.

## Code Explanation In Autoencoders.ipynb:

The code demonstrates how to build, train, and visualize a convolutional autoencoder using the MNIST dataset. The autoencoder compresses the input images into a lower-dimensional latent space and then reconstructs the images from this representation. This process forces the network to learn a compressed yet meaningful representation of the data.

### Regular Convolution (Conv2D)

Regular convolutional layers in neural networks, specifically for 2D data like images (Conv2D), work by sliding a filter (kernel) across the input to capture local features. This typically shrinks the image size (downsamples).

* **Sliding Window:** The kernel (filter) slides over the input image, typically moving one pixel at a time (stride).
* **Element-wise Multiplication and Summation:** At each position, the element-wise multiplication between the input and the kernel is performed, and the results are summed up to produce a single value in the output feature map.
* **Stride:** The number of pixels by which the kernel moves across the input image.
* **Downsampling:** The convolution operation often results in a smaller output (downsampled) due to the stride and padding used.

### Transpose Convolution (Conv2DTranspose)

A Conv2D Transpose layer does the opposite of a regular convolution layer. It takes an input and expands it to a larger output. 

* **Flipped Operation:** Unlike a regular convolution, the process involves virtually "padding" the input with zeros (or other values) and then sliding the kernel across this expanded input.
* **Input Padding:** Zeros are inserted between the elements of the input feature map to increase its spatial dimensions.
* **Element-wise Multiplication and Summation:** Similar to a regular convolution, at each position where the input overlaps the kernel, element-wise multiplication and summation are performed.
* **Output Construction:** The results of these multiplications for a specific position on the output are summed up to create a single value in the output map. This process is repeated for all positions on the output, resulting in a larger image.
* **Kernel Size and Stride:** The size of the output image can be controlled by the kernel size and stride of the Conv2D Transpose layer. A larger kernel size or smaller stride will generally lead to a larger output image.

#### Encoder

The encoder compresses the input image into a lower-dimensional latent representation. This part consists of Conv2D layers which downsample the input image:

```Python
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x = Conv2D(filters=32, kernel_size=3, activation="relu", strides=2, padding='same')(x)
x = Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
shape = K.int_shape(x)
x = Flatten()(x)
latent = Dense(latent_dim, name="latent_vector")(x)
```
#### Decoder Part

The decoder's job is to take this latent representation and reconstruct the original image. This is where Conv2DTranspose layers come into play. They upsample the lower-dimensional latent representation back to the original image dimensions.

 - Latent Input: The decoder takes the latent vector as input.
 - Dense Layer: The latent vector is first passed through a Dense layer to expand it back to the shape before the Flatten layer in the encoder. This matches the shape that the Conv2DTranspose layers will expect.
 - Reshape Layer: The output of the Dense layer is reshaped to match the shape of the last Conv2D layer's output in the encoder.
 - Conv2DTranspose Layers: These layers upsample the feature maps to gradually reconstruct the original input size.
 - Output Layer: The final Conv2DTranspose layer produces the output image with a single channel and the same size as the input.

```Python
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
x = Conv2DTranspose(filters=32, kernel_size=3, activation="relu", strides=2, padding="same")(x)
outputs = Conv2DTranspose(filters=1, kernel_size=3, activation="sigmoid", padding="same", name="decoder_output")(x)

decoder = Model(latent_inputs, outputs, name="decoder")
decoder.summary()
plot_model(decoder, show_shapes=True)
```
## Code Explanation In Denoising_Autoencoders.ipynb
Denoising Autoencoders take the above concept a step further by adding noise to the input data and training the model to remove this noise. This helps in learning more robust features and representations. 
1. **Structure:** Noisy Input: A noisy version of the input data is fed into the encoder. Encoder: The encoder compresses this noisy input into a latent representation. Decoder: The decoder reconstructs the clean, original input from this noisy latent representation.
2. **Objective:** The goal is to minimize the difference between the clean original input and the reconstructed output. This encourages the model to learn to denoise the input.
3. **Steps to Train a Denoising Autoencoder:** Add Noise: Add some form of noise (e.g., Gaussian noise) to the original data. For example, if you have an image dataset, you can add random pixel noise to create a noisy version of the images. Model Training: Train the autoencoder on pairs of noisy and clean data. The input to the encoder is the noisy data, and the target output for the decoder is the clean data. Loss Function: Use a loss function that measures the reconstruction error, such as Mean Squared Error (MSE) between the clean data and the reconstructed output.

In the code, we have done the same thing on MNIST Daatset. 
