## Autoencoders

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
