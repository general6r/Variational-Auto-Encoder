In this project, you'll implement a variational autoencoder, a type of latent variable model. 

A variational autoencoder contains two components, an encoder and decoder. The encoder encodes any image into a latent space, with some tricks to ensure a smooth sampling 
distribution. The decoder is trained to reconstruct the input image given vectors sampled from the latent space. 
To train the model, we'll use a simple cross-entropy loss that directly compares the original image pixels with its reconstruction. 
We'll also have a regularization term to enforce a prior distribution on the learned distributions for each variable.
