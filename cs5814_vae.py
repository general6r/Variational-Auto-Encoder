import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, num_categories=10, latent_dimensionality=15):
        super(ConditionalVariationalAutoencoder, self).__init__()
        
        
        self.input_dim = input_dim  # Flattened input dimensionality
        self.latent_dimensionality = latent_dimensionality  # Dimensionality of latent space
        self.num_categories = num_categories  # Number of output categories
        
        # Define sizeds for intermediate hidden layers
        self.hidden_size1 = 350
        self.hidden_size2 = 175
        self.hidden_size3 = 80
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_categories, self.hidden_size1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, self.hidden_size3),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(self.hidden_size3, self.latent_dimensionality)
        self.logvariance_layer = nn.Linear(self.hidden_size3, self.latent_dimensionality)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dimensionality + num_categories, self.hidden_size3),
            nn.ReLU(),
            nn.Linear(self.hidden_size3, self.hidden_size2),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, self.hidden_size1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, input_dim),
            nn.Sigmoid()
        )        
#         self.hidden_size = None  # Hidden layer size
#         self.encoder = None  # Encoder network
#         self.mean_layer = None  # Layer to compute mean of latent distribution
#         self.logvariance_layer = None  # Layer to compute log variance of latent distribution
#         self.decoder = None  # Decoder network
        
#         # In this section, you'll define the encoder network.
#         # The encoder takes a batch of input data (N, input_dim) and converts it
#         # to a latent representation of shape (N, latent_dim).
#         # The mean and log variance layers estimate the parameters of the latent
#         # distribution using the encoded features.
#         # You may need to reshape the input and concatenate additional information.
#         # ENCODER IMPLEMENTATION GOES HERE (create this in an nn.Sequential(...)).
#         pass
        
#         # Below, you'll define the decoder network. The decoder takes N samples
#         # from the latent space (N, latent_dim + num_categories) and reconstructs
#         # the input data with the same dimensionality as the original input.
#         # DECODER IMPLEMENTATION GOES HERE (create this in an nn.Sequential(...)).
#         pass


    def forward(self, data_input, condition):
        """
        The data_input var is a batch of images
        The condition variable are one-hot vectors

        Returns:
        Your function needs to return the reconstructed images, the mean
        - data_recon: Reconstructed data of shape (N, data_dim)
        - mean_params: Predicted means
        - logvariance_params: Predicted log variances
        """
        data_input = torch.flatten(data_input, start_dim=1)
#         data_input = data_input.view(-1, 1*28*28)
        # Concatenate the input and condition
        combined_input = torch.cat([data_input, condition], dim= 1)
#         print(f"combined_input shape is: {combined_input.shape}")

        # Encoder step
        encoded = self.encoder(combined_input)
#         print(f"encoded shape is: {encoded.shape}")
        mean_params = self.mean_layer(encoded)
        logvariance_params = self.logvariance_layer(encoded)
        
        # Reparameterization
        std = torch.exp(0.5 * logvariance_params)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean_params)
        
        # Decoder
        z_cond = torch.cat([z, condition], dim= 1)
#         print(f"z_cond shape is: {z_cond.shape}")
        data_recon = self.decoder(z_cond)
        data_recon = data_recon.reshape((-1, 1, 28, 28))
        
        return data_recon, mean_params, logvariance_params

#         data_recon = None
#         mean_params = None
#         logvariance_params = None
        
#         # To code this up, you should run the concatenated data_input and condition through the encoder
#         # Use the reparameterization trick to compute the latent representation
#         # Use the decoder to reconstruct the inputs

#         return data_recon, mean_params, logvariance_params


def sample_latent(means, logvariances):
    """
    Use the reparameterization trick to get latents

    Returns:
    - latent_samples: Tensor containing latents (N, latent_dim)
    """
    std = torch.exp(0.5 * logvariances)
    eps =torch.randn_like(std)
    
    return eps.mul(std).add_(means)
#     latent_samples = None

#     # Implement the reparameterization trick by:
#     # (1) Generating random noise from a standard normal distribution
#     # (2) Scaling the noise by the square root of the variance parameters
#     # (3) Shifting the scaled noise by the mean parameters

#     return latent_samples

def loss(reconstructions, originals, means, logvariances):
    """
    Computes the loss for training the latent variable model.
    
    Returns:
    - loss: Loss value for the latent variable model
    """
    # Reconstruction loss
    BCE = F.binary_cross_entropy(reconstructions, originals, reduction='sum')

    # Kullback-Leibler divergence
    KLD = -0.5 * torch.sum(1 + logvariances - means.pow(2) - logvariances.exp())

    return BCE + KLD
#     loss = None

#     # Compute the loss for the latent variable model, which includes:
#     # (1) A reconstruction loss between the original and reconstructed data
#     # (2) A regularization term on the latent distribution parameters

#     return loss
