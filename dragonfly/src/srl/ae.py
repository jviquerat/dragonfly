import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
            layers.Reshape(shape)
        ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ae():
    def __init__(self, dim, size):
        # Initialize from arguments
	self.obs_dim = dim
	self.buff_size = size
	
	# Initialize counter
	self.counter = 1
	
	# Create buffers
	self.names = ["obs"]
	self.sizes = [self.obs_dim]
	self.gbuff = gbuff(self.buff_size, self.names, self.sizes)
                
    def update(self):
        pass

    def process(self, obs):
        pass

    
