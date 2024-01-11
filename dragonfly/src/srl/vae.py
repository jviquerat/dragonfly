import numpy as np
import random as rnd
import tensorflow as tf
#import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

# Custom imports
from dragonfly.src.srl.base import *


#@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a state."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


#@keras.saving.register_keras_serializable()
class Encoder(layers.Layer):
    """Maps states to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim, intermediate_dim):
        super().__init__()
        self.dense_proj = layers.Dense(intermediate_dim, activation="tanh")
        self.dense_mean = layers.Dense(latent_dim, activation="tanh")
        self.dense_log_var = layers.Dense(latent_dim, activation="tanh")
        self.sampling = Sampling()

    def call(self, inputs):
        #x = self.dense_proj(inputs)
        x = inputs
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


#@keras.saving.register_keras_serializable()
class Decoder(layers.Layer):
    """Converts z, the encoded state, back into a state."""

    def __init__(self, original_dim, intermediate_dim):
        super().__init__()
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="linear")

    def call(self, inputs):
        #x = self.dense_proj(inputs)
        x = inputs
        return self.dense_output(x)


#@keras.saving.register_keras_serializable()
class VariationalAutoEncoder(Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim,
        latent_dim,
    ):
        super().__init__()
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


class vae():
    def __init__(self, pms, dim, size):

        # Initialize from arguments
        self.obs_dim = dim
        self.buff_size = size
        self.reduced_dim = pms.srl.reduced_dim
        self.freq = pms.srl.freq
        self.learning_rate = pms.srl.learning_rate
        self.batch_size = pms.srl.batch_size
        self.epochs = pms.srl.epochs

        self.reset()

    def reset(self):

        # Initialize counter
        self.counter = 1

        # Create buffers
        self.names = ["obs"]
        self.sizes = [self.obs_dim]
        self.gbuff = gbuff(self.buff_size, self.names, self.sizes)

        # Initialize Autoencoder
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.vae = VariationalAutoEncoder(latent_dim=self.reduced_dim,
                                          intermediate_dim=2*self.reduced_dim,
                                          original_dim=self.obs_dim)
        self.vae.compile(optimizer=opt,
                                 loss=losses.MeanSquaredError())

    def update(self):

        # Get data
        # select_size = self.freq-1
        select_size = self.batch_size
        obs = self.gbuff.get_buffers({"obs"},self.counter)["obs"]
        obs = obs.numpy()
        # mu  = obs.mean(axis=0)
        # std = obs.std(axis=0)
        obs = obs[rnd.sample(range(obs.shape[0]),select_size),:]

        # Update mu,std
        #self.mu_obs = 0.5*self.mu_obs + 0.5*obs.mean(axis=0)
        #self.std_obs = 0.5*self.std_obs + 0.5*obs.std(axis=0)

        # Normalize data
        # obs  = (obs - mu)/std
        # obs -= obs.mean(axis=0)
        # std = obs.std(axis=0)
        # index = np.where(std!=0)[0]
        # obs[:,index] /= std[index]

        # Split data
        n = obs.shape[0]
        n_test = int(n/5)
        x_train = obs[:n-n_test,:]
        x_test = obs[n-n_test:,:]

        # Train autoencoder
        self.vae.fit(x_train, x_train,
                             epochs=self.epochs,
                             shuffle=True,
                             validation_data=(x_test, x_test),
                             batch_size=self.batch_size)


    def process(self, obs):

        # Before the update time
        #if (self.counter < self.freq) :
        #    return obs[:,:self.reduced_dim]

        # Check if it's the update time
        if ((self.counter%self.freq)==0):
            self.update()

        _,_,encoded_obs = self.vae.encoder(obs)
        #print(obs[:,0:3]/encoded_obs[:,0:3])
        #print(obs[:,[1,0,2]]/encoded_obs[:,[1,0,2]])
        return encoded_obs.numpy()
   
