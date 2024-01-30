# Generic imports
import numpy as np
import random as rnd

# Custom imports
from dragonfly.src.srl.base            import *
from dragonfly.src.network.network     import *
from dragonfly.src.optimizer.optimizer import *
from dragonfly.src.loss.loss           import *

###############################################
### Variational autoencoder srl class
class vae(base_srl):
    def __init__(self, obs_dim, buff_size, pms):

        # Initialize from arguments
        self.obs_dim       = obs_dim
        self.buff_size     = buff_size
        self.latent_dim    = pms.latent_dim
        self.update_freq   = pms.update_freq
        self.batch_size    = pms.batch_size
        self.n_epochs      = pms.n_epochs
        self.n_updates     = pms.n_updates

        # Initialize network
        self.net = net_factory.create("vae",
                                      inp_dim = self.obs_dim,
                                      lat_dim = self.latent_dim,
                                      pms     = pms.network)
        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

        # Create buffers
        self.names = ["obs"]
        self.sizes = [self.obs_dim]
        self.gbuff = gbuff(self.buff_size, self.names, self.sizes)

        self.reset()

    # Reset
    def reset(self):

        self.gbuff.reset()
        self.net.reset()
        self.opt.reset()

        self.counter  = 0
        self.n_update = 0

    # Update vae
    def update(self):

        if (self.n_update >= self.n_updates): return

        print("UPDATE VAE")

        # Update
        for i in range(self.n_epochs):
            obs = self.gbuff.get_batches(["obs"], self.batch_size)["obs"]
            loss = self.loss.train(obs, self)

        # Write to file
        with open("vae_loss.dat", "a") as f:
            f.write(str(loss.numpy())+"\n")

        self.n_update += 1

    # Full network forward pass
    def forward(self, state):

        x, mean, std = self.net.call(state)
        x = x[0]

        return x, mean, std

    # Process raw observations
    def process(self, obs):

        # Check if it's the update time
        if ((self.gbuff.length() > 0) and (self.counter > self.update_freq)):
            self.update()
            self.counter = 0

        encoded = self.net.encoder(obs)[0].numpy()

        return encoded


# #@keras.saving.register_keras_serializable()
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a state."""

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.random.normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# #@keras.saving.register_keras_serializable()
# class Encoder(layers.Layer):
#     """Maps states to a triplet (z_mean, z_log_var, z)."""

#     def __init__(self, latent_dim, intermediate_dim):
#         super().__init__()
#         self.dense_proj = layers.Dense(intermediate_dim, activation="tanh")
#         self.dense_mean = layers.Dense(latent_dim, activation="tanh")
#         self.dense_log_var = layers.Dense(latent_dim, activation="tanh")
#         self.sampling = Sampling()

#     def call(self, inputs):
#         #x = self.dense_proj(inputs)
#         x = inputs
#         z_mean = self.dense_mean(x)
#         z_log_var = self.dense_log_var(x)
#         z = self.sampling((z_mean, z_log_var))
#         return z_mean, z_log_var, z


# #@keras.saving.register_keras_serializable()
# class Decoder(layers.Layer):
#     """Converts z, the encoded state, back into a state."""

#     def __init__(self, original_dim, intermediate_dim):
#         super().__init__()
#         self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
#         self.dense_output = layers.Dense(original_dim, activation="linear")

#     def call(self, inputs):
#         #x = self.dense_proj(inputs)
#         x = inputs
#         return self.dense_output(x)


# #@keras.saving.register_keras_serializable()
# class VariationalAutoEncoder(Model):
#     """Combines the encoder and decoder into an end-to-end model for training."""

#     def __init__(
#         self,
#         original_dim,
#         intermediate_dim,
#         latent_dim,
#     ):
#         super().__init__()
#         self.original_dim = original_dim
#         self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
#         self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

#     def call(self, inputs):
#         z_mean, z_log_var, z = self.encoder(inputs)
#         reconstructed = self.decoder(z)
#         # Add KL divergence regularization loss.
#         kl_loss = -0.5 * tf.reduce_mean(
#             z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
#         )
#         self.add_loss(kl_loss)
#         return reconstructed

