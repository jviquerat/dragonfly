import numpy as np
import random as rnd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# Custom imports
from dragonfly.src.srl.base import *


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.total_shape = tf.math.reduce_prod(shape)
        #self.initializer = tf.keras.initializers.Ones()

        self.encoder = tf.keras.Sequential([
            #layers.Flatten(),
            #layers.Dense(total_shape, activation='relu'),
            #layers.Dense(int(total_shape/2), activation='relu'),
            #layers.Dense(int(total_shape/2), activation='relu'),
            layers.Dense(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            #layers.Dense(int(total_shape/2), activation='relu'),
            #layers.Dense(total_shape, activation='relu'),
            #layers.Dense(int(total_shape/2), activation='relu'),
            layers.Dense(self.total_shape, activation='linear'),
            #layers.Input(shape=self.shape)
        ])

    '''#Convolution 1D
    def encoder(self,x):
        x = layers.Input(shape=(1,self.total_shape,1))
        encoded = layers.Conv1D(1, 3, activation='relu', padding='same')(x)
        #encoded = layers.MaxPooling1D(2, padding='same')(encoded)
        return encoded

    def decoder(self,x):
        x = layers.Conv1D(1, 3, activation='relu', padding='same')(x)
        #decoded = layers.UpSampling1D(2)(decoded)
        #output_layer = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(decoded)
        output_layer = tf.reshape(x,self.shape)
        return output_layer
    '''
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ae():
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
        self.autoencoder = Autoencoder(self.reduced_dim, (self.obs_dim,1))
        self.autoencoder.compile(optimizer=opt,
                                 loss=losses.MeanSquaredError())

    def update(self):

        # Get data
        # select_size = self.freq-1
        select_size = self.batch_size
        obs = self.gbuff.get_buffers({"obs"},self.counter)["obs"]
        obs = obs.numpy()
        mu  = obs.mean(axis=0)
        std = obs.std(axis=0)
        obs = obs[rnd.sample(range(obs.shape[0]),select_size),:]

        # Update mu,std
        #self.mu_obs = 0.5*self.mu_obs + 0.5*obs.mean(axis=0)
        #self.std_obs = 0.5*self.std_obs + 0.5*obs.std(axis=0)

        # Normalize data
        obs  = (obs - mu)/std
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
        self.autoencoder.fit(x_train, x_train,
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

        encoded_obs = self.autoencoder.encoder(obs).numpy()
        #print(obs[:,0:3]/encoded_obs[:,0:3])
        #print(obs[:,[1,0,2]]/encoded_obs[:,[1,0,2]])
        return encoded_obs
