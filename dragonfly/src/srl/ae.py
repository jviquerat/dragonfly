import numpy as np
import random as rnd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Custom imports
from dragonfly.src.srl.base import *


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.total_shape = tf.math.reduce_prod(shape)
        self.initializer = tf.keras.initializers.Ones()

        #input_layer = layers.Input(shape=self.shape)
        #self.encoder = layers.Conv1D(self.latent_dim, 3, activation='relu', padding='same')(input_layer)
        #encoded = layers.MaxPooling1D(2, padding='same')(self.encoder)
        #self.decoder = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(encoded)
        #self.decoder = layers.Reshape(self.shape)(self.decoder)
            
        #self.encoder = tf.keras.Sequential([
            #layers.Flatten(),
            #layers.Dense(total_shape, activation='relu'),
            #layers.Dense(int(total_shape/2), activation='relu'),
            #layers.Dense(int(total_shape/2), activation='relu'),
            #layers.Dense(latent_dim, activation='tanh')
            #layers.Reshape((total_shape,1)),
            #layers.Input(shape=(1,total_shape,1)),
            #layers.Conv1D(1, 3, padding='causal',
            #              kernel_initializer=initializer),
            #layers.Dense(latent_dim, activation='tanh'),
            #layers.Input(shape=(total_shape))
            #layers.Reshape(shape)
        #])
        #self.decoder = tf.keras.Sequential([
            #layers.Flatten(),
            #layers.Dense(int(total_shape/2), activation='relu'),
            #layers.Dense(total_shape, activation='relu'),
            #layers.Dense(int(total_shape/2), activation='relu'),
            #layers.Input(shape=self.shape),
            #layers.Dense(self.total_shape, activation='linear'),
            #layers.Conv1DTranspose(total_shape, 2, activation='linear'),
            #layers.Conv1D(1, 2, activation='linear'),
            #layers.Reshape(shape)
        #])

    def encoder(self,x):
        x = layers.Input(shape=(1,self.total_shape,1))
        encoded = layers.Conv1D(1, 3, activation='relu', padding='same')(x)
        #encoded = layers.MaxPooling1D(2, padding='same')(encoded)
        return encoded

    def decoder(self,x):
        x = layers.Conv1D(1, 3, activation='relu', padding='same')(x)
        #decoded = layers.UpSampling1D(2)(decoded)
        #output_layer = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(decoded)
        output_layer = layers.Reshape(self.shape)(x)
        return output_layer

    
    def call(self, x):
        encoded = self.encoder(x)
        print("en",encoded.shape)
        decoded = self.decoder(encoded)
        print(decoded)
        print("de",decoded.shape)
        return decoded


class ae():
    def __init__(self, dim, new_dim, freq, size):

        # Initialize from arguments
        self.obs_dim = dim
        self.buff_size = size
        self.reduced_dim = new_dim
        self.freq = freq

        # Initialize counter
        self.counter = 1

        # Create buffers
        self.names = ["obs"]
        self.sizes = [self.obs_dim]
        self.gbuff = gbuff(self.buff_size, self.names, self.sizes)

        # Initialize Autoencoder
        self.autoencoder = Autoencoder(new_dim, (dim,1))
        self.autoencoder.compile(optimizer='adam',
                                 loss=losses.MeanSquaredError())

    def update(self):

        # Get data
        obs = self.gbuff.get_buffers({"obs"},self.counter)["obs"]
        obs = obs.numpy()
        obs = obs[rnd.sample(range(obs.shape[0]),100),:]
        # Normalize data
        obs -= obs.mean(axis=0)
        std = obs.std(axis=0)
        index = np.where(std!=0)[0]
        obs[:,index] /= std[index]
        # Split data        
        n = obs.shape[0]
        n_test = int(n/7)
        x_train = obs[:n-n_test,:]
        x_test = obs[n-n_test:,:]

        # Train autoencoder
        self.autoencoder.fit(x_train, x_train,
                             epochs=1,
                             shuffle=True,
                             validation_data=(x_test, x_test),
                             batch_size=100)


    def process(self, obs):

        # Before the update time
        if (self.counter < self.freq) :
            return obs[:,:self.reduced_dim]

        # Check if it's the update time
        if ((self.counter % self.freq)==0) :
            self.update()

        encoded_obs = self.autoencoder.encoder(obs).numpy()
        #print(obs[:,0:3]/encoded_obs[:,0:3])
        #print(obs[:,[1,0,2]]/encoded_obs[:,[1,0,2]])
        return encoded_obs

    
