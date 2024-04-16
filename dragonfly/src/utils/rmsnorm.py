from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow as tf


class RMSNorm(Layer):
    def __init__(
        self,
        epsilon=1e-7,
        gamma_initializer="ones",
        gamma_regularizer=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(RMSNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        shape = (input_shape[-1],)
        self.gamma = self.add_weight(
            name="gamma",
            shape=shape,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
            trainable=True,
        )
        super(RMSNorm, self).build(input_shape)

    def call(self, inputs):
        rms = tf.sqrt(
            tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon
        )
        return self.gamma * inputs / rms

    def compute_output_shape(self, input_shape):
        return input_shape
