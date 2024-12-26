# classifier/custom_layers.py

import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionWeightedSum(Layer):
    def __init__(self, **kwargs):
        super(AttentionWeightedSum, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Computes attention weights and expands dimensions for multiplication.
        Args:
            inputs: Tensor of shape (batch_size, time_steps, 1)
        Returns:
            Tensor of shape (batch_size, time_steps, 1)
        """
        # Remove the last dimension: (batch_size, time_steps)
        score_vec_squeezed = tf.squeeze(inputs, axis=-1)

        # Apply softmax to get attention weights: (batch_size, time_steps)
        attention_weights = tf.nn.softmax(score_vec_squeezed, axis=1)

        # Expand dimensions to multiply with lstm_out: (batch_size, time_steps, 1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

        return attention_weights_expanded


class ReduceSumCustom(Layer):
    def __init__(self, **kwargs):
        super(ReduceSumCustom, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Sums the weighted LSTM outputs over the time_steps dimension.
        Args:
            inputs: Tensor of shape (batch_size, time_steps, units)
        Returns:
            Tensor of shape (batch_size, units)
        """
        return tf.reduce_sum(inputs, axis=1)
