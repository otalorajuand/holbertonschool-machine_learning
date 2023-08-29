#!/usr/bin/env python3
"""This module contains the class RNNEncoder"""
import tensorflow as tf


class RNNEncoder(K.layers.Layer):
    """encode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """class constructor

        Args:
        - vocab is an integer representing the size of the input vocabulary
        - embedding is an integer representing the dimensionality of the
          embedding vector
        - units is an integer representing the number of hidden units in the
          RNN cell
        - batch is an integer representing the batch size
        """

        super().__init__()

        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       kernel_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a tensor of zeros

        Returns: a tensor of shape (batch, units) containing the initialized
                 hidden states
        """

        hidden_states = tf.zeros([self.batch, self.units], tf.float32)
        return hidden_states

    def call(self, x, initial):
        """
        Args:
        - x is a tensor of shape (batch, input_seq_len) containing the input
          to the encoder layer as word indices within the vocabulary
        - initial is a tensor of shape (batch, units) containing the initial
          hidden state

        Returns: outputs, hidden
        - outputs is a tensor of shape (batch, input_seq_len, units)
          containing the outputs of the encoder
        - hidden is a tensor of shape (batch, units) containing the last
          hidden state of the encoder
        """

        initial = self.initialize_hidden_state()
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded, initial_state=initial)

        return outputs, hidden
