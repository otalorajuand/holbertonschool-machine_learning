#!/usr/bin/env python3
"""This module contains the class DecoderBlock"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """creates an encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor

        Args:
        - dm: the dimensionality of the model
        - h: the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - drop_rate: the dropout rate
        """
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Args:
        - x: a tensor of shape (batch, target_seq_len, dm)containing the input
             to the decoder block
        - encoder_output: a tensor of shape (batch, input_seq_len, dm)
                          containing the output of the encoder
        - training: a boolean to determine if the model is training
        - look_ahead_mask: the mask to be applied to the first multi head
                           attention layer
        - padding_mask: the mask to be applied to the second multi head
                        attention layer

        Returns: a tensor of shape (batch, target_seq_len, dm) containing
                 the block's output
        """
        attn_output_1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn_output_1 = self.dropout1(attn_output_1, training=training)
        out1 = self.layernorm1(x + attn_output_1)

        attn_output_2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn_output_2 = self.dropout2(attn_output_2, training=training)
        out2 = self.layernorm2(out1 + attn_output_2)

        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
