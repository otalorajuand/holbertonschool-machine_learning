#!/usr/bin/env python3
"""This module contains the class Decoder"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """creates the decoder for a transformer"""

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_len,
            drop_rate=0.1):
        """Class constructor

        Args:
        - N: the number of blocks in the encoder
        - dm: the dimensionality of the model
        - h: the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - target_vocab: the size of the target vocabulary
        - max_seq_len: the maximum sequence length possible
        - drop_rate: the dropout rate
        """
        super().__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)


    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Args:
        - x: a tensor of shape (batch, target_seq_len, dm)
             containing the input to the decoder
        - encoder_output: a tensor of shape (batch, input_seq_len, dm)
                          containing the output of the encoder
        - training: a boolean to determine if the model is training
        - look_ahead_mask: the mask to be applied to the first
                           multi head attention layer
        - padding_mask: the mask to be applied to the second
                        multi head attention layer

        Returns: a tensor of shape (batch, target_seq_len, dm)
                 containing the decoder output
        """
        seq_len = x.shape[1]
        # adding embedding and position encoding.
        embedding = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding += self.positional_encoding[:seq_len]
        decoder_out = self.dropout(embedding, training=training)

        for block in self.blocks:
            dencoder_out = block(
                decoder_out,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask)

        return dencoder_out
