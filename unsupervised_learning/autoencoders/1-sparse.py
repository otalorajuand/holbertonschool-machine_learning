#!/usr/bin/env python3
"""This module contains the function autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """creates a sparse autoencoder

    Args:
    - input_dims is an integer containing the dimensions of the model input
    - hidden_layers is a list containing the number of nodes for each hidden
      layer in the encoder, respectively the hidden layers should be reversed
      for the decoder
    - latent_dims is an integer containing the dimensions of the latent space
      representation
    - lambtha is the regularization parameter used for L1 regularization on
      the encoded output

    Returns: encoder, decoder, auto
    - encoder is the encoder model
    - decoder is the decoder model
    - auto is the sparse autoencoder model
    """
    input_encoded = K.Input(shape=(input_dims,))

    encoder = K.layers.Dense(hidden_layers[0],
                             activation='relu')(input_encoded)

    for i in hidden_layers[1:]:
        encoder = K.layers.Dense(i, activation='relu')(encoder)

    L1 = K.regularizers.l1(lambtha)
    latent = K.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=L1)(encoder)

    encoder = K.Model(input_encoded, latent)

    input_decoded = K.Input(shape=(latent_dims,))

    decode_layer = K.layers.Dense(hidden_layers[-1],
                                  activation='relu')(input_decoded)

    for i in reversed(hidden_layers[:-1]):
        decode_layer = K.layers.Dense(i, activation='relu')(decode_layer)

    decode_layer = K.layers.Dense(
        input_dims, activation='sigmoid')(decode_layer)

    decoder = K.Model(input_decoded, decode_layer)

    X_input = K.Input(shape=(input_dims,))
    encoder_o = encoder(X_input)
    decoder_o = decoder(encoder_o)
    auto = K.Model(X_input, decoder_o)

    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
