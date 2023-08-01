#!/usr/bin/env python3
"""This module contains the function autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder

    Args:
    - input_dims is a tuple of integers containing the dimensions of the model
      input
    - filters is a list containing the number of filters for each
      convolutional layer in the encoder, respectively the filters should be
      reversed for the decoder
    - latent_dims is a tuple of integers containing the dimensions of the
      latent space representation

    Returns: encoder, decoder, auto
    - encoder is the encoder model
    - decoder is the decoder model
    - auto is the full autoencoder model
    """
    input_encoded = K.Input(shape=input_dims)

    conv_encoded = K.layers.Conv2D(filters=filters[0], kernel_size=(
        3, 3), padding='same', activation=K.activations.relu)(input_encoded)

    MP_encoded = K.layers.MaxPool2D(pool_size=(2, 2),
                                    padding="same")(conv_encoded)

    for i in filters[1:]:
        conv_encoded = K.layers.Conv2D(
            filters=i,
            kernel_size=(3,3),
            padding='same',
            activation=K.activations.relu)(MP_encoded)

        MP_encoded = K.layers.MaxPool2D(pool_size=(2, 2),
                                        padding="same")(conv_encoded)

    latent = MP_encoded

    encoder = K.Model(input_encoded, latent)

    input_decoded = K.Input(shape=latent_dims)

    conv_decoded = K.layers.Conv2D(filters=filters[-1], kernel_size=(
        3, 3), padding='same', activation=K.activations.relu)(input_decoded)

    up_sample_decode = K.layers.UpSampling2D(size=(2, 2))(conv_decoded)

    for i in reversed(filters[:-2]):

        conv_decoded = K.layers.Conv2D(
            filters=i,
            kernel_size=(
                3,
                3),
            padding='same',
            activation=K.activations.relu)(up_sample_decode)

        up_sample_decode = K.layers.UpSampling2D(size=(2, 2))(conv_decoded)

    conv_decoded = K.layers.Conv2D(filters=filters[0], 
    kernel_size=(3, 3), 
    padding='valid', 
    activation=K.activations.relu)(up_sample_decode)

    up_sample_decode = K.layers.UpSampling2D(size=(2, 2))(conv_decoded)

    conv_decoded = K.layers.Conv2D(filters=input_dims[-1], 
                                   kernel_size=(3, 3), 
                                   padding='same', 
                            activation=K.activations.sigmoid)(up_sample_decode)

    decoder = K.Model(input_decoded, conv_decoded)

    X_input = K.Input(shape=input_dims)
    encoder_o = encoder(X_input)
    decoder_o = decoder(encoder_o)
    auto = K.Model(X_input, decoder_o)

    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
