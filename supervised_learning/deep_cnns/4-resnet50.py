#!/usr/bin/env python3
"""This module contains the function resnet50"""
import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds the ResNet-50 architecture as described in
       Deep Residual Learning for Image Recognition (2015)

    Returns: the keras model
    """

    initializer = K.initializers.he_normal()

    # 7x7 convolution
    C1 = K.layers.Conv2D(filters=64,
                         kernel_size=(1, 1),
                         padding='same',
                         strides=2,
                         kernel_initializer=initializer)(A_prev)

    # Batch normalization
    BN_1 = K.layers.BatchNormalization(axis=3)(C1)

    # activation
    relu_1 = K.layers.ReLU()(BN_1)

    # maxpool
    MP_2 = K.layers.MaxPool2D(pool_size=(3, 3),
                              padding='same',
                              strides=2)(relu_1)

    projection_1 = projection_block(MP_2, [64, 64, 256], 1)
    indentity_11 = identity_block(projection_1, [64, 64, 256])
    indentity_12 = identity_block(indentity_11, [64, 64, 256])

    projection_2 = projection_block(indentity_12, [128, 128, 512])
    indentity_21 = identity_block(projection_2, [128, 128, 512])
    indentity_22 = identity_block(indentity_21, [128, 128, 512])
    indentity_23 = identity_block(indentity_22, [128, 128, 512])

    projection_3 = projection_block(indentity_23, [256, 256, 1024])
    indentity_31 = identity_block(projection_3, [256, 256, 1024])
    indentity_32 = identity_block(indentity_31, [256, 256, 1024])
    indentity_33 = identity_block(indentity_32, [256, 256, 1024])
    indentity_34 = identity_block(indentity_33, [256, 256, 1024])
    indentity_35 = identity_block(indentity_34, [256, 256, 1024])

    projection_4 = projection_block(indentity_35, [512, 512, 2048])
    indentity_41 = identity_block(projection_4, [512, 512, 2048])
    indentity_42 = identity_block(indentity_41, [512, 512, 2048])

    AP = K.layers.AveragePooling2D(pool_size=(7, 7),
                                   padding='valid')(indentity_42)

    linear = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=initializer)(AP)

    model = K.models.Model(inputs=X, outputs=linear)

    return model
