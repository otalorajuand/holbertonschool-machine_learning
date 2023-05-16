#!/usr/bin/env python3
"""This module contains the function lenet5"""
import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the LeNet-5 architecture using tensorflow

    Params:
        - x is a tf.placeholder of shape (m, 28, 28, 1) containing
          the input images for the network
        - y is a tf.placeholder of shape (m, 10) containing
          the one-hot labels for the network

    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization (with default hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """

    x = tf.placeholder("float", [None, 28, 28, 1], name="x")
    y = tf.placeholder("float", [None, 10], name="y")

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    C1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(
            5,
            5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=kernel_initializer)
    output_1 = C1(x)

    P1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    output_2 = P1(output_1)

    C2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(
            5,
            5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=kernel_initializer)
    output_3 = C2(output_2)

    P2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    output_4 = P2(output_3)

    FC3 = tf.layers.dense(activation=tf.nn.relu,
                          units=120,
                          kernel_initializer=kernel_initializer)
    output_5 = FC3(output_4)

    FC4 = tf.layers.dense(activation=tf.nn.relu,
                          units=84,
                          kernel_initializer=kernel_initializer)
    output_6 = FC4(output_5)

    FC5 = tf.layers.dense(activation=tf.nn.relu,
                          units=10,
                          kernel_initializer=kernel_initializer)
    output_7 = FC4(output_6)

    softmax = tf.nn.softmax(output_7)

    loss = tf.losses.softmax_cross_entropy(y, output_7)

    y_pred = tf.math.argmax(output_7, axis=1)
    y = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(equality, "float"))

    train = tf.train.AdamOptimizer().minimize(loss)

    return softmax, train, loss, accuracy
