#!/usr/bin/env python3
"""This module include the function evaluate"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        return y_pred, loss, accuracy
