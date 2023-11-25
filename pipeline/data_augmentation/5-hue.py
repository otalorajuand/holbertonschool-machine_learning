#!/usr/bin/env python3
"""This module contains the function change_hue"""
import tensorflow as tf


def change_hue(image, delta):
    """changes the hue of an image

    Args:
    - image is a 3D tf.Tensor containing the image to change
    - delta is the amount the hue should change

    Returns the altered image
    """
    return tf.image.adjust_hue(image, delta)