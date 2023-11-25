#!/usr/bin/env python3
"""This module contains the function change_brightness"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """randomly changes the brightness of an image

    Args:
    - image is a 3D tf.Tensor containing the image to change
    - max_delta is the maximum amount the image should be brightened
      (or darkened)

    Returns the altered image
    """
    return tf.image.random_brightness(image, max_delta)