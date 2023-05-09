#!/usr/bin/env python3
"""This module contains the function convolve_grayscale_same"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images

    Params:
        images: a numpy.ndarray with shape (m, h, w)
                containing multiple grayscale images
        kernel: a numpy.ndarray with shape (kh, kw)
                containing the kernel for the convolution

    Returns: a numpy.ndarray containing the convolved images
    """

    m = images.shape[0]
    img_h = images.shape[1]
    img_w = images.shape[2]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    output_h = img_h
    output_w = img_w

    output_image = np.zeros((m, output_h, output_w))

    image = np.arange(m)

    pad_h = ceil(kernel_h / 2)
    pad_w = ceil(kernel_w / 2)

    images_pad = np.pad(
        images, pad_width=(
            (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    for x in range(output_h):
        for y in range(output_w):
            output_image[image, x, y] = np.sum(
                images_pad[image, x:x + kernel_h, y:y + kernel_w] * kernel,
                axis=(1, 2))

    return output_image
