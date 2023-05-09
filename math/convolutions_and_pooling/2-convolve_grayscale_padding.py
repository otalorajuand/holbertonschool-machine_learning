#!/usr/bin/env python3
"""This module contains the function convolve_grayscale_same"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding

    Params:
        images: a numpy.ndarray with shape (m, h, w)
                containing multiple grayscale images
        kernel: a numpy.ndarray with shape (kh, kw)
                containing the kernel for the convolution
        padding: a tuple of (ph, pw)
                 ph is the padding for the height of the image
                 pw is the padding for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    """

    m = images.shape[0]
    img_h = images.shape[1]
    img_w = images.shape[2]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    pad_h = padding[0]
    pad_w = padding[1]

    output_h = img_h + 2 * pad_h - kernel_h + 1
    output_w = img_w + 2 * pad_w - kernel_w + 1

    output_image = np.zeros((m, output_h, output_w))

    image = np.arange(m)

    images_pad = np.pad(
        images, pad_width=(
            (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    for x in range(output_h):
        for y in range(output_w):
            output_image[image, x, y] = np.sum(
                images_pad[image, x:x + kernel_h, y:y + kernel_w] * kernel,
                axis=(1, 2))

    return output_image
