#!/usr/bin/env python3
"""This module contains the function pool"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images

    Params:
        images: a numpy.ndarray with shape (m, h, w, c)
                containing multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel_shape: a tuple of (kh, kw) containing
                      the kernel shape for the pooling
            kh is the height of the kernel
            kw is the width of the kernel
        stride: a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling

    Returns: a numpy.ndarray containing the pooled images
    """
    m = images.shape[0]
    img_h = images.shape[1]
    img_w = images.shape[2]
    c = images.shape[3]

    kernel_h = kernel_shape[0]
    kernel_w = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    output_h = int((img_h - kernel_h) / sh + 1)
    output_w = int((img_w - kernel_w) / sw + 1)

    output_image = np.zeros((m, output_h, output_w, c))

    image = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            if mode == 'max':
                res = np.max(images[image,
                             (x * sh):(x * sh) + kernel_h,
                             (y * sw):(y * sw) + kernel_w],
                             axis=(1, 2))
            elif mode == 'avg':
                res = np.mean(images[image,
                                     (x * sh):(x * sh) + kernel_h,
                                     (y * sw):(y * sw) + kernel_w],
                              axis=(1, 2))

            output_image[image, x, y] = res

    return output_image
