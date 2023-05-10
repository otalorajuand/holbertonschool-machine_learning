#!/usr/bin/env python3

import numpy as np
oh_encode = __import__('24-one_hot_encode').one_hot_encode

np.random.seed(1)
classes = np.random.randint(0, 10)
m = np.random.randint(100, 200)
Y = np.random.randint(0, classes, m)
np.set_printoptions(threshold=np.inf)
print(Y)
print(max(Y) > classes + 10)
print(classes)
print(oh_encode(Y, classes + 10))
