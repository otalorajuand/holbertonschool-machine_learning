#!/usr/bin/env python3

import numpy as np
Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

np.random.seed(16)
nx = np.random.randint(3, 10)
try:
    d = Deep(nx, [3, 5, -1])
    print('FAIL')
except TypeError as e:
    if str(e) == 'layers must be a list of positive integers':
        print('OK', end='')
    else:
        print('FAIL')
