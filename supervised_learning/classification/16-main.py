#!/usr/bin/env python3

import numpy as np
Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

np.random.seed(16)
nx = np.random.randint(100, 1000)
l = np.random.randint(3, 10)
sizes = np.random.randint(5, 20, l - 1).tolist()
sizes.append(1)
d = Deep(nx, sizes)
print('cache:')
for k, v in sorted(d.cache.items()):
    print(k, v)
print('weights:')
for k, v in sorted(d.weights.items()):
    print(k, v)
print(d.L)
