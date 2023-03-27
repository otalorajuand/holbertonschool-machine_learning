#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

x = ['Farrah', 'Fred', 'Felicia']

plt.figure(figsize=(10, 6))
plt.bar(x, fruit[0], color='r', label='apples', width=0.5)
plt.bar(x, fruit[1], bottom=fruit[0], color='yellow', label='bananas',
        width=0.5)
plt.bar(
    x,
    fruit[2],
    bottom=fruit[0] +
    fruit[1],
    color='#ff8000',
    label='oranges',
    width=0.5)
plt.bar(x, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2], color='#ffe5b4',
        label='peaches', width=0.5)
plt.legend(loc='upper right')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.show()
