#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

f, ax = plt.subplots()
points = ax.scatter(x, y, c=z, s=50, cmap="viridis")
cbar = f.colorbar(points)
cbar.set_label('Elevation (m)')

plt.title('Mountain Elevation')
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.show()
