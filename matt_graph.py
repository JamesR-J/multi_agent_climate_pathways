import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import math


def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

num_gaussians = 20
x_start = 0
x_end = 10
granularity = 100
spacing = abs(x_end - x_start)
x = np.linspace(x_start, x_end, granularity)  # 0)
mu = spacing / 2
sigma = 1.0
y = np.array(gaussian(x, mu, sigma))
new_y = torch.tensor([y]).repeat(num_gaussians, 1).view(num_gaussians * granularity)
new_x = np.linspace(x_start, x_end * num_gaussians, 100 * num_gaussians)
plt.plot(new_x, new_y)
plt.ylim(0, 20)
plt.show()



x = np.linspace(0, 10, 100)
# y = np.linspace(1, -1, 100)
# y = [0] * 100
y = [math.sin(i) for i in x]
plt.plot(x, y)
new_x = [i + math.sin(i) for i in x]
new_y = [math.sin(i) for i in y]
plt.plot(new_x, new_y)
# plt.plot(x, [math.sin(i) for i in x])
# plt.show()











