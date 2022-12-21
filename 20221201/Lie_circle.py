import sys
sys.path.append("/home/fwq/PycharmProjects/my_code_cpu/LieStationaryKernels-master")
import torch
from lie_stationary_kernels.spaces import Torus
from lie_stationary_kernels.spectral_kernel import EigenbasisSumKernel
from lie_stationary_kernels.prior_approximation import RandomPhaseApproximation
from lie_stationary_kernels.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

random.seed(1)
torch.manual_seed(1)
# plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

space = Torus(n=1, order=101)
print(space.id, space.dim)

lengthscale, nu, variance = 0.1, 1.5, 1
#measure = SqExpSpectralMeasure(space.dim, lengthscale, variance=1.0)
measure = MaternSpectralMeasure(space.dim, lengthscale, nu, variance)

kernel = EigenbasisSumKernel(measure, space)
sampler = RandomPhaseApproximation(kernel, phase_order=100)


points = np.linspace(-0.5, 0.5, 1000).reshape(1000, 1)

samples = []
batch_size = 100

x = torch.tensor(points, device=device, dtype=dtype)
x = x.view(-1, 1)

kernel_values = kernel(x, space.id).cpu().detach().numpy()
print(kernel_values.shape)
kernel_values = np.squeeze(kernel_values)

samples = [None, None, None]
for j in range(3):
    sampler.resample()
    samples[j] = sampler(x).cpu().detach().numpy()
    samples[j] = np.squeeze(samples[j])

points = np.squeeze(points)


def plot_3d(value, name):
    plt.figure()
    ax = plt.axes(projection='3d')
    angle = 2 * np.pi * points

    x = np.cos(angle)
    y = np.sin(angle)
    z = np.zeros_like(x)
    ax.plot(x, y, z, '--', c='gray', linewidth=6, dashes=(5, 5))

    max_id = value.argmax()
    if name == "circle_kernel":
        ax.scatter(x[max_id], y[max_id], 0, s=600, c='black')
    ax.plot([x[max_id], x[max_id]], [y[max_id], y[max_id]], [0, value[max_id]], '--', c='gray', linewidth=6,
            dashes=(3, 3))
    ax.plot(x, y, value, linewidth=8)

    for i in range(999):
        rectangle = Poly3DCollection([[[x[i], y[i], 0], [x[i], y[i], value[i]],
                                       [x[i + 1], y[i + 1], value[i + 1]], [x[i + 1], y[i + 1], 0]]], color='#1f77b4',
                                     alpha=0.01)
        rectangle.set_edgecolor('#1f77b4')
        ax.add_collection3d(rectangle)

    ax.elev = 60
    if name == "circle_kernel":
        ax.azim = 150
    else:
        ax.azim = 240

    plt.axis('off')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1.5)

    #uncomment if you want to save plots
    # plt.tight_layout()
    # plt.savefig(name + "_plot3d.pdf", transparent=True)
    # plt.clf()

plot_3d(kernel_values, "circle_kernel")
plot_3d(samples[0], "sample_1")
plot_3d(samples[1], "sample_2")
plot_3d(samples[2], "sample_3")
plt.show()

