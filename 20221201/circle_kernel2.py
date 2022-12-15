import lab as B
import numpy as np
import sys
sys.path.append("/home/fwq/PycharmProjects/my_code_cpu/GeometricKernels-main/")
from geometric_kernels.spaces import ProductDiscreteSpectrumSpace
from geometric_kernels.spaces import Circle
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
import matplotlib.pyplot as plt

# plot matern kernel on circle. And this method should be generalized to Lie group method.
# And it should refer to the paper <<Stationary Kernels and Gaussian Processes
# on Lie Groups and their Homogeneous Spaces>>.
num = 200
space = Circle()

# %%
theta = B.linspace(0, 2*3.14159, num).reshape(-1,1)
# map it to circle
Y = np.sin(theta)
X = np.cos(theta)
Z = np.zeros_like(Y)

eigs = space.get_eigenfunctions(num)
eigfuncs = eigs(theta)
print(X.shape, Y.shape, Z.shape, eigfuncs.shape)
index = 20

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X, Y, Z)
ax.scatter(X, Y, eigfuncs[:,20].reshape(-1, 1), 'bo')
ax.scatter(X[index], Y[index], Z[index],  c='r', marker = 'D', s=50)


point = theta[index].reshape(1,-1)
print(point.shape)
kernel = MaternKarhunenLoeveKernel(space, num)
params, state = kernel.init_params_and_state()
print(params)

k_xx = kernel.K(params, state, theta, point)
off_set = np.min(k_xx)
print(off_set)
k_point = (k_xx - off_set).reshape(-1, 1)
print(k_xx.shape, len(np.unique(k_xx)))
ax.scatter(X, Y, k_point,  c='r', marker = 'o', s=50)
ax.scatter(X[index], Y[index], k_point, 'kD')
ax.axis('off')
plt.show()
# %%
