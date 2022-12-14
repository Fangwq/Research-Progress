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
num = 100
circ = Circle()
prod_space = ProductDiscreteSpectrumSpace(Circle(), Circle(), num_eigen=num)
# %%
X = B.linspace(0, 2*3.14159, num).reshape(-1,1)
Y = np.sin(X)
grid = np.meshgrid(X, Y)
mesh = B.stack(*grid, axis=-1).reshape((-1, 2))
print(mesh.shape)
# map it to circle
X = np.cos(X)
Y = np.sin(Y)
Z = np.zeros_like(Y)

eigs = prod_space.get_eigenfunctions(num)
eigfuncs = eigs(mesh).reshape((num, num, -1))
print(eigfuncs)
print(X.shape, Y.shape, Z.shape, eigfuncs.shape)


fig = plt.figure()
ax = fig.gca(projection='3d')
# ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X, Y, Z)
ax.scatter(X, Y, eigfuncs[80, :, 300], 'bo')
ax.scatter(X[0], Y[0], Z[0],  c='r', marker = 'D', s=50)


point = mesh[0].reshape(1,-1)
print(point.shape)
kernel = MaternKarhunenLoeveKernel(prod_space, num)
params, state = kernel.init_params_and_state()
print(params)

k_xx = kernel.K(params, state, mesh, point).reshape(num, num)
off_set = np.min(k_xx)
k_point = (k_xx[0,:] - off_set).reshape(-1, 1)
print(k_xx.shape, len(np.unique(k_xx)), k_point.shape)
ax.scatter(X, Y, k_point,  c='r', marker = 'o', s=50)
ax.scatter(X[0], Y[0], k_point, 'kD')
ax.axis('off')
plt.show()
# %%
