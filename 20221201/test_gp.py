# %%
import torch
import sys
sys.path.append("/home/fwq/PycharmProjects/my_jax_python3.7/SteerableCNP")
from steer_cnp.kernel import (
    RBFKernel,
    SeparableKernel,
    DotProductKernel,
    RBFDivergenceFreeKernel,
    RBFCurlFreeKernel,
    kernel_smooth,
)
from steer_cnp.gp import sample_gp_prior, conditional_gp_posterior
from steer_cnp.utils import sample_gp_grid_2d, plot_inference
import matplotlib.pyplot as plt
import numpy as np

# %%
rbf = SeparableKernel(2, 2, RBFKernel(2, 3.0))
div = RBFDivergenceFreeKernel(2, 3.0)
curl = RBFCurlFreeKernel(2, 3.0)

# %%
x = torch.arange(-4, 4, step=0.5)
x1, x2 = torch.meshgrid(x, x)
x1 = x1.flatten()
x2 = x2.flatten()

X_grid = torch.stack([x1, x2], dim=-1)

# %%
Y_rbf = sample_gp_prior(X_grid, rbf)
plt.figure(1)
plt.quiver(
    X_grid[:, 0],
    X_grid[:, 1],
    Y_rbf[:, 0],
    Y_rbf[:, 1],
    color="r",
    scale=50,
)
# %%
Y_div = sample_gp_prior(X_grid, div)
plt.figure(2)
plt.quiver(
    X_grid[:, 0],
    X_grid[:, 1],
    Y_div[:, 0],
    Y_div[:, 1],
    color="r",
    scale=50,
)

# %%
Y_curl = sample_gp_prior(X_grid, curl)
plt.figure(3)
plt.quiver(
    X_grid[:, 0],
    X_grid[:, 1],
    Y_curl[:, 0],
    Y_curl[:, 1],
    color="r",
    scale=50,
)
# %%

X_context = torch.Tensor([[1, 2], [2, 1], [-1, -1]])
Y_context = torch.Tensor([[1, 1], [1, -2], [-4, 3]])

x = torch.arange(-4, 4, step=0.5)
x1, x2 = torch.meshgrid(x, x)
x1 = x1.flatten()
x2 = x2.flatten()

X_target = torch.stack([x1, x2], dim=-1)

# %%

Y_mean, _, _ = conditional_gp_posterior(X_context, Y_context, X_target, rbf)
plt.figure(4)
plt.quiver(
    X_target[:, 0], X_target[:, 1], Y_mean[:, 0], Y_mean[:, 1], color="b", scale=100
)
plt.quiver(
    X_context[:, 0],
    X_context[:, 1],
    Y_context[:, 0],
    Y_context[:, 1],
    color="r",
    scale=100,
)
# %%

Y_mean, _, _ = conditional_gp_posterior(X_context, Y_context, X_target, div)
plt.figure(5)
plt.quiver(
    X_target[:, 0], X_target[:, 1], Y_mean[:, 0], Y_mean[:, 1], color="b", scale=100
)
plt.quiver(
    X_context[:, 0],
    X_context[:, 1],
    Y_context[:, 0],
    Y_context[:, 1],
    color="r",
    scale=100,
)
# %%
Y_mean, _, _ = conditional_gp_posterior(X_context, Y_context, X_target, curl)
plt.figure(6)
plt.quiver(
    X_target[:, 0], X_target[:, 1], Y_mean[:, 0], Y_mean[:, 1], color="b", scale=100
)
plt.quiver(
    X_context[:, 0],
    X_context[:, 1],
    Y_context[:, 0],
    Y_context[:, 1],
    color="r",
    scale=100,
)
# %%

X, Y = sample_gp_grid_2d(curl, n_axis=30)
plt.figure(7)
plt.quiver(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], color="b", scale=50)

# %%
Y_mean, _, variances = conditional_gp_posterior(X_context, Y_context, X_target, curl)
plot_inference(X_context, Y_context, X_target, Y_mean, variances)

# rotated data: check the rotation effect
theta = np.pi/2
rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
print(rotate_matrix@rotate_matrix.T)
X_context_rotate = list(map(lambda x: rotate_matrix@x, X_context.numpy()))
X_context_rotate = torch.Tensor(X_context_rotate)
print(X_context_rotate)
Y_context_rotate = list(map(lambda x: rotate_matrix@x, Y_context.numpy()))
y_context_rotate = torch.Tensor(Y_context_rotate)
print(Y_context_rotate)
X_rotate = list(map(lambda x: rotate_matrix@x, X_target.numpy()))
X_rotate = torch.Tensor(X_rotate)


# print(X_rotate)
Y_mean_rotate, _, variances_rotate = conditional_gp_posterior(X_context_rotate, y_context_rotate, X_rotate, curl)
plot_inference(X_context_rotate, y_context_rotate, X_rotate, Y_mean_rotate, variances_rotate)
plt.show()

