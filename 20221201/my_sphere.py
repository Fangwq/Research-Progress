import torch
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
import sys
sys.path.append("/home/fwq/PycharmProjects/my_code_cpu/LieStationaryKernels-master")
from lie_stationary_kernels.spaces.sphere import Sphere
from lie_stationary_kernels.spaces.grassmannian import Grassmannian, OrientedGrassmannian
from lie_stationary_kernels.spectral_kernel import EigenbasisSumKernel
from lie_stationary_kernels.prior_approximation import RandomPhaseApproximation
from lie_stationary_kernels.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure
from examples.gpr_model import ExactGPModel, train
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import gpytorch
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.autograd.set_detect_anomaly(True)
torch.cuda.set_device("cuda:1")
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def sphere_grid(r=1., nlats=42, nlons=42, center=None):
    if center is None:
        center = np.r_[0, 0, 0]
    phi, theta = np.mgrid[0:np.pi:nlats*1j, 0:2 * np.pi:nlons*1j]

    x = r * np.sin(phi) * np.cos(theta) + center[0]
    y = r * np.sin(phi) * np.sin(theta) + center[1]
    z = r * np.cos(phi) + center[2]

    return torch.tensor(x, device=device, dtype=dtype), \
           torch.tensor(y, device=device, dtype=dtype),\
           torch.tensor(z, device=device, dtype=dtype)

def plot_sphere(x, y, z, c):
    x_, y_, z_, c_ = x.detach().cpu().numpy(), y.detach().cpu().numpy(), z.detach().cpu().numpy(), c.detach().cpu().numpy()
    cmap = plt.get_cmap('plasma')
    # cmap = cm.viridis
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax=plt.subplot(111, projection='3d')
    surf = ax.plot_surface(x_, y_, z_,
                           cstride=1, rstride=1,
                           cmap=cmap,
                           facecolors=cmap(c_))
    ax.scatter(np.array([0]), np.array([0]), np.array([1]), c='k', marker = 'o', s=50)
    # ax.scatter(x_, y_, z_, 'ko')

    #ax.set_axis_off()
    
order = 10
sphere = Sphere(n=2, order=order)

lengthscale, nu, variance = 0.25, 5.0 + sphere.dim, 3
measure = SqExpSpectralMeasure(sphere.dim, lengthscale, variance)
#self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

sphere_kernel = EigenbasisSumKernel(measure=measure, manifold=sphere)
sphere_sampler = RandomPhaseApproximation(kernel=sphere_kernel, phase_order=10**2)

x, y, z = sphere_grid()
points = torch.stack((x,y,z), dim=-1).reshape(-1, 3)
sphere_kernel_matrix = sphere_kernel(torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype), points).reshape(x.shape[0], x.shape[1])
sphere_samples = sphere_sampler(points).reshape(x.shape[0], x.shape[1])
plot_sphere(x, y, z, sphere_samples)
plot_sphere(-x, -y, -z, sphere_samples)
plot_sphere(x, y, z, sphere_kernel_matrix)
# print(torch.min(sphere_samples), torch.max(sphere_samples))

points_ = torch.unsqueeze(points, -1)
print(points_.shape)

#grassmannian = Grassmannian(3,1, order=10, average_order=20)
grassmannian = OrientedGrassmannian(3,1, order=10, average_order=100)
grassmannian_kernel = EigenbasisSumKernel(measure=measure, manifold=grassmannian)
grassmannian_sampler = RandomPhaseApproximation(kernel=grassmannian_kernel, phase_order=100)

grassmannian_samples = grassmannian_sampler(points_).reshape(x.shape[0], x.shape[1])
plot_sphere(x, y, z, grassmannian_samples)
plot_sphere(-x,-y,-z, grassmannian_samples)

# print(torch.max(grassmannian_samples), torch.min(grassmannian_samples))

rand_x = sphere.rand(50)
rand_x_ = torch.unsqueeze(rand_x, -1)
cov_grassmanian = grassmannian_kernel(rand_x_, rand_x_)
cov_sphere = sphere_kernel(rand_x, rand_x)
# print(torch.max(torch.abs(cov_grassmanian-cov_sphere)))
plt.show()