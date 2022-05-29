import numpy as np
import scipy.spatial as sps

import torch
import matplotlib.pyplot as plt
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import gpflow
import os

torch.set_default_dtype(torch.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = "cpu"

class FC(nn.Module): # FC(xtr.size(1), args.h, 1, args.L, act, args.bias, args.last_bias, args.var_bias)
    def __init__(self, d, h, c, L, act, bias=False, last_bias=False, var_bias=0):
        super().__init__()

        hh = d
        for i in range(L):
            W = torch.randn(h, hh)

            # next two line are here to avoid memory issue when computing the kernel
            n = max(1, 128 * 256 // hh)
            W = nn.ParameterList([nn.Parameter(W[j: j + n]) for j in range(0, len(W), n)])

            setattr(self, "W{}".format(i), W)
            if bias:
                self.register_parameter("B{}".format(i), nn.Parameter(torch.randn(h).mul(var_bias**0.5)))
            hh = h

        self.register_parameter("W{}".format(L), nn.Parameter(torch.randn(c, hh)))
        if last_bias:
            self.register_parameter("B{}".format(L), nn.Parameter(torch.randn(c).mul(var_bias**0.5)))

        self.L = L
        self.act = act
        self.bias = bias
        self.last_bias = last_bias

    def forward(self, x):
        h, d = x.size()
        # print("dimension: ", h, d)
        for i in range(self.L + 1):
            W = getattr(self, "W{}".format(i))

            if isinstance(W, nn.ParameterList):
                W = torch.cat(list(W))

            if self.bias and i < self.L:
                B = self.bias * getattr(self, "B{}".format(i))
            elif self.last_bias and i == self.L:
                B = self.last_bias * getattr(self, "B{}".format(i))
            else:
                B = 0


            if i < self.L:
                x = x @ (W.t() / d ** 0.5)
                x = self.act(x + B)
            else:
                x = x @ (W.t() / h) + B

        if x.shape[1] == 1:
            return x.view(-1)
        return x

def loss_func(f, y):
    loss = (1.0 - 1.0e-6* f * y).relu() / 1.0e-6
    return loss


def inverf2(x):
    """ Inverse error function in 2d."""
    if 'torch' not in str(type(x)):
        x = torch.tensor(x)
    return (-2 * (1 - x).log()).sqrt()


def get_binary_dataset(dataset, ps, seeds, d, params=None, device=None, dtype=None):
    sets = get_normalized_dataset(dataset, ps, seeds, d, params)

    outs = []
    for x, y, i in sets:
        x = x.to(device=device, dtype=dtype)

        assert len(y.unique()) % 2 == 0
        b = x.new_zeros(len(y))
        for j, z in enumerate(y.unique()):
            if j % 2 == 0:
                b[y == z] = 1
            else:
                b[y == z] = -1

        outs += [(x, b, i)]

    return outs


@functools.lru_cache(maxsize=2)
def get_normalized_dataset(dataset, ps, seeds, d=0, params=None):
    out = []
    s = 0
    for p, seed in zip(ps, seeds):
        print(p, seed, d)
        s += seed + 1
        torch.manual_seed(s)
        x = torch.randn(p, d, dtype=torch.float64)
        if dataset == 'stripe':
            y = (x[:, 0] > -0.3) * (x[:, 0] < 1.18549)
        if dataset == 'cylinder':
            dsph = int(params[0])
            stretching = params[1]
            x[:, dsph:] *= stretching
            r = x[:, :dsph].norm(dim=1)
            if dsph == 2:
                y = r > inverf2(1/2)
            else:
                y = (r**2 > dsph - 2 / 3)

        y = y.to(dtype=torch.long)
        out += [(x, y, None)]
    return out

N, d = 100, 2
# for stripe
[(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)] = get_binary_dataset(
        'stripe',
        (N, 100, N),
                (0,1,2),
        d,
    device= device
    )


f = FC(d, N, 1, 1, torch.relu, 1, 0, 0)



def h2(p):
    """Binary entropy function."""
    eps = 1e-10
    return - p * np.log2(p + eps) - (1-p) * np.log2(1-p + eps)

def cond_proba(tree, point, y, k=3):
    """ Query a cKDTree for kNNs for y>0 and y<=0.
        point is where the query is centered.
        Returns conditional probability of the label > 0:
                    P(y>0 | x) = q+ = 1 / (1 + (r+/r-)^d)"""

    eps = 1e-10
    d = point.shape[0]

    flag_p = False
    flag_m = False

    k_large = k
    while not (flag_p * flag_m):
        k_large = min(k_large*2 + 5, len(y) - 1)

        q = tree.query(point, k_large, p=2)
        if sum(y[q[1]] > 0) >= k and not flag_p:
            i = np.where(y[q[1]] > 0)[0][k-1]
            rp = q[0][i] + eps
            flag_p = True

        if sum(y[q[1]] <= 0) >= k and not flag_m:
            i = np.where(y[q[1]] <= 0)[0][k-1]
            rm = q[0][i] + eps
            flag_m = True

    qp = 1 / (1 + (rp/rm)**d)

    if qp == 0:
        qp = eps

    assert qp > 0, "Set probability larger than zero"
    assert qp <= 1, "Set probability smaller than one"

    return qp

def mi_binary(x, y, k=3):
    """Compute mutual information for continuous variable x,
    and binary variable y with p(y=+) = p(y=-) = 0.5 ."""

    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"

    eps = 1e-10  # small noise to break degeneracy.
    x += eps * np.random.rand(*x.shape)
    tree = sps.cKDTree(x)
    q = np.asarray([cond_proba(tree, point, y, k=k) for point in x])

    return 1 - np.mean(h2(q))


def component_mi(eigenvectors, y, k=5, max_r=10, standardize=True):
    """Compute mutual information between last max_r elements of eigenvectors
    and y.
    Returns a list [I(phi_r; y)] for i = 1 to max_r .
    (input eigenvectors normally a torch.tensor)"""
    mi_ = []

    for r in range(1, max_r + 1):
        eig = eigenvectors[:, -r]

        if standardize:
            eig = (eig - eig.mean()) / eig.std()
        mi_.append(mi_binary(np.asarray([[e] for e in eig]), y, k=k))

    return mi_


X = np.random.rand(20, 2)
Y = ytr[0:20]
Y = Y.numpy()
k = gpflow.kernels.Matern12()
kernel = k.K(X).numpy()
evalues, evectors = np.linalg.eig(kernel)
result = component_mi(evectors, Y)
bar = list(range(len(result)))

fig, axis = plt.subplots(figsize =(10, 5))
for i in range(len(bar)):
    plt.bar(bar[i], result[i])
plt.ylim(0,1)
plt.show()


