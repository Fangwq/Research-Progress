import matplotlib.pyplot as plt
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os

torch.set_default_dtype(torch.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = "cuda"

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

N, d = 10000, 2
# for stripe
[(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)] = get_binary_dataset(
        'stripe',
        (N, 100, N),
                (0,1,2),
        d,
    device= device
    )

# for cylinder
# [(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)] = get_binary_dataset(
#         'cylinder',
#         (500, 100, 500),
#                 (0,1,2),
#         3,                       #(xp1, xp2, xp)
#         (2, 1)
#     )

print(np.shape(xtr), np.shape(ytr))
print(np.shape(xtk), np.shape(ytk))
print(np.shape(xte), np.shape(yte))
# print(np.unique(ytr))
# print(np.unique(ytk))
# print(np.unique(yte))
print(inverf2(1/2))

f = FC(d, N, 1, 1, torch.relu, 1, 0, 0)
f0 = FC(d, N, 1, 1, torch.relu, 1, 0, 0)
f = f.to(device)
f0 = f0.to(device)
lr = 0.1

optimizer = optim.Adam(f.parameters(), lr=lr)
# optimizer = optim.SGD(f.parameters(), lr=lr)
n_epochs = 50000
otr0 = f0(xtr)
loss_his = []
for epoch in range(n_epochs):
    optimizer.zero_grad()
    otr =  f(xtr) - otr0
    loss = loss_func(otr, ytr).mean()
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f.W0[0][0:10])
    print(otr[0:10], ytr[0:10])
    loss_his.append(loss.item())
    # print(f.W1[0][0:10])

xte = xte.cpu().numpy()
yte = yte.cpu().numpy()


fig, ax = plt.subplots(1, 1)
for i, mark in [[-1, 'rx'], [1, 'bo']]:
    ind = yte[:] == i
    ax.plot(xte[ind, 0], xte[ind, 1], mark)
# plt.plot([-0.3, -0.3], [-5, 5], 'k-')
# plt.plot([1.18549, 1.18549], [-5, 5], 'k-')
plt.ylim([-4, 4])
plt.xlim([-4, 4])
ax.set_aspect(1)
plt.figure(2)
plt.plot(range(len(loss_his)), loss_his, 'ro-')
plt.show()


