import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time

torch.set_default_dtype(torch.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class FC(nn.Module): # FC(xtr.size(1), args.h, 1, args.L, act, args.bias, args.last_bias, args.var_bias)
    def __init__(self, d, h, c, L, act, bias=False, last_bias=False, var_bias=1):
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


def compute_kernels(f, xtr, xte, parameters=None):
    from hessian import gradient

    if parameters is None:
        parameters = list(f.parameters())

    ktrtr = xtr.new_zeros(len(xtr), len(xtr))
    ktetr = xtr.new_zeros(len(xte), len(xtr))
    ktete = xtr.new_zeros(len(xte), len(xte))
    params = []
    current = []
    for p in sorted(parameters, key=lambda p: p.numel(), reverse=True):
        current.append(p)
        print(sum(p.numel() for p in current), 2e9 // (8 * (len(xtr) + len(xte))))
        print(sum(p.numel() for p in current) > 2e9 // (8 * (len(xtr) + len(xte))))
        if sum(p.numel() for p in current) > 2e9 // (8 * (len(xtr) + len(xte))):
            if len(current) > 1:
                params.append(current[:-1])
                current = current[-1:]
            else:
                params.append(current)
                current = []
    if len(current) > 0:
        params.append(current)

    for i, p in enumerate(params):
        print("[{}/{}] [len={} numel={}]".format(i, len(params), len(p), sum(x.numel() for x in p)), flush=True)
        # print(p)
        
        jtr = xtr.new_empty(len(xtr), sum(u.numel() for u in p))  # (P, N~)
        jte = xte.new_empty(len(xte), sum(u.numel() for u in p))  # (P, N~)

        for j, x in enumerate(xtr):
            jtr[j] = gradient(f(x[None]), p)  # (N~)

        for j, x in enumerate(xte):
            jte[j] = gradient(f(x[None]), p)  # (N~)

        ktrtr.add_(jtr @ jtr.t())
        ktetr.add_(jte @ jtr.t())
        ktete.add_(jte @ jte.t())
        del jtr, jte

    return ktrtr, ktetr, ktete


def ntk_kernel(net, gamma_train, gamma_test, use_cuda=True):
	# suppose cuda available
	n_test = len(gamma_test)
	n_train = len(gamma_train)
	# the following computes the gradients with respect to all parameters
	grad_list = []
	for gamma in gamma_train:
		loss = net(gamma.reshape((1,-1)))
		grad_list.append(torch.autograd.grad(loss, net.parameters(), retain_graph=True))

	# testvstrain kernel
	if np.all((gamma_train != gamma_test).cpu().numpy()):
		K_X1X2 = torch.zeros((n_test, n_train))
		for i, gamma in enumerate(gamma_test):
			loss = net(gamma.reshape((1, -1)))
			grads = torch.autograd.grad(loss, net.parameters(), retain_graph=True)  # extract NN gradients
			for j in range(len(grad_list)):
				pt_grad = grad_list[j]  # the gradients at the jth (out of 4) data point
				K_X1X2[i, j] = sum([torch.sum(torch.mul(grads[u], pt_grad[u])) for u in range(len(grads))])
		return K_X1X2
	else:
		K_XX = torch.zeros((n_train, n_train))
		for i in range(n_train):
			grad_i = grad_list[i]
			for j in range(i + 1):
				grad_j = grad_list[j]
				K_XX[i, j] = sum([torch.sum(torch.mul(grad_i[u], grad_j[u])) for u in range(len(grad_j))])
				K_XX[j, i] = K_XX[i, j]
		return K_XX

d, N = 2, 10000
X = torch.randn(N, d).to('cuda')
Y = torch.randn(N, d).to('cuda')
print(np.amax(np.abs((X.cpu().numpy()-Y.cpu().numpy()))))
f = FC(d, N, 1, 1, torch.relu, 1, 0, 1).to('cuda')
start = time.time()
kXX = ntk_kernel(f, X, X)   # much slower: 20737.296113967896 for N=10000
kXY = ntk_kernel(f, X, Y)
kYY = ntk_kernel(f, Y, Y)
end1 = time.time()
a, b, c = compute_kernels(f, X, Y)   # very fast: 86.82520890235901 for N = 10000
end2 = time.time()
print(end1 - start)
print(end2 - end1)
# print(kXX, kXY, kYY)
# print(a, b, c)
print(np.amax(np.abs(kXX.cpu().numpy()-a.cpu().numpy())))
print(np.amax(np.abs(kXY.cpu().numpy()-b.cpu().numpy())))
print(np.amax(np.abs(kYY.cpu().numpy()-c.cpu().numpy())))



