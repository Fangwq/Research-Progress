import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append("/Users/fangwq/Desktop/orthogonal-additive-gaussian-processes-main")
from oak.model_utils import oak_model, save_model
from oak.utils import get_model_sufficient_statistics, get_prediction_component, my_prediction_component
from scipy import io
from sklearn.model_selection import KFold
from pathlib import Path


def f(x1, x2):
    y = x1**2-2*x2 +np.cos(3*x1)*np.sin(5*x2)
    return y

def f1(x1):
    return x1**2

def f2(x2):
    return -2*x2

def f12(x1, x2):
    return np.cos(3*x1)*np.sin(5*x2)

def Inf1(x1): #integrate x2 from -lim to lim
    return np.zeros(np.shape(x1))

def Inf2(lim, x2): #integrate x1 from -lim to lim
    return 2/3*np.sin(3*lim)*np.sin(5*x2)


# show the figure
# lim = 1.5
# X1, X2 = np.mgrid[-lim:lim:100j, -lim:lim:100j]
# Y = f(X1, X2)


# plt.figure()
# plt.contour(X1, X2, Y, 20)
# plt.xlim(-lim, lim)
# plt.ylim(-lim, lim)
# plt.show()

# data
lim = 1.0
X_train1, X_train2 = np.mgrid[-lim:lim:15j, -lim:lim:25j]


# train data
X_train = np.concatenate((X_train1.reshape(-1, 1), X_train2.reshape(-1, 1)), axis=1)
Y_train = f(X_train[:, 0], X_train[:, 1])
Y_train = Y_train.reshape(-1, 1)
# print(X_train, np.shape(X_train))
# print(Y_train, np.shape(Y_train))
print(np.shape(X_train), np.shape(Y_train))


#test data
X_test1, X_test2 = np.mgrid[-lim:lim:23j, -lim:lim:30j]
X_test = np.concatenate((X_test1.reshape(-1, 1), X_test2.reshape(-1, 1)), axis=1)
Y_test = f(X_test[:, 0], X_test[:, 1])
Y_test = Y_test.reshape(-1, 1)
m1 = 23; m2 =30
N = m1*m2
# print(X_test, np.shape(X_test))
# print(Y_test, np.shape(Y_test))
print(np.shape(X_test), np.shape(Y_test))

oak = oak_model(
    max_interaction_depth=X_train.shape[1],
    num_inducing=30
    )
oak.fit(X_train, Y_train)


oak.get_sobol()
tuple_of_indices, normalised_sobols, sobols = (
    oak.tuple_of_indices,
    oak.normalised_sobols,
    oak.sobols
)
y_pred = oak.predict(X_test)
print(tuple_of_indices, normalised_sobols, sobols)

XT = oak._transform_x(X_test)
oak.alpha = get_model_sufficient_statistics(oak.m, get_L=False)
# get the predicted y for all the kernel components
prediction_list, var = my_prediction_component(
    oak.m,
    oak.alpha,
    XT
)

f1_pred = oak.scaler_y.inverse_transform(
        prediction_list[0].numpy().reshape(-1, 1)
    )
f2_pred = oak.scaler_y.inverse_transform(
        prediction_list[1].numpy().reshape(-1, 1)
    )
f12_pred = oak.scaler_y.inverse_transform(
        prediction_list[2].numpy().reshape(-1, 1)
    )

# after inverse transform
X1_f1 = np.concatenate((X_test[:, 0].reshape(-1, 1), f1_pred, 
                        oak.scaler_y.inverse_transform(var[0][0].reshape(-1, 1)), 
                        oak.scaler_y.inverse_transform(var[0][1].reshape(-1, 1))), axis=1)
X2_f2 = np.concatenate((X_test[:, 1].reshape(-1, 1), f2_pred, 
                        oak.scaler_y.inverse_transform(var[1][0].reshape(-1, 1)), 
                        oak.scaler_y.inverse_transform(var[1][1].reshape(-1, 1))), axis=1)
X1_f1 = np.unique(X1_f1, axis=0)
X2_f2 = np.unique(X2_f2, axis=0)
X1_lower = X1_f1[:, 2]
X1_upper = X1_f1[:, 3]
X2_lower = X2_f2[:, 2]
X2_upper = X2_f2[:, 3]
X12_f12 = f12_pred.reshape(m1, m2)

constant_term = oak.alpha.numpy().sum() * oak.m.kernel.variances[0].numpy()
y_pred_component = np.ones(y_pred.shape[0]) * constant_term

cumulative_sobol, rmse_component = [], []
order = np.argsort(normalised_sobols)[::-1]
for n in order:
    # add predictions of the terms one by one ranked by their Sobol index
    y_pred_component += prediction_list[n].numpy()
    y_pred_component_transformed = oak.scaler_y.inverse_transform(
        y_pred_component.reshape(-1, 1)
    )
    error_component = np.sqrt(
        ((y_pred_component_transformed - y_pred) ** 2).mean()
    )
    rmse_component.append(error_component)
    cumulative_sobol.append(normalised_sobols[n])
cumulative_sobol = np.cumsum(cumulative_sobol)

# sobol_order = np.zeros(len(tuple_of_indices[-1]))
# for i in range(len(tuple_of_indices)):
#     sobol_order[len(tuple_of_indices[i]) - 1] += normalised_sobols[i]

# nll = (
#     -oak.m.predict_log_density(
#         (
#             oak._transform_x(X_test),
#             oak.scaler_y.transform(Y_test),
#         )
#     )
#     .numpy()
#     .mean()
# )

# save_model(
#     oak.m,
#     filename=Path("./output/figure2/model_oak"),
# )
# # save model performance metrics
# np.savez(
#     "./output/figure2/out",
#     cumulative_sobol=cumulative_sobol,
#     order=order,
#     nll=nll,
#     sobol_order=sobol_order,
# )

plt.figure(1)
plt.plot(range(N), y_pred_component_transformed[:, 0], 'kd-', label='components predicted')
plt.plot(range(N), y_pred, 'bo-', label='oak predicted')
plt.plot(range(N), Y_test, 'r-.', label='test')
plt.legend(loc='best')
plt.figure(2)
plt.plot(X1_f1[:, 0], X1_f1[:, 1], 'k-o', label='predicted')
plt.plot(X1_f1[:, 0], f1(X1_f1[:, 0]), 'r-o', label='test')
plt.fill_between(X1_f1[:, 0], X1_lower, X1_upper, alpha=0.2, color="C0")
plt.legend(loc='best')
plt.figure(3)
plt.plot(X2_f2[:, 0], X2_f2[:, 1], 'k-o', label='predicted')
plt.plot(X2_f2[:, 0], f2(X2_f2[:, 0]), 'r-o', label='test')
plt.fill_between(X2_f2[:, 0], X2_lower, X2_upper, alpha=0.2, color="C0")
plt.legend(loc='best')
plt.figure(4)
plt.contour(X_test1, X_test2, f12(X_test1, X_test2).reshape(m1, m2), colors='k', levels=20)
plt.contour(X_test1, X_test2, X12_f12, colors='r', levels=20)
num = 10000
plt.figure(5)
X1_grid, X2_grid = np.meshgrid(X1_f1[:,0], np.linspace(-lim, lim, num))
f12_grid = f12(X1_grid, X2_grid)
print(np.shape(X1_grid), np.shape(X2_grid), np.shape(f12_grid))
plt.plot(X1_f1[:, 0], np.sum(f12_grid, axis=0)/num, 'k-o', label='predicted')
plt.plot(X1_f1[:, 0], Inf1(X1_f1[:, 0]), 'r-o', label='test')
plt.legend(loc='best')
plt.figure(6)
X1_grid, X2_grid = np.meshgrid(np.linspace(-lim, lim, num), X2_f2[:,0])
f12_grid = f12(X1_grid, X2_grid)
print(np.shape(X1_grid), np.shape(X2_grid), np.shape(f12_grid))
plt.plot(X2_f2[:, 0], np.sum(f12_grid, axis=1)/num, 'k-o', label='predicted')
plt.plot(X2_f2[:, 0], Inf2(lim, X2_f2[:, 0]), 'r-o', label='test')
plt.legend(loc='best')
plt.show()







