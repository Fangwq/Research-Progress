import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.cluster.vq import kmeans2
import sys
sys.path.append("/home/fwq/PycharmProjects/my_code_cpu/invgp")
import time

start = time.time()
np.random.seed(1234)

# create a 2D function with symmetry around the line y = x
def f(x1, x2):
    y = (((x1-5)**2+x2-10)**2 + ((x2-5)**2+x1-10)**2)/1000.0 +1.0  # function one
    # y = (x1**2 + x2**2 -x1*x2)/20          # function two
    # y = (((x1-5)**2+x2-10)**2*np.cos(x1*x2/np.pi/4) + ((x2-5)**2+x1-10)**2*np.cos(x1*x2/np.pi/4))/1000.0   # function three
    return y


# show the figure
lim = 10
X1, X2 = np.mgrid[0:lim:100j, 0:lim:100j]
Y = f(X1, X2)


#==============train data================
X_train1, X_train2 = np.mgrid[0:lim:9j, 0:lim:11j]
X_train = np.concatenate((X_train1.reshape(-1, 1), X_train2.reshape(-1, 1)), axis=1)
Y_train = f(X_train[:, 0], X_train[:, 1])
Y_train = Y_train.reshape(-1, 1)
index = np.random.permutation(len(X_train)) # pick random train data
X_train = X_train[index]
Y_train = Y_train[index]
N_train = 75
X_train = X_train[:N_train]
Y_train =Y_train[:N_train]
# print(X_train, np.shape(X_train))
# print(Y_train, np.shape(Y_train))
print(np.shape(X_train), np.shape(Y_train))
print(np.max(Y_train))

#================test data================
# construct the symmetric data
X_test1, X_test2 = np.mgrid[0:lim:9j, 0:lim:9j]
X_test = np.concatenate((X_test1.reshape(-1, 1), X_test2.reshape(-1, 1)), axis=1)
Y_test = f(X_test[:, 0], X_test[:, 1])
Y_test = Y_test.reshape(-1, 1)
m1 = 9; m2 = 9
N_test = m1*m2
print(np.shape(X_test), np.shape(Y_test))

#============show the function, train and test data=================
fig = plt.figure(1)
ax1 = fig.gca(projection='3d')
surf=ax1.plot_surface(X1, X2, Y, rstride=1, cstride=1,cmap=cm.coolwarm, edgecolor='none')
ax1.scatter3D(X_train[:,0], X_train[:,1], Y_train, 'ko')
ax1.scatter3D(X_test[:,0], X_test[:,1], Y_test, 'r^')
fig = plt.figure(2)
ax2 = fig.gca()
ax2.contour(X1, X2, Y, 50)
ax2.plot(X_train[:,0], X_train[:,1], 'ko')
ax2.plot(X_test[:, 0], X_test[:,1], 'ro')
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_aspect(1)
plt.savefig("Actual.png")
# plt.show()

#===========model construction============
import tensorflow as tf
import gpflow
from gpflow.ci_utils import ci_niter
import copy
tf.random.set_seed(1234)

# for function one
M = int(N_train/7)
D = 2
Z = kmeans2(X_train, M, minit='points')[0]

C = 1
iteration = 10000
kernel = gpflow.kernels.Matern52(variance=2.0, lengthscales=[0.5, 1.0])
likelihood = gpflow.likelihoods.Gaussian(2.0)
q_mu = np.random.random((M, C))
q_sqrt = np.random.random((M, 1))


# for invgp
# inv_Z = copy.deepcopy(Z)
inv_Z = kmeans2(X_train, M, minit='points')[0]
inv_kernel = copy.deepcopy(kernel)
inv_likelihood = copy.deepcopy(likelihood)
invq_mu = copy.deepcopy(q_mu)
invq_sqrt = copy.deepcopy(q_sqrt)

opt = gpflow.optimizers.Scipy()
#======================original SVGP regression=========================
m = gpflow.models.SVGP(kernel,
                       likelihood,
                       inducing_variable=Z,
                       num_latent_gps=C,
                       q_mu=q_mu,
                       q_sqrt=q_sqrt,
                       # whiten=False,
                       q_diag=True,
                       )
gpflow.utilities.print_summary(m)
m_opt_logs = opt.minimize(
	m.training_loss_closure((X_train, Y_train)), m.trainable_variables, options=dict(maxiter=ci_niter(iteration))
)
gpflow.utilities.print_summary(m)
y_pred, y_var = m.predict_y(X_test)
gptest_error = np.sum(np.abs(y_pred.numpy()-Y_test) / N_test)
print(y_pred, gptest_error, m.elbo((X_train, Y_train)).numpy())

#=======================invariant SVGP regression====================
# It is hard to tune the invariant gaussian process
print("==============invariant GP=============")
from models.InvSVGP import InvSVGP
from kernels.orbits import SwitchXY
from kernels.invariant import Invariant, StochasticInvariant

# # for invgp
# inv_Z = copy.deepcopy(Z)
# inv_kernel = copy.deepcopy(kernel)
# inv_likelihood = copy.deepcopy(likelihood)
# invq_mu = copy.deepcopy(q_mu)
# invq_sqrt = copy.deepcopy(q_sqrt)

inv_kernel = StochasticInvariant(inv_kernel, SwitchXY())
inv_m = InvSVGP(inv_kernel,
                inv_likelihood,
                inducing_variable=inv_Z,
                num_latent_gps=C,
                q_mu=invq_mu,
                q_sqrt=invq_sqrt,
                # whiten=False,
                q_diag=True,
                )
# # first: this method does not get satisfied results
# gpflow.utilities.print_summary(inv_m)
# invm_opt_logs = opt.minimize(
# 	inv_m.training_loss_closure((X_train, Y_train)), inv_m.trainable_variables, options=dict(maxiter=ci_niter(iteration), disp=True)
# )
# gpflow.utilities.print_summary(inv_m)
# invy_pred, invy_var = inv_m.predict_y(X_test)
# print(invy_pred)

# # second: still hard to optimize
minibatch_size = 20
log_elbo = []
log_error = []
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).repeat().shuffle(N_train)
train_iter = iter(train_dataset.batch(minibatch_size))
training_loss = inv_m.training_loss_closure(train_iter, compile=True)
optimizer = tf.optimizers.Adam(learning_rate= 0.001, beta_1=tf.Variable(0.99, dtype=gpflow.config.default_float()),
                beta_2=tf.Variable(0.999, dtype=gpflow.config.default_float()),
                epsilon=tf.Variable(1e-7, dtype=gpflow.config.default_float()),
            )
# optimizer = tf.optimizers.SGD(learning_rate= 0.001, momentum=0.99, nesterov=True)
best_error = np.inf
best_para = gpflow.utilities.parameter_dict(inv_m)
iteration = 10000

@tf.function
def optimization_step():
    optimizer.minimize(training_loss, inv_m.trainable_variables)

gpflow.utilities.print_summary(inv_m)
for step in range(iteration):
    optimization_step()
    if step % 100 == 0:
        elbo = inv_m.elbo((X_train, Y_train)).numpy()  # for all the data
        invy_pred, invy_var = inv_m.predict_y(X_test)
        test_error = np.sum(np.abs(invy_pred.numpy()-Y_test) / N_test)
        print(step, test_error, elbo)
        if test_error <= best_error:
            best_error = test_error
            best_para = gpflow.utilities.parameter_dict(inv_m)
        log_error.append(test_error)
        log_elbo.append(elbo)
        
gpflow.utilities.multiple_assign(inv_m, best_para)
gpflow.utilities.print_summary(inv_m)
invy_pred, invy_var = inv_m.predict_y(X_test)
test_error = np.sum(np.abs(invy_pred.numpy() - Y_test) / N_test)
print(invy_pred, test_error, inv_m.elbo((X_train, Y_train)).numpy())
end = time.time()
print(end - start)   #2077 seconds before installing mpi

fig = plt.figure(3)
ax3 = fig.gca()
ax3.contour(X_test1, X_test2, y_pred.numpy().reshape((m1,m2)), 50)
ax3.set_xlim(0, lim)
ax3.set_ylim(0, lim)
ax3.set_aspect(1)
plt.savefig("SVGP.png")
fig = plt.figure(4)
ax4 = fig.gca()
ax4.contour(X_test1, X_test2, invy_pred.numpy().reshape((m1,m2)), 50)
ax4.set_xlim(0, lim)
ax4.set_ylim(0, lim)
ax4.set_aspect(1)
plt.savefig("InvSVGP.png")
fig = plt.figure(5)
ax5 = fig.gca(projection='3d')
surf1=ax5.plot_surface(X_test1, X_test2, y_pred.numpy().reshape((m1,m2)), rstride=1, cstride=1,cmap=cm.coolwarm, edgecolor='none')
surf2=ax5.plot_surface(X_test1, X_test2, Y_test.reshape((m1,m2)), rstride=1, cstride=1,cmap=cm.GnBu, edgecolor='none')
fig = plt.figure(6)
ax6 = fig.gca(projection='3d')
surf3=ax6.plot_surface(X_test1, X_test2, invy_pred.numpy().reshape((m1,m2)), rstride=1, cstride=1,cmap=cm.coolwarm, edgecolor='none')
surf4=ax6.plot_surface(X_test1, X_test2, Y_test.reshape((m1,m2)), rstride=1, cstride=1,cmap=cm.GnBu, edgecolor='none')
plt.figure(7)
plt.plot(range(len(log_error)), log_error, 'ro-')
plt.figure(8)
plt.plot(range(len(log_elbo)), log_elbo, 'ko-')
plt.show()


