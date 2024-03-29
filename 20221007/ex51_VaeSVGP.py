# for figure 3 in the paper: binary classification
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
import time
from scipy.cluster.vq import kmeans2
from data_process import all_data, transform_data, filter_01, odd_vs_even
from utils import invlink
import joblib
from polya_gamma import PolyaGammaLikelihood
from pg_bernoulli import PolyaGammaBernoulli
# from pg_bernoulli1 import PolyaGammaBernoulli
from models.exVaeSVGP import VaeSVGP
from gpflow.ci_utils import ci_niter
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

#=================all data=================
alltrain_X, alltrain_Y, alltest_X, alltest_Y = all_data()
alltrain_N = np.shape(alltrain_X)[0]
alltest_N = np.shape(alltest_X)[0]

#=============transformed data=============
pick_train_N = int(alltrain_N/1)
pick_test_N = alltest_N
train_X, train_Y = transform_data(alltrain_X[:pick_train_N], alltrain_Y[:pick_train_N])
test_X, test_Y = transform_data(alltest_X[:pick_test_N], alltest_Y[:pick_test_N])

#===============chose binary================
# 1.zero and one binary
# c1 = 1; c2 = 0
# train_X, train_Y = filter_01(train_X, train_Y, c1, c2)
# test_X, test_Y = filter_01(test_X, test_Y, c1, c2)
# train_Y = tf.cast(train_Y == c1, dtype=gpflow.config.default_float())
# test_Y = tf.cast(test_Y == c1, dtype=gpflow.config.default_float())
# 2.odd and even binary
train_Y = tf.cast(train_Y % 2, dtype=gpflow.config.default_float())
test_Y = tf.cast(test_Y % 2, dtype=gpflow.config.default_float())
train_N, D = np.shape(train_X)
test_N = np.shape(test_X)[0]
print(np.shape(train_X), np.shape(train_Y))
print(np.shape(test_X), np.shape(test_Y))
print(np.unique(train_Y), np.unique(test_Y))
# print(train_X)
# set the model
M = 100 # since it is minibatch optimization
# M = int(train_N / 10)
# Z = np.random.random((M, D))
# Z = kmeans2(train_X, M, minit='points')[0]
Z = train_X[np.random.permutation(train_N)[:M], :]
C = 1
print(np.shape(Z))
iteration = ci_niter(50000)
loss = 0.0
# q_mu = np.random.random((M, C))
# q_sqrt = np.random.random((C, M, M))
q_mu = np.ones((M, C))
q_sqrt = np.ones((C, M, M))

# kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=np.random.uniform(size=D))
# kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=1.0 * np.ones(D))
kernel = gpflow.kernels.RBF(variance=10.0, lengthscales=np.log(np.mean(np.sum(Z**2, axis=1))) * np.ones(D))
likelihood = PolyaGammaBernoulli()
m = VaeSVGP(kernel,
            likelihood,
            inducing_variable=Z,
            num_latent_gps=C,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            # whiten=False,
            # q_diag=False,
            )

# set this two variable to be untrainable variable if the collapsed ELBO is chosen
gpflow.utilities.print_summary(m)

# #============================One: train all data========================
# opt = gpflow.optimizers.Scipy()
# opt_logs = opt.minimize(
# 	m.training_loss_closure((train_X, train_Y)), m.trainable_variables, options=dict(maxiter=ci_niter(iteration),
# 	                                                                                 disp=True)
# )
# print(opt_logs)
# predict_mu, predict_var = m.predict_y(test_X)
# predict_mu = (invlink(predict_mu) >0.5)
# test_error = np.sum(predict_mu != test_Y) / test_N
# print(test_error)
# loss = opt_logs['loss']
# plt.figure(1)
# plt.plot(range(len(loss)), loss, 'ro-')
# plt.show()

#============================method two=========================
minibatch_size = 200
log_elbo = []
log_error = []
log_time = []
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).repeat().shuffle(train_N)
train_iter = iter(train_dataset.batch(minibatch_size))
training_loss = m.training_loss_closure(train_iter, compile=True)
# boundaries = [10000, 20000, 50000, 80000]
# values = [0.03, 0.01, 0.005, 0.001, 0.0001]
# lr = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
lr = 0.001
optimizer = tf.optimizers.Adam(learning_rate=lr)
log_dir = f"./log/ex51_VaeSVGP_compiled"
model_task = ModelToTensorBoard(log_dir, m)
lml_task = ScalarToTensorBoard(log_dir, training_loss, "training_objective")
monitor = Monitor(MonitorTaskGroup([model_task, lml_task]))

@tf.function
def optimization_step(step, trainable_variables):
	optimizer.minimize(training_loss, var_list=trainable_variables)
	monitor(step)
	
start = time.time()
for step in tf.range(iteration):
	#1. train all parameters
	# optimization_step(step, m.trainable_variables)
	#2. once a time
	# for _ in range(10):
	# 	optimization_step(step, m.kernel.trainable_variables)
	# 	optimization_step(step, m.likelihood.trainable_variables)
	# 	optimization_step(step, m.inducing_variable.trainable_variables)
	# for _ in range(5):
	# 	optimization_step(step, m.q_mu.trainable_variables)
	# 	optimization_step(step, m.q_sqrt.trainable_variables)
	#3.batch a time
	for _ in range(5):
		variational_params = (
			m.q_mu.trainable_variables + m.q_sqrt.trainable_variables
		)
		optimization_step(step, variational_params)
	for _ in range(10):
		trainable_variables = (
			m.kernel.trainable_variables
			+ m.likelihood.trainable_variables
			+ m.inducing_variable.trainable_variables
		)
		optimization_step(step, trainable_variables)
	if step % 10 == 0:
		end = time.time()
		elbo = m.elbo((train_X, train_Y)).numpy()  # for all the data
		para_m_log = gpflow.utilities.parameter_dict(m)
		joblib.dump(para_m_log, './log/Adam_para_{}_ex51.pkl'.format(step))
		predict_mu, predict_var = m.predict_y(test_X)
		predict_mu = (invlink(predict_mu) >0.5)
		# print(predict_mu)
		test_error = np.sum(predict_mu != test_Y) / test_N
		print(step, elbo, test_error)
		log_time.append(end - start)
		log_error.append(test_error)
		log_elbo.append(elbo)

gpflow.utilities.print_summary(m)
print("length of unique kernel lengthscales: ", len(np.unique(m.kernel.lengthscales)))
# save and assign the parameters
para_m = gpflow.utilities.parameter_dict(m)
joblib.dump(para_m, "m_para_final_ex51.pkl")
para_m = joblib.load("m_para_final_ex51.pkl")
# print(para_m)
gpflow.utilities.multiple_assign(m, para_m)
predict_mu, predict_var = m.predict_y(test_X)
predict_mu = invlink(predict_mu)
np.savez('./log/test_Y_ex51.npz', XX=test_Y)
np.savez('./log/final_predict_ex51.npz', XX=predict_mu)
np.savez('./log/log_time_ex51.npz', XX=log_time)
np.savez('./log/log_error_ex51.npz', XX=log_error)
np.savez('./log/log_elbo_ex51.npz', XX=log_elbo)

plt.figure(1)
plt.plot(log_time, log_error, 'ro-')
plt.figure(2)
plt.plot(log_time, log_elbo, 'ro-')
plt.figure(3)
plt.plot(range(len(test_Y)), predict_mu.reshape(-1), 'ro', label="predicted result")
plt.legend(loc='best')
plt.figure(5)
plt.imshow(test_X[3].reshape(28, 28), cmap='gray')
plt.figure(6)
plt.imshow(test_X[4].reshape(28, 28), cmap='gray')
plt.figure(111)
plt.imshow(test_X[1].reshape(28, 28), cmap='gray')
plt.figure(112)
plt.imshow(test_X[2].reshape(28, 28), cmap='gray')
plt.show()

