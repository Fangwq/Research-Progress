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
from models.exInvVaeSVGP import InvVaeSVGP
import copy
from kernels.invariant import Invariant, StochasticInvariant
from kernels.orbits import ImageRotation
import horovod.tensorflow as hvd
import os
# from gpflow.ci_utils import ci_niter

hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
cuda = False
if cuda:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#===================all data==================
alltrain_X, alltrain_Y, alltest_X, alltest_Y = all_data()
alltrain_N = np.shape(alltrain_X)[0]
alltest_N = np.shape(alltest_X)[0]

#================transformed data==============
pick_train_N = int(alltrain_N/1)
pick_test_N = alltest_N
train_X, train_Y = transform_data(alltrain_X[:pick_train_N], alltrain_Y[:pick_train_N])
test_X, test_Y = transform_data(alltest_X[:pick_test_N], alltest_Y[:pick_test_N])

#=================chose binary=================
# 1.zero and one binary
# c1 = 1; c2 = 0
# train_X, train_Y = filter_01(train_X, train_Y, c1, c2)
# test_X, test_Y = filter_01(test_X, test_Y, c1, c2)
# train_Y = tf.cast(train_Y == c1, dtype=gpflow.config.default_float())
# test_Y = tf.cast(test_Y == c1, dtype=gpflow.config.default_float())
# 2.odd and even binary
train_Y = tf.cast(train_Y % 2, dtype=gpflow.config.default_float())
test_Y = tf.cast(test_Y % 2, dtype=gpflow.config.default_float())
# print(train_X)
# print(gpflow.config.default_float())
train_N, D = np.shape(train_X)
test_N = np.shape(test_X)[0]
print(np.shape(train_X), np.shape(train_Y))
print(np.shape(test_X), np.shape(test_Y))
print(np.unique(train_Y), np.unique(test_Y))

# set the model
M = 100 # since it is minibatch optimization
# M = int(train_N / 10)
# Z = np.random.random((M, D))
# Z = kmeans2(train_X, M, minit='points')[0]
Z = train_X[np.random.permutation(train_N)[:M], :]
C = 1
print(np.shape(Z))
iteration = 50000
loss = 0.0
# q_mu = np.random.random((M, C))
# q_sqrt = np.random.random((C, M, M))
# q_sqrt = np.random.random((C, M, M))
q_mu = np.ones((M, C))
q_sqrt = np.ones((C, M, M))

# kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=np.random.uniform(size=D))
# kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=1.0 * np.ones(D))
kernel = gpflow.kernels.RBF(variance=2.0, lengthscales=np.log(np.mean(np.sum(Z**2, axis=1))) * np.ones(D))
likelihood = PolyaGammaBernoulli()
# set the use_stn to true, or the gradient to parameter angle will be none since the function will be discontinuous
inv_kernel = StochasticInvariant(kernel, ImageRotation(orbit_size=8, minibatch_size =8, img_size=28, use_stn=True))    # set invariant kernel

m = InvVaeSVGP(inv_kernel,
            likelihood,
            inducing_variable=Z,
            num_latent_gps=C,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            # whiten=False,
            # q_diag=False,
            )
# set this two variable to be untrainable variable if the collapsed ELBO is chosen
if hvd.rank() == 0:
	gpflow.utilities.print_summary(m)

minibatch_size = 200
log_elbo = []
log_error = []
log_time = []
log_angle = []
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).repeat().shuffle(train_N)
train_iter = iter(train_dataset.batch(minibatch_size))
# boundaries = [10000, 20000, 50000, 80000]
# values = [0.1, 0.05, 0.01, 0.001, 0.0001]
# lr = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
lr = 0.001 #* hvd.size()
optimizer = tf.optimizers.Adam(learning_rate=lr)
checkpoint_dir = './log/ex51_Inv_horovod.ckpt'
checkpoint = tf.train.Checkpoint(model=m, optimizer=optimizer)
compression = hvd.Compression.fp16

# this method does not work in this case because of inversion problem (so strange)
# and 'InvVaeSVGP' object has no attribute 'compile' for another code style.
# In addition, I change the optimization plan, but the inversion problem is still there.
# I give up.
@tf.function
def training_step(X, Y, first_batch, trainable_variables):
	with tf.GradientTape() as tape:
		loss = -m.elbo((X, Y))
	tape = hvd.DistributedGradientTape(tape, compression=compression)
	# grads = tape.gradient(loss, m.trainable_variables)
	# optimizer.apply_gradients(zip(grads, m.trainable_variables))
	grads = tape.gradient(loss, trainable_variables)
	optimizer.apply_gradients(zip(grads, trainable_variables))
	if first_batch:
		# hvd.broadcast_variables(m.variables, root_rank=0)
		hvd.broadcast_variables(trainable_variables, root_rank=0)
		hvd.broadcast_variables(optimizer.variables(), root_rank=0)
	return loss

if hvd.rank() == 0:
	print("=============================training begins============================")

start = time.time()
for step in range(iteration):
	temp_X, temp_Y = next(train_iter)
	# elbo = training_step(temp_X, temp_Y, step == 0, m.trainable_variables).numpy()
	for _ in range(5):
		variational_params = (
			m.q_mu.trainable_variables + m.q_sqrt.trainable_variables
		)
		elbo = training_step(temp_X, temp_Y, step == 0, variational_params).numpy()
	for _ in range(10):
		trainable_variables = (
			m.kernel.trainable_variables
			+ m.likelihood.trainable_variables
			+ m.inducing_variable.trainable_variables
		)
		elbo = training_step(temp_X, temp_Y, step == 0, trainable_variables).numpy()
	if step % 10 == 0 and hvd.rank() == 0:
		end = time.time()
		para_m_log = gpflow.utilities.parameter_dict(m)
		joblib.dump(para_m_log, './log/Adam_para_{}_ex51_Inv_horovod.pkl'.format(step))
		predict_mu, predict_var = m.predict_y(test_X)
		predict_mu = (invlink(predict_mu) >0.5)
		# print(predict_mu)
		test_error = np.sum(predict_mu != test_Y) / test_N
		print(step, elbo, test_error, m.kernel.orbit.angle)
		log_time.append(end - start)
		log_error.append(test_error)
		log_elbo.append(elbo)
		log_angle.append(m.kernel.orbit.angle.numpy())


if hvd.rank() == 0:
	gpflow.utilities.print_summary(m)
	print("length of unique kernel lengthscales: ", len(np.unique(m.kernel.basekern.lengthscales)))
	# save and assign the parameters
	para_m = gpflow.utilities.parameter_dict(m)
	joblib.dump(para_m, "m_para_final_ex51_Inv_horovod.pkl")
	para_m = joblib.load("m_para_final_ex51_Inv_horovod.pkl")
	# print(para_m)
	gpflow.utilities.multiple_assign(m, para_m)
	predict_mu, predict_var = m.predict_y(test_X)
	predict_mu = invlink(predict_mu)
	np.savez('./log/test_Y_ex51_Inv_horovod.npz', XX=test_Y)
	np.savez('./log/final_predict_ex51_Inv_horovod.npz', XX=predict_mu)
	np.savez('./log/log_time_ex51_Inv_horovod.npz', XX=log_time)
	np.savez('./log/log_error_ex51_Inv_horovod.npz', XX=log_error)
	np.savez('./log/log_elbo_ex51_Inv_horovod.npz', XX=log_elbo)
	np.savez('./log/log_angle_ex51_Inv_horovod.npz', XX=log_angle)
	checkpoint.save(checkpoint_dir)
	
	plt.figure(1)
	plt.plot(log_time, log_error, 'ro-')
	plt.figure(2)
	plt.plot(log_time, log_elbo, 'ro-')
	plt.figure(3)
	plt.plot(log_time, log_angle, 'ro-', label="parameter angle")
	plt.legend(loc='best')
	plt.figure(4)
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


