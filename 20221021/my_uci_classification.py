import time

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from gpflow.optimizers import NaturalGradient
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append("/home/fwq/PycharmProjects/my_code_cpu/t-SVGP/")
from experiments.util import data_load
from src.models.tsvgp import t_SVGP
from src.models.tsvgp_white import t_SVGP_white
from src.models.tsvgp_sites import t_SVGP_sites
from src.models.tvgp import t_VGP

# Define parameters
n_e_steps = 8
n_m_steps = 20
nat_lr = 0.8
adam_lr = 0.1
M = 50
nm = 6  # number of models [svgp, svgp_nat, t-svgp, t-vgp, t_svgp_sites, tsvgp_white]
nit = 20
t_nit = n_e_steps * nit + n_m_steps * nit

mb_size = "full"
n_folds = 5

data_name = "diabetes"  # Script can run:'diabetes', 'ionosphere', 'sonar', 'banana'
optim = "Adam"

rng = np.random.RandomState(19)
tf.random.set_seed(19)


def init_model(n_train, train_dataset):
	models = []
	names = []
	
	# ========================Define t_SVGP_white
	m_tsvgp_white = t_SVGP_white(
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
		inducing_variable=Z.copy(),
		num_data=n_train,
	)
	
	# Turn off natural params
	gpflow.set_trainable(m_tsvgp_white.lambda_1, False)
	gpflow.set_trainable(m_tsvgp_white.lambda_2, False)
	
	models.append(m_tsvgp_white)
	names.append("tsvgp_white")
	
	# ========================Define t_SVGP_sites
	m_tsvgp_sites = t_SVGP_sites(
		data=train_dataset,
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
		inducing_variable=Z.copy(),
	)
	
	# Turn off natural params
	gpflow.set_trainable(m_tsvgp_sites.lambda_1, False)
	gpflow.set_trainable(m_tsvgp_sites.lambda_2, False)
	
	models.append(m_tsvgp_sites)
	names.append("tsvgp_sites")
	
	# ======================Define standard SVGP
	m = gpflow.models.SVGP(
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
		inducing_variable=Z.copy(),
		num_data=n_train,
		whiten=True,
	)
	
	models.append(m)
	names.append("svgp")
	
	# ======================Define natgrad SVGP
	m_svgp_nat = gpflow.models.SVGP(
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
		inducing_variable=Z.copy(),
		num_data=n_train,
		whiten=True,
	)
	
	gpflow.set_trainable(m_svgp_nat.q_mu, False)
	gpflow.set_trainable(m_svgp_nat.q_sqrt, False)
	
	models.append(m_svgp_nat)
	names.append("svgp_nat")
	
	# =======================Define t_SVGP
	m_tsvgp = t_SVGP(
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
		inducing_variable=Z.copy(),
		num_data=n_train,
	)
	
	# Turn off natural params
	gpflow.set_trainable(m_tsvgp.lambda_1, False)
	gpflow.set_trainable(m_tsvgp.lambda_2_sqrt, False)
	
	models.append(m_tsvgp)
	names.append("tsvgp")
	
	# ========================Define t_VGP
	m_tvgp = t_VGP(
		data=train_dataset,
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
	)
	
	# Turn off natural params
	gpflow.set_trainable(m_tvgp.lambda_1, False)
	gpflow.set_trainable(m_tvgp.lambda_2, False)
	
	models.append(m_tvgp)
	names.append("tvgp")
	
	return models, names


def run_optim(model, iterations):
	"""
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
	
	# Create an Adam Optimizer action
	logf = []
	nlpd = []
	
	natgrad_opt = NaturalGradient(gamma=nat_lr)
	
	if optim == "Adam":
		optimizer = tf.optimizers.Adam(adam_lr)
	
	elif optim == "SGD":
		optimizer = tf.optimizers.SGD(adam_lr)
	
	optimizer2 = tf.optimizers.Adam(nat_lr)
	
	train_iter = iter(train_dataset.batch(mb_size))
	print(model.name)
	# should add ExternalDataTrainingLossMixin or InternalDataTrainingLossMixin class to get training_loss_closure method
	if model.name != "t_vgp" and model.name != "t_svgp_sites":
		training_loss = model.training_loss_closure(train_iter, compile=True)  # ExternalDataTrainingLossMixin
	else:
		training_loss = model.training_loss_closure(compile=True)  # InternalDataTrainingLossMixin
	
	# @tf.function
	def optimization_step_nat(training_loss, variational_params):
		natgrad_opt.minimize(training_loss, var_list=variational_params)
	
	@tf.function
	def optimization_step_tsvgp(model, training_loss):
		model.natgrad_step(data, lr=nat_lr)
	
	@tf.function
	def optimization_step_tsvgp_sites(model, training_loss):
		model.natgrad_step(lr=nat_lr)
	
	@tf.function
	def optimization_step(model, training_loss, params):
		optimizer.minimize(training_loss, var_list=params)
	
	@tf.function
	def optimization_step2(model, training_loss, params):
		optimizer2.minimize(training_loss, var_list=params)
	
	for step in range(iterations):
		data = next(train_iter)
		
		if model.name == "svgp" and model.q_mu.trainable == False:
			variational_params = [(model.q_mu, model.q_sqrt)]
			# print("SVGP_nat_1: ", variational_params)
			for i in range(n_e_steps):
				optimization_step_nat(training_loss, variational_params)
			# print("SVGP_nat_2: ", [(model.q_mu, model.q_sqrt)])
			elbo = model.maximum_log_likelihood_objective(data).numpy()
			logf.append(elbo)
			
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
			
			for j in range(n_m_steps):
				optimization_step(model, training_loss, model.trainable_variables)
		
		elif model.name == "t_svgp":
			# print("t_svgp_1: ", model.trainable_variables)
			for i in range(n_e_steps):
				optimization_step_tsvgp(model, training_loss)
			
			elbo = model.maximum_log_likelihood_objective(data).numpy()
			logf.append(elbo)
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
			# gpflow.utilities.print_summary(model)
			for i in range(n_m_steps):
				optimization_step(model, training_loss, model.trainable_variables)
			# print("t_svgp_2: ", model.trainable_variables)
		
		elif model.name == "svgp" and model.q_mu.trainable == True:
			
			for i in range(n_e_steps):
				variational_params = (
						model.q_mu.trainable_variables + model.q_sqrt.trainable_variables
				)
				# print("SVGP: ", variational_params)
				optimization_step2(model, training_loss, variational_params)
			
			elbo = model.maximum_log_likelihood_objective(data).numpy()
			logf.append(elbo)
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
			
			for i in range(n_m_steps):
				trainable_variables = (
						model.kernel.trainable_variables
						+ model.likelihood.trainable_variables
						+ model.inducing_variable.trainable_variables
				)
				# print("SVGP: ", trainable_variables)
				optimization_step(model, training_loss, trainable_variables)
		
		elif model.name == "t_vgp":
			# for i in range(n_e_steps):
			#     model.update_variational_parameters()
			
			elbo = model.maximum_log_likelihood_objective().numpy()[0][0]
			logf.append(elbo)
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
			
			for i in range(n_m_steps):
				model.update_variational_parameters()
				trainable_variables = (
						model.kernel.trainable_variables
						+ model.likelihood.trainable_variables
				)
				optimization_step(model, training_loss, trainable_variables)
			# print("t_svgp_2: ", model.trainable_variables)
		
		elif model.name == "t_svgp_white":
			# print("t_svgp_white_1: ", model.trainable_variables)
			for i in range(n_e_steps):
				optimization_step_tsvgp(model, training_loss)
			# print("1")
			# gpflow.utilities.print_summary(model)
			
			elbo = model.maximum_log_likelihood_objective(data).numpy()
			logf.append(elbo)
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
			for i in range(n_m_steps):
				optimization_step(model, training_loss, model.trainable_variables)
			# print("2")
			# gpflow.utilities.print_summary(model)
			# print("t_svgp_white_2: ", model.trainable_variables)
		
		elif model.name == "t_svgp_sites":
			# print("t_svgp_sites_1: ", model.trainable_variables)
			for i in range(n_e_steps):
				optimization_step_tsvgp_sites(model, training_loss)
			elbo = model.maximum_log_likelihood_objective().numpy()
			logf.append(elbo)
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
			for i in range(n_m_steps):
				trainable_variables = (
						model.kernel.trainable_variables
						+ model.likelihood.trainable_variables
						+ model.inducing_variable.trainable_variables
				)
				# gpflow.utilities.print_summary(model)
				optimization_step(model, training_loss, trainable_variables)
			# # print("t_svgp_sites_2: ", model.trainable_variables)
	
	return logf, nlpd


ell = 1.0
var = 1.0

if data_name == "elevators":
	# Load all the data
	data = np.array(loadmat("../../demos/data/elevators.mat")["data"])
	X = data[:, :-1]
	Y = data[:, -1].reshape(-1, 1)
else:
	data, test = data_load(data_name, split=1.0, normalize=False)
	X, Y = data

X_scaler = StandardScaler().fit(X)
# Y_scaler = StandardScaler().fit(Y)
X = X_scaler.transform(X)
# Y = Y_scaler.transform(Y)
N = X.shape[0]
D = X.shape[1]
print("0: ", np.unique(Y))
if (np.unique(Y) == [-1, 1]).all():
	Y = (Y+1)/2.0
print("1: ", np.unique(Y))
# Initialize inducing locations to the first M inputs in the dataset
# kmeans = KMeans(n_clusters=M, random_state=0).fit(X)
# Z = kmeans.cluster_centers_
Z = X[:M, :].copy()

kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)

RMSE = np.zeros((nm, n_folds))
ERRP = np.zeros((nm, n_folds))
NLPD = np.zeros((nm, n_folds))
TIME = np.zeros((nm, n_folds))

NLPD_i = np.zeros((nm, nit, n_folds))
LOGF_i = np.zeros((nm, nit, n_folds))

fold = 0
for train_index, test_index in kf.split(X):
	
	# The data split
	x = X[train_index]
	y = Y[train_index]
	xt = X[test_index]
	yt = Y[test_index]
	
	if mb_size == "full":
		mb_size = x.shape[0]
	
	train_dataset = (
		tf.data.Dataset.from_tensor_slices((x, y)).repeat().shuffle(x.shape[0])
	)
	
	mods, names = init_model(x.shape[0], (x, y))
	
	maxiter = ci_niter(nit)
	
	j = 0
	
	for m in mods:
		print(m)
		t0 = time.time()
		logf_i, nlpd_i = run_optim(m, maxiter)
		t = time.time() - t0
		
		nlpd = -tf.reduce_mean(m.predict_log_density((xt, yt))).numpy()
		
		yp, _ = m.predict_y(xt)
		errp = 1.0 - np.sum((yp > 0.5) == (yt > 0.5)) / yt.shape[0]
		
		print("NLPD for model {} with name {}: {}".format(m.name, names[j], nlpd))
		print("ERR% for model {} with name {}: {}".format(m.name, names[j], errp))
		
		# Store results
		ERRP[j, fold] = errp
		NLPD[j, fold] = nlpd
		TIME[j, fold] = t
		
		NLPD_i[j, :, fold] = np.array(nlpd_i)
		LOGF_i[j, :, fold] = np.array(logf_i)
		
		j += 1
	
	fold += 1

# Calculate averages and standard deviations
rmse_mean = np.mean(ERRP, 1)
rmse_std = np.std(ERRP, 1)
nlpd_mean = np.mean(NLPD, 1)
nlpd_std = np.std(NLPD, 1)
time_mean = np.mean(TIME, 1)
time_std = np.std(TIME, 1)

elbo_mean = np.mean(LOGF_i, 2)
nlpd_i_mean = np.mean(NLPD_i, 2)

plt.figure(1)
plt.title("ELBO" + "_" + data_name)
plt.plot(range(nit), elbo_mean[0, :][:], label=names[0])
plt.plot(range(nit), elbo_mean[1, :][:], label=names[1])
plt.plot(range(nit), elbo_mean[2, :][:], label=names[2])
plt.plot(range(nit), elbo_mean[3, :][:], label=names[3])
plt.plot(range(nit), elbo_mean[4, :][:], label=names[4])
plt.plot(range(nit), elbo_mean[5, :][:], label=names[5])
plt.legend()

plt.figure(2)
plt.title("NLPD" + "_" + data_name)
plt.plot(range(nit), nlpd_i_mean[0, :][:], label=names[0])
plt.plot(range(nit), nlpd_i_mean[1, :][:], label=names[1])
plt.plot(range(nit), nlpd_i_mean[2, :][:], label=names[2])
plt.plot(range(nit), nlpd_i_mean[3, :][:], label=names[3])
plt.plot(range(nit), nlpd_i_mean[4, :][:], label=names[4])
plt.plot(range(nit), nlpd_i_mean[5, :][:], label=names[5])
plt.legend()

# Report
print("Data: {}, n: {}, m: {}, steps: {}".format(data_name, x.shape[0], mb_size, nit))
print("{:<14} {:^13}   {:^13}  ".format("Method", "NLPD", "RMSE"))

for i in range(len(mods)):
	print(
		"{:<14} {:.3f}+/-{:.3f}   {:.3f}+/-{:.3f}  ".format(
			names[i], nlpd_mean[i], nlpd_std[i], rmse_mean[i], rmse_std[i]
		)
	)
plt.show()

