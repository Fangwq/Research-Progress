import time

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from gpflow.optimizers import NaturalGradient
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append("/home/fwq/PycharmProjects/my_code_cpu/t-SVGP/")
from experiments.util import data_load
from src.models.tsvgp import t_SVGP

# Define parameters
n_e_steps = 6
n_m_steps = 14
nat_lr = 0.75
adam_lr = 0.2
M = 10
nm = 3  # number of models [svgp, svgp_nat, t-svgp]
nit = 1000
t_nit = n_e_steps * nit + n_m_steps * nit

mb_size = "full"
n_folds = 1

data_name = "banana"  # Script can run:'diabetes', 'ionosphere', 'sonar'
optim = "Adam"

rng = np.random.RandomState(12)
tf.random.set_seed(4)


def init_model(n_train):
	models = []
	names = []
	
	# ============Define SVGP
	m = gpflow.models.SVGP(
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
		inducing_variable=Z.copy(),
		num_data=n_train,
	)
	# gpflow.utilities.print_summary(m)
	models.append(m)
	names.append("svgp")
	
	# ===============Define whiten SVGP
	m_svgp_white = gpflow.models.SVGP(
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
		inducing_variable=Z.copy(),
		num_data=n_train,
		whiten=True,
	)
	
	gpflow.set_trainable(m_svgp_white.q_mu, False)
	gpflow.set_trainable(m_svgp_white.q_sqrt, False)
	
	models.append(m_svgp_white)
	names.append("svgp_white")
	
	# ===============Define t_SVGP
	m_tsvgp = t_SVGP(
		kernel=gpflow.kernels.Matern52(
			lengthscales=np.ones((1, x.shape[1])) * ell, variance=var
		),
		likelihood=gpflow.likelihoods.Bernoulli(),
		inducing_variable=Z.copy(),
		num_data=n_train,
	)
	
	# Turn off natural params
	# gpflow.utilities.print_summary(m_tsvgp)
	gpflow.set_trainable(m_tsvgp.lambda_1, False)
	gpflow.set_trainable(m_tsvgp.lambda_2_sqrt, False)
	
	models.append(m_tsvgp)
	names.append("tsvgp")
	
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
	data = (x, y)
	training_loss = model.training_loss_closure(data, compile=True)
	# @tf.function
	def optimization_step_tsvgp(model, training_loss):
		model.natgrad_step(data, lr=nat_lr)
	
	@tf.function
	def optimization_step(model, training_loss, params):
		optimizer.minimize(training_loss, var_list=params)

	theta = np.linspace(0.1, 5, iterations)
	for step in range(iterations):
		
		if model.name == "svgp" and model.q_mu.trainable == False:
			gpflow.set_trainable(model.q_mu, True)
			gpflow.set_trainable(model.q_sqrt, True)
			variational_params = (
						model.q_mu.trainable_variables + model.q_sqrt.trainable_variables
				)
			optimization_step(model, training_loss, variational_params)
			model.kernel.variance.assign(theta[step])
			elbo = model.maximum_log_likelihood_objective(data).numpy()
			logf.append(elbo)
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
			gpflow.set_trainable(model.q_mu, False)
			gpflow.set_trainable(model.q_sqrt, False)
		elif model.name == "t_svgp":
			# for i in range(n_e_steps):
			optimization_step_tsvgp(model, training_loss)
			model.kernel.variance.assign(theta[step])
			elbo = model.maximum_log_likelihood_objective(data).numpy()
			logf.append(elbo)
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
		else:
			variational_params = (
						model.q_mu.trainable_variables + model.q_sqrt.trainable_variables
				)
			optimization_step(model, training_loss, variational_params)
			model.kernel.variance.assign(theta[step])
			elbo = model.maximum_log_likelihood_objective(data).numpy()
			logf.append(elbo)
			nlpd.append(-tf.reduce_mean(model.predict_log_density((xt, yt))).numpy())
	
	return logf, nlpd




ell = 1.0
var = 1.0

data, test = data_load(data_name, split=1.0, normalize=False)
X, Y = data
X_scaler = StandardScaler().fit(X)
X = X_scaler.transform(X)

N = X.shape[0]
D = X.shape[1]
# Initialize inducing locations to the first M inputs in the dataset

# kmeans = KMeans(n_clusters=M, random_state=0).fit(X)
# Z = kmeans.cluster_centers_
Z = X[:M, :].copy()

RMSE = np.zeros((nm, n_folds))
ERRP = np.zeros((nm, n_folds))
NLPD = np.zeros((nm, n_folds))
TIME = np.zeros((nm, n_folds))

NLPD_i = np.zeros((nm, nit, n_folds))
LOGF_i = np.zeros((nm, nit, n_folds))

fold = 0
train_index = int(N)

# The data split
x = X[:train_index]
y = Y[:train_index]
xt = X[:train_index]
yt = Y[:train_index]
	
if mb_size == "full":
	mb_size = x.shape[0]

mods, names = init_model(x.shape[0])
	
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
plt.plot(np.linspace(0.1, 5, nit), elbo_mean[0, :][:], "r-", label=names[0])
plt.plot(np.linspace(0.1, 5, nit), elbo_mean[1, :][:], "b--", label=names[1])
plt.plot(np.linspace(0.1, 5, nit), elbo_mean[2, :][:], "k-.", label=names[2])
plt.legend()

plt.figure(2)
plt.title("NLPD" + "_" + data_name)
plt.plot(np.linspace(0.1, 5, nit), nlpd_i_mean[0, :][:], label=names[0])
plt.plot(np.linspace(0.1, 5, nit), nlpd_i_mean[1, :][:], label=names[1])
plt.plot(np.linspace(0.1, 5, nit), nlpd_i_mean[2, :][:], label=names[2])
plt.legend()

# Report
print("Data: {}, n: {}, m: {}, steps: {}".format(data_name, x.shape[0], mb_size, nit))
print("{:<14} {:^13}   {:^13}  ".format("Method", "NLPD", "ERRP"))

for i in range(len(mods)):
	print(
		"{:<14} {:.3f}+/-{:.3f}   {:.3f}+/-{:.3f}   ".format(
			names[i], nlpd_mean[i], nlpd_std[i], rmse_mean[i], rmse_std[i]
		)
	)
plt.show()