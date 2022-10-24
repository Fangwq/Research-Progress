# for figure 3 in the paper: binary classification
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/fwq/PycharmProjects/my_code_cpu/invgp")
from kernels.image_transforms import rotate_img_angles, rotate_img_angles_stn
import tensorflow as tf
import gpflow
np.random.seed(1234)
tf.random.set_seed(1234)

def all_data(path="/home/fwq/PycharmProjects/my_code/RMT4ML-master/datasets/mnist.npz"):
	# prepare the data
	mnist_data = np.load(path, allow_pickle=True)
	train_X = mnist_data['x_train']   # (60000, 28, 28)
	train_Y = mnist_data['y_train']   # (60000, )
	test_X = mnist_data['x_test']     # (10000, 28, 28)
	test_Y = mnist_data['y_test']
	# a, b = train_X.max(), test_X.max()
	# print(a, b)
	# train_X = train_X/a
	# test_X = test_X/b
	N = np.shape(train_X)[0]
	# print("before permutation: ", train_X[:1], train_Y[:10])
	index = np.random.permutation(N)
	train_X = train_X[index]
	train_Y = train_Y[index]
	# print("after permutation: ", train_X[:1], train_Y[:10])
	return train_X, train_Y.reshape((-1, 1)), test_X, test_Y.reshape((-1, 1))
	
def transform_data(X, Y):
	row, _, _ = np.shape(X)
	after_transform = np.ones((row, 784))  # (60000, 784)
	alpha = 90.0 # or 180
	for i in range(row):
		angle = - alpha + 2*alpha*np.random.random()
		# angle = [alpha, -alpha][np.random.randint(0, 2)]
		after_transform[i] = rotate_img_angles(tf.expand_dims(X[i], axis=0), np.array([angle]),
		                                   "nearest").numpy().reshape(-1, 784)
	mean = np.mean(after_transform, axis=0)
	std = np.std(after_transform, axis=0)
	std[std == 0] = 1.0
	after_transform = (after_transform - mean)/std
	return after_transform, Y

def filter_01(X, Y, c1, c2):
	lbls01 = np.logical_or(Y == c1, Y == c2).flatten()
	return X[lbls01, :], Y[lbls01]

def odd_vs_even(X, Y, c1, c2):
	lbls01 = np.logical_or(Y % 2 == c1, Y % 2 == c2).flatten()
	return X[lbls01, :], Y[lbls01]

# print("====================before transform======================")
# train_X, train_Y, test_X, test_Y = all_data()
# print(np.shape(train_X), np.shape(train_Y))
# print(np.shape(test_X), np.shape(test_Y))
# print(np.unique(train_Y), np.unique(test_Y))
# print(train_Y[0], test_Y[0])
# plt.figure(1)
# plt.imshow(train_X[0], cmap='gray')
# plt.figure(2)
# plt.imshow(test_X[0], cmap='gray')
# # plt.show()
#
#
# print("===================after transform====================")
# train_X, train_Y = transform_data(train_X, train_Y)
# test_X, test_Y = transform_data(test_X, test_Y)
# print(np.shape(train_X), np.shape(train_Y))
# print(np.shape(test_X), np.shape(test_Y))
# print(np.unique(train_Y), np.unique(test_Y))
# print(train_Y[0], test_Y[0])
# plt.figure(3)
# plt.imshow(train_X[0].reshape((28, 28)), cmap='gray')
# plt.figure(4)
# plt.imshow(test_X[0].reshape((28, 28)), cmap='gray')
# # plt.show()
#
# print("===================after filter====================")
# c1 = 1; c2 = 0
# train_X, train_Y = filter_01(train_X, train_Y, c1, c2)
# test_X, test_Y = filter_01(test_X, test_Y,  c1, c2)
# train_Y = tf.cast(train_Y == c1, dtype=gpflow.config.default_float())
# test_Y = tf.cast(test_Y == c1, dtype=gpflow.config.default_float())
# print(np.shape(train_X), np.shape(train_Y))
# print(np.shape(test_X), np.shape(test_Y))
# print(np.unique(train_Y), np.unique(test_Y))
# print(train_Y[0], test_Y[0])
# plt.figure(5)
# plt.imshow(train_X[0].reshape((28, 28)), cmap='gray')
# plt.figure(6)
# plt.imshow(test_X[0].reshape((28, 28)), cmap='gray')
# plt.show()

