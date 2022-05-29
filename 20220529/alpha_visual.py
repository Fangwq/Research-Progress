from grid import load_grouped, load
import matplotlib.pyplot as plt
import numpy as np
import pickle

# with open("Stripe1LalphaKernel_test1.txt", 'rb') as f:
# 	runs = pickle.loads(f)
# 	print(runs)

args, groups = load_grouped('./Stripe1LalphaKernel_test1', ['alpha'])
index = 0
alpha = groups[0][1][index]['args'].alpha
result = groups[0][1][index]['regular']['dynamics']


t = range(len(result))
w1 = list(map(lambda x: x['w'][0], result))
w2 = list(map(lambda x: x['w'][1], result))
beta = list(map(lambda x: x['beta'], result))
b = list(map(lambda x: x['b'], result))
train_err = list(map(lambda x: x['train']['err'], result))
test_err = list(map(lambda x: x['test']['err'], result))
train_loss = list(map(lambda x: x['train']['loss'], result))
test_loss = list(map(lambda x: x['test']['loss'], result))
train_fnorm = list(map(lambda x: x['train']['fnorm'], result))
test_fnorm = list(map(lambda x: x['test']['fnorm'], result))
neural_w = np.array(list(map(lambda x: x['neurons']['w'], result)))
neural_b = np.array(list(map(lambda x: x['neurons']['b'], result)))
neural_beta = np.array(list(map(lambda x: x['neurons']['beta'], result)))
neural_fun100 = result[100]['function']['x1'].detach().numpy()
neural_fun200 = result[200]['function']['x1'].detach().numpy()
neural_fun300 = result[300]['function']['x1'].detach().numpy()
x1 = np.linspace(-3, 3, 100)
print(np.shape(neural_w))


# for kernel
kernel_result = groups[0][1][index]['init_kernel']['dynamics']
kernel_t = range(len(kernel_result))
kernel_train_err = list(map(lambda x: x['train']['err'], kernel_result))
kernel_test_err = list(map(lambda x: x['test']['err'], kernel_result))
kernel_train_loss = list(map(lambda x: x['train']['loss'], kernel_result))
kernel_test_loss = list(map(lambda x: x['test']['loss'], kernel_result))

kernel_ptr_result = groups[0][1][index]['init_kernel_ptr']['dynamics']
kernel_ptr_t = range(len(kernel_ptr_result))
kernel_ptr_train_err = list(map(lambda x: x['train']['err'], kernel_ptr_result))
kernel_ptr_test_err = list(map(lambda x: x['test']['err'], kernel_ptr_result))
kernel_ptr_train_loss = list(map(lambda x: x['train']['loss'], kernel_ptr_result))
kernel_ptr_test_loss = list(map(lambda x: x['test']['loss'], kernel_ptr_result))




# import numpy as np
# import matplotlib.pyplot as plt
#
# V = np.array([[1,1], [-2,2], [4,-7]])
# origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point

# plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)


plt.figure(1)
plt.plot(t, w1, 'b-', label = r'$w_1$')
plt.plot(t, w2, 'k-', label = r'$w_2$')
plt.plot(t, beta, 'c-', label=r'$beta$')
plt.plot(t, b, 'r-', label=r'$b$')
plt.title(r'$\alpha$'+'={}'.format(alpha))
plt.legend(loc='best')
plt.yscale('log')

plt.figure(2)
plt.plot(t, train_err, 'r-', label=r'$train_{err}$')
plt.plot(t, test_err, 'b-', label=r'$test_{err}$')
plt.yscale('log')
plt.legend(loc='best')

plt.figure(3)
plt.plot(t, train_loss, 'r-', label=r'$train_{loss}$')
plt.plot(t, test_loss, 'b-', label=r'$test_{loss}$')
plt.legend(loc='best')
plt.yscale('log')

plt.figure(4)
plt.plot(t, train_loss, 'r-', label=r'$train_{loss}$')
plt.plot(t, test_err, 'b-', label=r'$test_{err}$')
plt.legend(loc='best')
plt.yscale('log')
plt.figure(5)
plt.plot(t, train_fnorm, 'r-', label=r'$train_{norm}$')
plt.plot(t, test_fnorm, 'b-', label=r'$test_{norm}$')
plt.legend(loc='best')
plt.yscale('log')
fig, axs = plt.subplots(1,3, figsize=(10, 10), sharex='col', sharey='row')
_, n, _ = np.shape(neural_w)
origin = np.zeros((2,n)) # origin point
# plt.quiver(*origin, neural_w[3,:,0], V[3,:,1], scale=21)
axs[0].quiver(*origin, neural_beta[3]*neural_w[3,:,0], neural_beta[3]*neural_w[3,:,1], scale=30)
axs[1].quiver(*origin, neural_beta[int(t[-1]/2)]*neural_w[int(t[-1]/2),:,0], neural_beta[int(t[-1]/2)]*neural_w[int(t[-1]/2),:,1], scale=30)
axs[2].quiver(*origin, neural_beta[t[-1]-1]*neural_w[t[-1]-1,:,0], neural_beta[t[-1]-1]*neural_w[t[-1]-1,:,1], scale=30)
axs[2].quiver(*origin, neural_beta[t[-1]-1]*neural_w[t[-1]-1,:,0], neural_beta[t[-1]-1]*neural_w[t[-1]-1,:,1], scale=30)
fig, axs = plt.subplots(3,1, figsize=(10, 10))
axs[0].plot(x1, neural_fun100, 'r-', label='100')
axs[0].legend(loc='best')
axs[1].plot(x1, neural_fun200, 'b-', label='200')
axs[1].legend(loc='best')
axs[2].plot(x1, neural_fun300, 'c-', label='300')
axs[2].legend(loc='best')
# ================================check for the kernel

plt.figure(8)
cut = 100
plt.plot(kernel_t[cut:], kernel_train_err[cut:], 'r-', label=r'$kernel_{train_{err}}$')
plt.plot(kernel_t[cut:], kernel_test_err[cut:], 'b-', label=r'$kernel_{test_{err}}$')
plt.legend(loc='best')
# plt.yscale('log')
plt.figure(9)
plt.plot(kernel_t[cut:], kernel_train_loss[cut:], 'r-', label=r'$kernel_{train_{loss}}$')
plt.plot(kernel_t[cut:], kernel_test_loss[cut:], 'b-', label=r'$kernel_{test_{loss}}$')
plt.legend(loc='best')
# plt.yscale('log')
plt.figure(10)
plt.plot(kernel_ptr_t[cut:], kernel_ptr_train_err[cut:], 'r-', label=r'$kernel_{ptr_{train_{err}}}$')
plt.plot(kernel_ptr_t[cut:], kernel_ptr_test_err[cut:], 'b-', label=r'kernel_{ptr_{test_{err}}}$')
plt.legend(loc='best')
# plt.yscale('log')
plt.figure(11)
plt.plot(kernel_ptr_t[cut:], kernel_ptr_train_loss[cut:], 'r-', label=r'$kernel_{ptr_{train_{loss}}}$')
plt.plot(kernel_ptr_t[cut:], kernel_ptr_test_loss[cut:], 'b-', label=r'$kernel_{ptr_{test_{loss}}}$')
plt.legend(loc='best')
# plt.yscale('log')


plt.show()