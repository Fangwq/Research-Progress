from grid import load_grouped, load
import matplotlib.pyplot as plt

#runs = load('./interface')
#print(runs)

# args, groups = load_grouped('./ODElimit1000/', ['dt'])
args, groups = load_grouped('./ODElimit1000T20/', ['dt'])
# args, groups = load_grouped('./ODElimit1000T2/', ['dt'])
print(groups[0])
print(args)
cut = int(2.5/0.0001)
t = groups[0][1][0]['t'][0][:cut]
w1 = groups[0][1][0]['w'][0][0][:cut]
wp = groups[0][1][0]['w'][0][1][:cut]
b = groups[0][1][0]['b'][0][:cut]
# beta = groups[0][1][0]['beta'][0]
print(len(t), len(w1), len(wp), len(b))

plt.figure()
plt.plot(t, b, 'b-', label = r'$b$')
plt.plot(t, w1, 'k-', label = r'$w_1$')
plt.plot(t, wp, 'r-', label=r'$w_p$')
# plt.yscale('log')
plt.legend(loc='best')
plt.show()
