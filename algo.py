import numpy as np
import scipy.io
from operator import itemgetter
from random import *
import matplotlib.pyplot as plt


def spectral(w_f, k):
    # print(w_f.shape)
    d = np.zeros(w_f.shape)
    for i in range(w_f.shape[0]):
        for j in range(w_f.shape[1]):
            d[i, i] += w_f[i, j]
    l = d - w_f
    w, v = np.linalg.eig(l)
    # print(w.shape)
    # print(sorted_w)
    ind = w.argsort()[:k]
    V = np.zeros((v.shape[0], k))
    for a in range(len(ind)):
        V[:, a] = v[:, ind[a]].T
    # print(V)
    return V


def graph(x, sig, k):
    w = np.zeros((x.shape[1], x.shape[1]))
    for i in range(w.shape[1]):
        temp = {}
        for j in range(w.shape[1]):
            if i == j:
                continue
            temp[j] = np.linalg.norm(x[:, i] - x[:, j], 2)
        sorted_x = sorted(temp.items(), key=itemgetter(1))
        # print(sorted_x[0], sorted_x[1])
        for t in range(k):
            weight = np.exp(-sorted_x[t][1] / 2 * pow(sig, 2))
            w[i, sorted_x[t][0]] = weight
            w[sorted_x[t][0], i] = weight
    # temp[j] = np.exp(-np.linalg.norm(y[:, i] - y[:, j], 2) / 2 * pow(sig, 2))
    return w


def kmeans(x_f, k_f, r_f):
    c = np.zeros((x_f.shape[0], k_f))
    total_loss = float('Inf')
    cluster_indices = []
    for q in range(r_f):
        for j in range(k_f):
            c[:, j] = x_f[:, randint(0, x_f.shape[1])-1].T
        for t in range(500):
            if t % 2 == 0:
                z = np.zeros((x_f.shape[1], k_f))
                for i in range(x_f.shape[1]):
                    min_d = float('Inf')
                    min_ind = 0
                    for l in range(k_f):
                        temp = np.linalg.norm(x_f[:, i].T - c[:, l].T)
                        if temp < min_d:
                            min_d = temp
                            min_ind = l
                    z[i, min_ind] = 1
                    # print(q, i, min_ind)
            else:
                c = np.zeros((x_f.shape[0], k_f))
                for l in range(k_f):
                    n = 0
                    for j in range(x_f.shape[1]):
                        if z[j, l] == 1:
                            temp = c[:, l] + x_f[:, j].T
                            c[:, l] = temp
                            # print(c.shape)
                            n += 1
                    if n != 0:
                        c[:, l] /= n
        total_loss_q = 0
        cluster_indices_q = []
        for b in range(k_f):
            cluster_indices_q.append([0])
        # print(cluster_indices_q)
        for l in range(k_f):
            for j in range(x_f.shape[1]):
                if z[j, l] == 1:
                    total_loss_q += np.linalg.norm(c[:, l].T - x_f[:, j].T)
                    cluster_indices_q[l].append(j)
        if total_loss_q <= total_loss:
            total_loss = total_loss_q
            cluster_indices = cluster_indices_q
        print(total_loss, total_loss_q)
    cluster_indices[0].pop(0)
    cluster_indices[1].pop(0)
    return cluster_indices

def pca(y):
    mean = np.zeros((y.shape[0], 1))
    for i in range(y.shape[1]):
        mean += y[:, i]/y.size
    centred_y = np.zeros(y.shape)
    for i in range(y.shape[1]):
        temp = y[:, i] - mean
        centred_y[:, i] = temp.T
    u, s, v = np.linalg.svd(centred_y)
    print(u.shape)
    plt.plot(s)
    plt.show()
    x = np.zeros((2, y.shape[1]))
    for i in range(x.shape[1]):
        x[:, i] = np.matmul(u[:, :2].T, (centred_y[:, i]))
    return mean, u, x


data = scipy.io.loadmat('dataset2.mat')
Y = np.asmatrix(data['Y'])
m, u, x = pca(Y)
print(x.shape)
plt.scatter(x[0, :], x[1, :])
plt.show()
w = graph(Y, 1, 10)
# print(w)
v = spectral(w, 2)
v = v.T
# print(v)
c_ind = kmeans(v, 2, 5)
# print(c_ind)
print(v)
for a1 in c_ind[0]:
    plt.scatter(v[0, a1], v[1, a1], c='blue')
for a1 in c_ind[1]:
    plt.scatter(v[0, a1], v[1, a1], c='red')
plt.show()
