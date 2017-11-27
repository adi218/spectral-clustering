import numpy as np
import scipy.io
from operator import itemgetter


def spectral(w_f, k):
    d = np.zeros(w_f.shape)
    for i in range(w_f.shape[0]):
        for j in range(w_f.shape[1]):
            d[i, i] += w_f[i, j]
    l = d - w_f
    print(np.array_equal(w_f, w_f.T))
    w, v = np.linalg.eig(l)
    ind = w.argsort()[:k]
    print(ind)
    V = np.zeros((v.shape[0], k))
    for a in range(len(ind)):
        V[:, a] = v[:, ind[a]]
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
            weight = np.exp(-(sorted_x[t][1]**2) / (2 * pow(sig, 2)))
            w[i, sorted_x[t][0]] = weight
            w[sorted_x[t][0], i] = weight
    return w


data = scipy.io.loadmat('dataset2.mat')
Y = np.asmatrix(data['Y'])
w = graph(Y, 0.25, 10)
v = spectral(w, 2)
v = v.T
print(v)
