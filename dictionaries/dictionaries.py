__author__ = 'js3611'

import numpy as np


def a(x, N):
    return np.sqrt(2.0/N) if x > 0 else np.sqrt(1.0/N)


def get_dct_dictionary(N=16):
    '''
    Gets dct dictionary elemetns
    :param N:
    :return: dct: [N, N*N] dictionary containing dct elements
    '''
    dct = []
    x = np.arange(N)
    for u in xrange(N):
        y1 = a(u, N) * np.cos((2*x+1)*u*np.pi / (2*N))
        for v in xrange(N):
            y2 = a(v, N) * np.cos((2*x+1)*v*np.pi / (2*N))
            dct.append(np.outer(y1, y2))
    return np.array(dct).reshape(N*N, (N*N))


def get_dct_transform(N=16):
    '''
    Gets unitary dct matrix
    :param N:
    :return:
    '''
    T = np.zeros((N,N))
    for u in xrange(N):
        for x in xrange(N):
            T[u,x] = a(u,N) * np.cos((2*x+1)*u*np.pi/(2*N))

    return T


def dct(X):
    '''
    Performs discrete cosine transform on image X
    :param X: Image
    :return: F: cosine transform
    '''
    N = X.shape[0]
    T = get_dct_transform(N)
    return np.dot(np.dot(T, X), T.T)

get_dct_dictionary(16)