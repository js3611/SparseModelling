import pickle

import matplotlib.pyplot as plt

from SparseModelling.dictionaries.dictionary_learning import k_svd
from SparseModelling.optimisation.l0_optimisations import orthogonal_matching_pursuit as OMP
from SparseModelling.preprocessing.preprocessing import *
from dictionaries import get_dct_dictionary

'''
Run dictionary on the entire image
'''

# Load MNIST DATA

# data = read_mnist()
# train, test = data
# train_x, train_y = train
# test_x, test_y = test

# data = read_mnist_data('test')
# test_x = data[0:10]
# plt.imshow(train_x[0])

# Load Small subset of MNIST data

pkl_file = open('sample_data.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

# Preprocessing the data

# normalise by column & demean
X = centering(data)
X = contrast_normalization(X)
m, n = X.shape

# Initialise the dictionary
M = int(np.sqrt(m))
dct_dictionary = get_dct_dictionary(M).T

dict_elts = np.arange(0, m, 5)
p = len(dict_elts)
# p = m
D = dct_dictionary[:, dict_elts]
# p = 50
# D = random.normal(size=[m, p])
D /= linalg.norm(D, axis=0)      # normalise by column

print D.shape

plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
vis = visualise_patches(D.T.reshape(p, M, M), (p/15,15), True)
plt.imshow(vis, cmap=plt.get_cmap('gray'))

# Perform KSVD
D = k_svd(X, D, mu=10, T=10)

vis = visualise_patches(D.T.reshape(p, M, M), (p/15,15), True)
plt.subplot(2,1,2)
plt.imshow(vis, cmap=plt.get_cmap('gray'))

x = X[:, 1]
alpha, err = OMP(D, x, k_max=10)

x_recon = np.dot(D, alpha).reshape(28, 28)

print err[-1]
plt.figure(0)
plt.subplot(1, 3, 1)
plt.imshow(x.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.subplot(1, 3, 2)
plt.bar(np.arange(p), alpha)
plt.subplot(1, 3, 3)
plt.imshow(x_recon, cmap=plt.get_cmap('gray'))

plt.show()



#Visualise the learnt dictionary

