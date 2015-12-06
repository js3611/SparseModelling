import matplotlib.pyplot as plt
from dictionary_learning import k_svd, stochastic_gradient_descent
from l0_optimisations import orthogonal_matching_pursuit as OMP
from preprocessing import *
from dictionaries import get_dct_dictionary
import pickle

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
# extract patches
m, n = data.shape
patch_size = 8
patches = []
for i in xrange(n):
    patches += extract_patches(data[:, i].reshape(28, 28), patch_shape=(patch_size, patch_size))

n_patch = len(patches)

new_data = np.array(patches).reshape(n_patch, patch_size*patch_size).T
# # normalise by column & demean
X = new_data
# X = centering(new_data)
X = contrast_normalization(X)

# new dimension of data
m, n = X.shape
print "Data dimension {} and {} samples".format(m, n)

# Initialise the dictionary
M = int(np.sqrt(m))
dct_dictionary = get_dct_dictionary(M).T # [m, p]
D = dct_dictionary
m, p = D.shape

print "Dictionary shape {}".format(D.shape)

plt.figure(figsize=(10, 10))
plt.subplot(2, 3, 1)
vis = visualise_patches(D.T.reshape(p, M, M) - np.min(D), (patch_size, patch_size), True)
plt.imshow(vis, cmap=plt.get_cmap('gray'))
#
#### Perform KSVD #######
train_idx = np.random.choice(X.shape[1], 10000)
D = k_svd(X[:, train_idx], D, mu=10, T=5)
#
vis = visualise_patches(D.T.reshape(p, M, M) - np.min(D), (patch_size, patch_size), True)
plt.subplot(2, 3, 4)
plt.imshow(vis, cmap=plt.get_cmap('gray'))


# use patch based reconstruction
x = data[:, 88] / 255.0
# x = x - np.mean(x)
x_noised = x.copy()
for i in xrange(28*28):
    if np.random.random() < 0.01:
        x_noised[i] = 0
    elif np.random.random() > 0.99:
        x_noised[i] = 1
# x_noised = x + np.random.normal(0, 0.01, 28*28)
# x_patches = extract_patches(x.reshape(28, 28), patch_shape=(patch_size, patch_size))
x_patches = extract_patches(x_noised.reshape(28, 28), patch_shape=(patch_size, patch_size))
patch_n = len(x_patches)
x_patches = np.array(x_patches).reshape(patch_n, patch_size*patch_size).T
x_patch_means = np.mean(x_patches, axis=0)
x_patches = centering(x_patches)
x_patches = contrast_normalization(x_patches)
patch_recons = []

print "Number of patches to assemble: {}".format(x_patches.shape[1])
for i in xrange(patch_n):
    alpha, _ = OMP(D, x_patches[:, i], k_max=20)
    patch_recons.append(np.dot(D, alpha).reshape(patch_size, patch_size))# + x_patch_means[i])

x_recon = assemble_patches(patch_recons, (28, 28))
#x_recon -= np.min(x_recon) # normalise the minimum
plt.subplot(2, 3, 2)
plt.title("Original")
plt.imshow(x.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.subplot(2, 3, 3)
plt.title("Noised")
plt.imshow(x_noised.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.subplot(2, 3, 5)
plt.title("Reconstruction")
plt.imshow(x_recon, cmap=plt.get_cmap('gray'))
plt.subplot(2, 3, 6)
plt.title("Difference")
plt.imshow(x.reshape(28, 28) - x_recon, cmap=plt.get_cmap('gray'),vmin=0,vmax=1)

mse = ((x.reshape(28, 28) - x_recon) ** 2).mean()
print "Reconstruction error {}".format(mse)

plt.show()



#Visualise the learnt dictionary

