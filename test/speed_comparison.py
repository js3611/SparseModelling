__author__ = 'js3611'

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import SparseModelling.optimisation.l0_optimisations as l0
import sklearn
import time



m = 500              # dimension of each atom
p = 100              # number of atoms
rand = np.random.normal
D = rand(size=[m,p])    # dictionary
x = rand(size=m)
# normalise by column
D = D / np.linalg.norm(D, axis=0)
x = x / np.linalg.norm(x)

start = time.clock()
alpha, obj_values = l0.orthogonal_matching_pursuit(D, x)
end = time.clock()
print "Execution time: %3f" % (end-start)
s = "sparsity: {}".format(len(alpha.nonzero()[0]))
plt.subplot(3, 1, 1)
plt.title(s)
plt.plot(np.array(obj_values))

start = time.clock()
alpha, obj_values = l0.OMP_residue(D, x)
end = time.clock()
print "Execution time: %3f" % (end-start)
s2 = "sparsity: {}".format(len(alpha.nonzero()[0]))
plt.subplot(3, 1, 2)
plt.title(s2)
plt.plot(np.array(obj_values))

start = time.clock()
alpha, obj_values = l0.OMP_Cholesky(D, x)
end = time.clock()
print "Execution time: %3f" % (end-start)
s2 = "sparsity: {}".format(len(alpha.nonzero()[0]))
plt.subplot(3, 1, 2)
plt.title(s2)
plt.plot(np.array(obj_values))

plt.show()

# print(__doc__)
#
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import OrthogonalMatchingPursuit
# from sklearn.linear_model import OrthogonalMatchingPursuitCV
# from sklearn.datasets import make_sparse_coded_signal
#
# n_components, n_features = 512, 100
# n_nonzero_coefs = 17
#
# # generate the data
# ###################
#
# # y = Xw
# # |x|_0 = n_nonzero_coefs
#
# y, X, w = make_sparse_coded_signal(n_samples=1,
#                                    n_components=n_components,
#                                    n_features=n_features,
#                                    n_nonzero_coefs=n_nonzero_coefs,
#                                    random_state=0)
#
# idx, = w.nonzero()
#
# # distort the clean signal
# ##########################
# y_noisy = y + 0.05 * np.random.randn(len(y))
#
# # plot the sparse signal
# ########################
# plt.figure(figsize=(7, 7))
# plt.subplot(4, 1, 1)
# plt.xlim(0, 512)
# plt.title("Sparse signal")
# plt.stem(idx, w[idx])
#
# # plot the noise-free reconstruction
# ####################################
#
# omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
# omp.fit(X, y)
# coef = omp.coef_
# idx_r, = coef.nonzero()
# plt.subplot(4, 1, 2)
# plt.xlim(0, 512)
# plt.title("Recovered signal from noise-free measurements")
# plt.stem(idx_r, coef[idx_r])
#
# # plot the noisy reconstruction
# ###############################
# omp.fit(X, y_noisy)
# coef = omp.coef_
# idx_r, = coef.nonzero()
# plt.subplot(4, 1, 3)
# plt.xlim(0, 512)
# plt.title("Recovered signal from noisy measurements")
# plt.stem(idx_r, coef[idx_r])
#
# # plot the noisy reconstruction with number of non-zeros set by CV
# ##################################################################
# omp_cv = OrthogonalMatchingPursuitCV()
# omp_cv.fit(X, y_noisy)
# coef = omp_cv.coef_
# idx_r, = coef.nonzero()
# plt.subplot(4, 1, 4)
# plt.xlim(0, 512)
# plt.title("Recovered signal from noisy measurements with CV")
# plt.stem(idx_r, coef[idx_r])
#
# plt.subplots_adjust(0.06, 0.04, 0.94, 0.90, 0.20, 0.38)
# plt.suptitle('Sparse signal recovery with Orthogonal Matching Pursuit',
#              fontsize=16)
# plt.show()