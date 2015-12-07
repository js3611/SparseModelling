# coding: utf-8
import numpy as np
import numpy.linalg as la
import scipy as sp
import heapq



def sqnorm(x):
    return np.dot(x, x)


# ## Sparse Reconstruction with $l_0$-penalty:
# $$\min_{\alpha \in \mathbf{R}^P} ||x - D \alpha||_2^2 \quad s.t. ||\alpha||_0 \leq k$$
# ### Coordinate Descent Algorithms
# #### Matching Pursuit
def matching_pursuit(D, x, k_max=10, eps=1e-10, max_itr=1000):
    itr = 0
    m, p = D.shape
    alpha = np.zeros(p)
    k = 0
    obj_values = []
    while k < k_max and itr < max_itr:
        d_prod = np.dot(D.T, x - np.dot(D,alpha))
        j = np.argmax(np.abs(d_prod))
        if alpha[j] == 0:
            k += 1
        
        alpha[j] += d_prod[j]
        itr += 1
        diff = sqnorm(x - np.dot(D, alpha))
        # print diff
        obj_values.append(diff)
    
    return (alpha, obj_values)


# #### Orthogonal Matching Pursuit (OMP)
def orthogonal_matching_pursuit(D, x, n_sparsity=None, verbose=False):
    '''
    DEPRECATED
    Performs orthogonal matching pursuit.
    The algorithm stops either when:
        1) target sparsity is reached,
        2) 15% of sparsity is reached or
        3) reconstruction error is lower than 1e-4

    :param D: Dictionary D
    :param x: original signal
    :param n_sparsity:
    :return: alpha, sparse representation
    '''
    m, p = D.shape
    if not n_sparsity:
        # At least 5 components will be used, unless p is less than 5
        n_sparsity = min(max(int(p * 0.15), 5), p)
        if verbose:
            'sparsity set to %d' % n_sparsity

    k = 0
    alpha = np.zeros(p)
    active_set = []
    obj_values = []
    error = sqnorm(x)
    while error > 10e-8 and k < n_sparsity:
        min_j = 0     # Index of new dictionary entry with lowest recon_error
        min_beta = 0  # Corresponding sparse representation
        for j in set(xrange(p)).difference(active_set):

            gamma = active_set + [j]
            D_gamma = D[:, gamma]
            # Compute: (D_gamma.T * D_gamma)^-1 * D_gamma.T * x
            beta = np.dot(la.inv(np.dot(D_gamma.T, D_gamma)), np.dot(D_gamma.T, x))
            x_projected = np.dot(D_gamma, beta)
            curr_delta = sqnorm(x - x_projected)

            if curr_delta <= error:
                error = curr_delta
                min_j = j
                min_beta = beta

        # update active set and the solution alpha
        active_set += [min_j]
        alpha[active_set] = min_beta
        complement_set = [idx for idx in xrange(p) if idx not in active_set ]
        alpha[complement_set] = 0
        # diff = sqnorm(x - np.dot(D,alpha)) # this should be the same as error
        # print diff 
        # obj_values.append(diff)
        obj_values.append(error)
        k += 1

    return alpha, obj_values


def OMP_residue(D, x, n_sparsity=None, verbose=False):
    '''
    Performs orthogonal matching pursuit.
    The algorithm stops either when:
        1) target sparsity is reached,
        2) 15% of sparsity is reached or
        3) reconstruction error is lower than 1e-4

    :param D: Dictionary D
    :param x: original signal
    :param n_sparsity:
    :return: alpha, sparse representation
    '''
    m, p = D.shape
    if not n_sparsity:
        # At least 5 components will be used, unless p is less than 5
        n_sparsity = min(max(int(p * 0.15), 5), p)
        if verbose:
            'sparsity set to %d' % n_sparsity

    k = 0
    obj_values = []
    r = x
    error = sqnorm(r)
    D_orthogonal = np.zeros((m, n_sparsity))
    while error > 10e-8 and k < n_sparsity:
        # Find dictionary element with highest correlation to the residue
        residue_projection = np.abs(np.dot(D.T, r))
        j = np.argmax(residue_projection)
        D_orthogonal[:, k] = D[:, j]
        D = np.delete(D, j, 1)
        D_I = D_orthogonal[:, 0:(k+1)]
        # compute sparse code via orthogonal projection
        alpha = np.dot(la.inv(np.dot(D_I.T, D_I)), np.dot(D_I.T, x))
        r = x - np.dot(D_I, alpha)
        error = sqnorm(r)
        obj_values.append(error)
        k += 1

    return alpha, obj_values

def OMP_Cholesky(D, x, n_sparsity=None, verbose=False):
    m, p = D.shape
    if not n_sparsity:
        # At least 5 components will be used, unless p is less than 5
        n_sparsity = min(max(int(p * 0.15), 5), p)
        if verbose:
            'sparsity set to %d' % n_sparsity

    # Initialise
    k = 0
    n = 1
    obj_values = []
    I = []
    r = x
    D_orthogonal = np.zeros((m, n_sparsity))
    L_full = np.zeros((n_sparsity, n_sparsity))
    L_full[0, 0] = 1
    l = 1
    alpha = np.dot(D.T, x)
    # alpha

    # helper
    orig_index = np.arange(0, p)

    error = sqnorm(r)
    while error > 10e-8 and k < n_sparsity:

        L = L_full[0:l, 0:l]
        # Find dictionary element with highest correlation to the residue
        residue_projection = np.abs(np.dot(D.T, r))
        j = np.argmax(residue_projection)
        d = D[:, j]

        if n > 1:
            # print D.T.shape, d.shape, L.shape
            w = sp.linalg.solve_triangular(L, np.dot(D_I.T, d))
            L_full[l, 0:l] = w.T
            L_full[l, l] = np.sqrt(1-np.dot(w, w))
            l += 1
            L = L_full[0:l, 0:l]

        # update dictionary elements
        D_orthogonal[:, k] = d
        D = np.delete(D, j, 1)

        # update I
        I.append(orig_index[j])
        orig_index[j:] = np.concatenate((orig_index[j+1:], [0]))  # update the orig index
        D_I = D_orthogonal[:, 0:(k+1)]
        alpha_I = alpha[I]

        # solve for gamma using cholesky
        # print L.shape, alpha_I.shape
        x_ = sp.linalg.solve_triangular(L, alpha_I)
        gamma_I = sp.linalg.solve_triangular(L, x_, trans=1)

        # update the residue
        r = x - np.dot(D_I, gamma_I)
        error = sqnorm(r)
        obj_values.append(error)
        k += 1
        n += 1

    alpha = np.zeros_like(alpha)
    alpha[I] = alpha_I
    return alpha, obj_values


def nth_largest(n, itr):
    return heapq.nlargest(n, itr)[-1]

def hard_thresholding(elt, threshold):
    return elt if np.abs(elt) >= threshold else 0

def iterative_hard_thresholding(D, x, k_max=None, lmbda=None, mu=0.01, alpha=None, T=100):
    '''
    Parameters:
        x     - input vector
        D     - dictionary
        k_max - target sparsity
        lmbda - sparsity penalty
        mu    - step size for each gradient descent
        alpha - initial alpha
        T     - # of iteration
    '''
    m, p    = D.shape
    if not lmbda:
        lmbda = 0.1
    tau = np.sqrt(2 * lmbda)
        
    if not alpha:
        alpha   = np.zeros(p)
    obj_values = []
    for i in xrange(T):
        # perform one step gradient
        alpha = alpha + mu * np.dot(D.T, x - np.dot(D,alpha))
        
        if k_max:
            # pick k largest elements
            tau = nth_largest(k_max, alpha)

            
#         print alpha
        
        alpha = map(hard_thresholding, alpha, [tau]*p)
            
        obj_values.append(sqnorm(x - np.dot(D,alpha)))
            
    return alpha, obj_values
