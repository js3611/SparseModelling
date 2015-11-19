
# coding: utf-8

# # Optimisation for Dictionary Learning

# Objective: Learn dictionary D such that (1)
# 
# $$\min_{D \in \mathcal{C}, A \in \mathbf{R}^{p x n}} \frac{1}{n}\sum_{i=1}^{n}\frac{1}{2} ||x_i - D \alpha_i||_2^2 + \lambda \psi (\alpha)$$
# 
# or (2)
# 
# $$\min_{D \in \mathcal{C}, A \in \mathbf{R}^{p x n}} \frac{1}{n}\sum_{i=1}^{n}\frac{1}{2} ||x_i - D \alpha_i||_2^2  \quad s.t. \quad \psi (\alpha) \leq \mu$$

# In[20]:

import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from l1_optimisations import coordinate_descent, soft_threshold
from l0_optimisations import orthogonal_matching_pursuit as OMP, iterative_hard_thresholding, matching_pursuit as MP

# get_ipython().magic(u'matplotlib inline')


# In[11]:

def convex_projection(D):
    return D / np.maximum(linalg.norm(D,axis=0), 1)


# ### Stochastic Gradient Descent
# 
# Alternates between 
# 
# 1) Gradient Step 
# 
# 2) Rescaling columns of $D$ (so $D \in \mathcal{C}$, set of matrices with $l_2$ norm of column is less than 1)
# 
# Penalisation funciton $\psi$ is either $l_1$-norm or smooth approximate sparsity-inducing penalty (i.e. $\psi(\alpha) = \sum_{j=1}^p \log(\alpha[j]^2 + \epsilon)$)
# 
# Problem: how to choose the hyper parameters?
# 
# (1) step size $\eta_t  = \eta / (t + t_0)^\gamma$ 
# 
# (2) number of iteration $T$
# 
# (3) Penalty Weight $\lambda$
# 
# (4) Batch Update 
# 
# (5) Whether to reuse A for initialisation

# In[6]:

def stochastic_gradient_descent(X, D0=None,lmbda=None,T=100,penalty='l1',lr=None, t0=None, gamma=None):
    '''
        Parameters:
            X  - Observations, n x m matrix, where column vector is each data instance
            D0 - Initial dictionary. It should be normalised by column
                
        Hyper-parameters:
            Step size at t is calculated as: lr_t = lr / (t + t0)^gamma
            
            lr    - initial learning rate (ish)
            t0    - controls the proportion of decay rate
            gamma - controls the decay rate polynomially
            
        Returns:
            D - dictionary
    '''    
    # Initialisation
    m, n = X.shape       
    lr = lr if lr else 0.1  # Learning rate / step size
    t0 = t0 if t0 else 1
    gamma = gamma if gamma else 1
    
    if not D0:        
        D = random.rand(m,m) 
        D = D / linalg.norm(D, axis=0)      # normalise by column
    else:
        D = D0

    if penalty == 'l1':
        encode = coordinate_descent
    else:
        # TODO: other penalty functions
        encode = coordinate_descent

    if not lmbda:
        lmbda = 0.1 # not sure how to pick a value here

    A = np.zeros(p, n)
        
    # Main Loop
    for t in xrange(T):
        # compute sparse code for x_i picked at random
        i = random.randint(0, n-1)
        x = X[:,i]
        alpha, _ = encode(D, x, lmbda)#, A[:, i])  # reuse old value of alpha to initialise?
        #A[:, i] = alpha
        
        # perform one projected gradient descent
        lr_t = lr / ((t0 + t) ** gamma)
        unprojected_D = D - lr_t * (np.dot(np.dot(D,alpha) - x, alpha.T))
        D = convex_projection(unprojected_D)
        
    return D
    


# In[16]:

# get_ipython().magic(u'pinfo OMP')


# ### Alternate Minimization

# #### Method of Optimal Directions (MOD): $l_0$-penalty
# 
# Uses OMP to compute each value of alpha in the loop

# In[13]:

def method_of_optimal_directions(X, D0=None, k_max=None, T=100):
    '''
    Parameters:
        X     - Observations, n x m matrix, where column vector is each data instance
        D0    - Initial dictionary. It should be normalised by column
        k_max - l0-sparsity constraint
        T     - number of iteration

    Returns:
        D - dictionary
        
    '''
    # Initialisation
    m, n = X.shape       
    
    if not D0:
        p = m
        D = random.rand(m, p) 
        D = D / linalg.norm(D, axis=0)      # normalise by column
    else:
        D = D0
        _, p = D.shape

    if not lmbda:
        lmbda = 0.1 # not sure how to pick a value here

    if not k_max:
        k_max = m / 10
        
    # main loop
    for t in xrange(T):
        # compute all alphas
        A = np.zeros(p, n)
        for i in xrange(n):
            alpha, _ = OMP(D, X[:,i], k_max)
            A[:,i] = alpha
        unprojected_D = np.dot(X, np.dot(A.T, linalg.inv(np.dot(A, A.T))))
        D = convex_projection(unprojected_D)
    
    return D
    


# #### Alternate Minimization: $l_1$-penalty
# 
# Ideally requires homotopy but can use coordinate descent for now

# In[25]:

def newton_update(D, X, A, T):
    ## TODO ##
    return D

def bcd_update(D, X, A, T):
    '''
    Parameters:
        T - number of iterations
    Returns:
        D - updated dictionary
    
    '''
    ## TODO ##
    return D


# In[23]:

def alternate_minimisation(X, D0=None, lmbda=None, mu=None, T=100, dict_update='Newton'):
    '''
    Parameters:
        X     - Observations, n x m matrix, where column vector is each data instance
        D0    - Initial dictionary. It should be normalised by column
        lmbda - l1-sparsity constraint for Lasso formulation
        mu    - hard constraint l1-sparsity
        T     - number of iteration
        dict_update - 'Newton' or 'Block' (Block Coordinate Descent)

    Returns:
        D - dictionary
        
    '''
    # Initialisation
    m, n = X.shape       
    
    if not D0:
        p = m
        D = random.rand(m, p) 
        D = D / linalg.norm(D, axis=0)      # normalise by column
    else:
        D = D0
        _, p = D.shape

    if not lmbda:
        lmbda = 0.1 # not sure how to pick a value here
        
    if dict_update == 'Newton':
        update_D = newton_update
    else:
        update_D = bcd_update
        
    # Main Loop    
    for t in xrange(T):
        A = np.zeros(p, n)
        for i in xrange(n):
            alpha, _ = coordinate_descent(D,X[:,i],lmbda)
            A[:,i] = alpha
        
        # Update D by minimising objective function for D with fixed X and A.
        D = update_D(D, X, A)
        
    return D


# ### Block Coordinate Descent

# In[27]:

def block_coordinate_descent(X, D0=None, A0=None, lmbda=None, T=100):
    '''
    Parameters:
        X     - Observations, n x m matrix, where column vector is each data instance
        D0    - Initial dictionary. It should be normalised by column
        A0    - sparse coefficients
        lmbda - l1-sparsity constraint for Lasso formulation
        mu    - hard constraint l1-sparsity
        T     - number of iteration

    Returns:
        D - dictionary
        
    '''
    # Initialisation
    m, n = X.shape       
    
    if not D0:
        p = m
        D = random.rand(m, p) 
        D = D / linalg.norm(D, axis=0)      # normalise by column
    else:
        D = D0
        _, p = D.shape
        
    if not A0:
        A = np.zeros(p, n)
    else:
        A = A0

    if not lmbda:
        lmbda = 0.1 # not sure how to pick a value here
        
    if dict_update == 'Newton':
        update_D = newton_update
    else:
        update_D = bcd_update
        
    # Main Loop    
    for t in xrange(T):
        
        # update each row of A
        for j in xrange(p):            
            A_row_j = A[j,:] + np.dot(D[:,j], X - np.dot(D,A)) / linalg.norm(D[:,j])
            A[j,:] = soft_threshold(A_row_j)
                    
        # Update D using block coordinate descent with iteration 1 ~ 5
        
        D = bcd_update(D, X, A, T=5)
        
    return D


# ### K-SVD
# 
# Solves for $l_0$-regularised dictionary learning

# In[40]:

def k_svd(X, D0=np.array([]), mu=None, T=100):
    '''
    Uses SVD to approximate each dictionary element
    Parameters:
        X   - data
        D0  - initial dictionary
        mu  - target sparsity
        T   - number of iterations
    Returns:
        D - updated dictionary        
    '''
    m, n = X.shape
    eps = 1e-6
    
    if D0.size == 0:
        p = m
        D = random.rand(m, p) 
        D = D / linalg.norm(D, axis=0)      # normalise by column
    else:
        _, p = D0.shape
        D = D0

    mu = mu if mu else p/10
        
    A = np.zeros([p, n])
    
    # Main loop
    for t in xrange(T):
        
        # Compute A
        for i in xrange(n):
            A[:,i] = OMP(D, X[:,i], mu)[0]
                    
        # Compute D and A (nonzero coeffs of A)
        for j in xrange(p):
            # define set of indices of X which uses D[:,j] for sparse encoding
            indices = [i for i, v in enumerate(A[j,:]) if abs(v) < eps]
            
            S = np.zeros([n, len(indices)])
            for i, idx in enumerate(indices):
                S[idx, i] = 1
            
            # error matrix
            E = X - np.dot(D, A) + np.outer(D[:,j],A[j,:]) 
            
            # Sparsify
            E_sparse = np.dot(E, S) 
            
            # compute the svd to get d and a
            u, s, v = linalg.svd(E_sparse, compute_uv=True)

            # update
            D[:,j] = u[:,0] 
            A[j,indices] = s[0] * v[0,:]
                    
    return D


# ### Stochastic Approximation

# In[ ]:



