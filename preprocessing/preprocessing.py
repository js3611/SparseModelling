import numpy as np
import numpy.linalg as linalg

def centering(X):
    '''
    Removes the mean intensity of the image from each image
    :param X: [m,n], image in m dimension, n samples
    :return: np.dot(np.identity(m) - np.ones([m,m]) / float(m) , X)
    '''
    # X: [m,n]
    m, n = X.shape
    # mathematically
    # return
    # prob faster:
    return X - np.mean(X, axis=0)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def contrast_normalization(X, c=0.2):
    # X: [m,n]
    return X / np.maximum(c, linalg.norm(X, axis=0))
    
def whitening(X, eps=10e-6):
    # X = [m,n]
    # returns transformation for mapping
    X = X - np.array([np.mean(X, axis=1)]).T
    covX = np.dot(X,X.T) / len(X)
    
    w, v = linalg.eig(covX)
    w = np.real(w)
    idx = w > eps
    w = np.sqrt(w[idx])
    v = np.real(v[:,idx])

    ## sort by decreasing order of eigen vals
    #     idx = w.argsort()[::-1]
    #     w = w[idx]
    #     v = v[:,idx]
    w_inv = 1 / w 

    transform = np.dot(np.dot(v, np.diag(w_inv)), v.T)
    
    return transform

def extract_patches(img, patch_shape):
    patches = []    
    for i in xrange(img.shape[0]-patch_shape[0]+1):
        for j in xrange(img.shape[1]-patch_shape[1]+1):
            patch = img[i:i+patch_shape[0], j:j+patch_shape[1]]
            patches.append(patch)
    
    return patches

def assemble_patches(patches, img_shape):
    if len(patches) == 0:
        return
    p0, p1 = patches[0].shape    
    k = 0        
    img = np.zeros(img_shape)
    overlap_ctr = np.zeros(img_shape)
    for i in xrange(img_shape[0]-p0+1):
        for j in xrange(img_shape[1]-p1+1):
            img[i:i+p0, j:j+p1] += patches[k]
            overlap_ctr[i:i+p0, j:j+p1] += 1
            k += 1
    return img / overlap_ctr

def visualise_patches(patches, tile_shape, padding=True):
    '''
    Input:
        patches: N element of [p0, p1] patches
        tile_shape: (t0,t1): t0 rows t1 cols
    '''
    # get dimension of the patch
    p0, p1 = patches[0].shape
    t0, t1 = tile_shape
    #print patches
    # pad the array
    n = len(patches)
    v_padding = t0 if padding else 0
    h_padding = t1 if padding else 0
    vismat = np.zeros((p0 * t0 + v_padding, p1 * t1 + h_padding))
    for i in xrange(len(patches)):
        # find place to insert
        cpad = (i % t1) if padding else 0
        rpad = (i / t1) if padding else 0
        cidx = (i % t1) * p0
        ridx = (i / t1) * p1
#        print i, cidx, ridx
        if i >= t0 * t1:
            break            
        vismat[ridx+rpad:ridx+p0+rpad,cidx+cpad:cidx+p1+cpad] = patches[i]
    return vismat
