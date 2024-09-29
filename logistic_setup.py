import numpy as np
from scipy.special import expit
import scipy as sp

'''l1-l2 regulartized logistic regression'''
def f_logistic(w, X, y, mu=0): 
    '''logistic loss with \mu l_2 regularization
        w: parameter (1-dim array)
        X: feature matrix (2-dim array)
        y: labels (1-dim array)
        mu: l_2 regularization parameter
    '''
    n = X.shape[0]
    z = X.dot(w)
    sigmoid = expit(z)
    loss = -y * np.log(sigmoid) - (1 - y) * np.log1p(-sigmoid)
    result = np.sum(loss) / n + mu / 2 * np.linalg.norm(w, ord=2)**2
    return result

def h(w, lam=1e-3):  # L1-regularization
    return lam * np.linalg.norm(w, ord=1)

def F_logistic(w, X, y, mu=0, lam=1e-3):  # total loss
    n = X.shape[0]
    z = X.dot(w)
    sigmoid = expit(z)
    loss = -y * np.log(sigmoid) - (1 - y) * np.log1p(-sigmoid)
    total_loss = np.sum(loss) / n + mu / 2 * np.linalg.norm(w, ord=2)**2 + lam * np.linalg.norm(w, ord=1)
    return total_loss

def f_logistic_grad(w, X, y, mu=0): 
    '''batch gradient of smooth part f
        X: batch feature matrix (2-dim array)
    '''
    n = X.shape[0]
    z = X.dot(w)
    c = expit(z)

    # Calculate the gradient
    error = c - y
    grad = X.T.dot(error) / n + mu * w
 
    return grad

def f_logistic_hess_z(w, X, z, mu=0):
    '''batch hessian vector product
        X: batch feature matrix (2-dim array)
        z: vector to be multiplied (1-dim array)
    '''
    n = X.shape[0]
    v = X.dot(w)
    c = expit(v)
    sigmoid_derivative = c * (1 - c)
    X_dot_z = X.dot(z)
    if isinstance(X, np.ndarray):
        hess_product = sigmoid_derivative[:, np.newaxis] * X * X_dot_z[:, np.newaxis]
        hess = hess_product.sum(axis=0)/n + mu * z
    else: # csr_sparse matrix
        hess_product = X.multiply(X_dot_z[:, np.newaxis]).multiply(sigmoid_derivative[:, np.newaxis]) # return a coo_matrix
        hess = hess_product.sum(axis=0)/n # return an array
        hess = np.asarray(hess).reshape(-1) + mu * z
    return hess

def compute_L_mu(w, X):
    '''compute max and min eigenvalues of batch hessian'''
    n, d = X.shape
    c = expit(sp.dot(X, w))
    sigmoid_derivative = c * (1 - c)
    hess = np.einsum('ni,nj->ij', X * sigmoid_derivative[:, np.newaxis], X)
    hess = hess / n
    eigenvalues = np.linalg.eigvalsh(hess)
    max_eigenvalue = np.max(eigenvalues)
    min_eigenvalue = np.min(eigenvalues)
    return max_eigenvalue, min_eigenvalue


def prox_h(w, lam=0.01):  # proximal operator of lam * L1_norm
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0)
