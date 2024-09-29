import numpy as np
from time import time
from tqdm import tqdm
from functools import partial
from fista import fista
from logistic_setup import F_logistic, f_logistic_grad, f_logistic_hess_z, prox_h
from scipy.sparse.linalg import norm


def approx_prox_sub_newton(X, y, mu=1e-3, lam=1e-3, w_init=None, lr=1e-3, opts=None, iter_num=1000):
    '''approximate proximal subsampled Newton method for regularized logistic regression
        problem setup: 
            X: feature matrix
            y: labels 
            mu: l_2 regularization parameter
            lam: l_1 regularization parameter
        algorithm setup:
            w_init: initial point
            lr: learning rate
            iter_num: number of iterations
            opts: options for the algorithm
    '''
    # initilization
    opts.setdefault('stopping', '')
    opts.setdefault('threshold', 1e-6)
    opts.setdefault('inner_max', 1000)
    opts.setdefault('grad_size', {'init': 0.01, 'growth': 1.1, 'period': 10})
    opts.setdefault('hess_size', {'init': 0.01, 'growth': 1.1, 'period': 10})
    opts.setdefault('linesearch', False)
    opts.setdefault('alpha_ls', 1e-3)
    opts.setdefault('rho_ls', 0.8)
    opts.setdefault('time', False)
    
    n, d = X.shape
    bg = int(n*opts['grad_size']['init'])
    bh = int(n*opts['hess_size']['init'])

    l_r = lr
    
    if w_init is None:
        w_init = 0.01 * np.ones(d)
    w_old = w_init.copy()
    w_sequence = np.zeros((iter_num, d)) # store x_k
    loss_sequence = np.zeros(iter_num) # store F(x_k)
    inner_iter_nums = np.zeros(iter_num) # store number of inner iterations to update x_k

    flag_dense = isinstance(X, np.ndarray)

    inner_opts = {'stopping': opts['stopping'], 'max_iter': opts['inner_max']}
    time_list = []

    tic=time()

    # main loop
    for k in tqdm(range(iter_num)):
        w_sequence[k] = w_old
        loss_sequence[k] = F_logistic(w_old, X, y, mu, lam)

        # select independent samples O_k and S_k
        if k>0:
            if k%opts['grad_size']['period'] == 0:
                bg = int(min(opts['grad_size']['growth'] * bg, n))
            if k%opts['hess_size']['period'] == 0:
                bh = int(min(opts['hess_size']['growth'] * bh, n))
        p1 = np.random.permutation(n)
        p2 = np.random.permutation(n)
        O = p1[:bg]
        S = p2[:bh]

        # set up for inner FISTA solver
        g_k = f_logistic_grad(w_old, X[O], y[O], mu)

        # use np.linalg.norm() for narray and norm() for sparse matrix (very important!!!)
        if flag_dense:
            L = np.max(np.linalg.norm(X[S], axis=1, ord=2)**2) + mu
        else: # csr_sparse matrix
            L = np.max(norm(X[S], axis=1, ord=2)**2) + mu
        def grad(w): # the gradient of inner FISTA, linear in w
            return g_k + f_logistic_hess_z(w_old, X[S], w-w_old, mu)
        
        # define options for inner FISTA
        if inner_opts['stopping'] == 'criteria':
            inner_opts['threshold'] = opts['threshold'] * mu
        elif inner_opts['stopping'] == 'criteria_lee':
            G_rhs = L * (w_old - prox_h(w_old - g_k / L, lam / L))
            inner_opts['threshold'] = opts['threshold'] * np.linalg.norm(G_rhs)
        else:
            inner_opts['threshold'] = opts['threshold']
        
        w_temp, inner_iter_nums[k] = fista(grad, L, lam, w_old, inner_opts)
        
        if opts['linesearch']:
            if bg < n:
                raise ValueError('Line search can only be used when grad_size = 1.')
            else:
                F_current = F_logistic(w_old, X, y, mu, lam)
                p = w_temp - w_old
                f_grad_current = f_logistic_grad(w_old, X, y, mu)
                Lam = p.dot(f_grad_current) + lam * np.linalg.norm(w_old + p, ord=1) - lam * np.linalg.norm(w_old, ord=1)
                lr_test = 1
                for _ in range(20):
                    w_test = w_old + lr_test * p
                    if F_logistic(w_test, X, y, mu, lam) > F_current + opts['alpha_ls'] * lr_test * Lam:
                        lr_test *= opts['rho_ls']
                    else:
                        break
                l_r = lr_test       
        w_new = w_old + l_r * (w_temp - w_old)
        w_old = w_new.copy()
        
        time_list.append(time()-tic)
    if opts['time']:
        return loss_sequence, w_sequence, inner_iter_nums, time_list
    else:
        return loss_sequence, w_sequence, inner_iter_nums

def prox_newton(X, y, mu=1e-3, lam=1e-3, w_init=None, opts=None, iter_num=1000): 
    '''inexact proximal Newton method proposed by Lee et al., stopping rule given in (2.24) of the paper
        problem setup: 
            X: feature matrix
            y: labels 
            mu: l_2 regularization parameter
            lam: l_1 regularization parameter
        algorithm setup:
            w_init: initial point
            lr: learning rate
            iter_num: number of iterations
            opts: options for the algorithm including keys 'inner_max' and 'threshold' (\mu/2)
    '''
    # initilization
    opts.setdefault('threshold', mu/2)
    opts.setdefault('inner_max', 1000)
    opts.setdefault('time', False)

    n, d = X.shape
    if w_init is None:
        w_init = 0.01 * np.ones(d)
    w_old = w_init.copy()
    w_sequence = np.zeros((iter_num, d)) # store x_k
    loss_sequence = np.zeros(iter_num) # store F(x_k)
    inner_iter_nums = np.zeros(iter_num) # store number of inner iterations to update x_k
    flag_dense = isinstance(X, np.ndarray)

    # use np.linalg.norm() for narray and norm() for sparse matrix (very important!!!)
    if flag_dense:
        L = np.max(np.linalg.norm(X, axis=1, ord=2)**2) + mu
    else: # csr_sparse matrix
        L = np.max(norm(X, axis=1, ord=2)**2) + mu
    
    # define options for inner FISTA
    inner_opts = {'stopping': opts['stopping'], 'max_iter': opts['inner_max']}

    time_list = []
    tic=time()
    
    # main loop
    for k in tqdm(range(iter_num)):
        w_sequence[k] = w_old
        loss_sequence[k] = F_logistic(w_old, X, y, mu, lam)

        # set up for inner FISTA solver
        g_k = f_logistic_grad(w_old, X, y, mu)
        def grad(w): # the gradient of inner FISTA, linear in w
            return g_k + f_logistic_hess_z(w_old, X, w-w_old, mu)
        
        # define options for inner FISTA
        if inner_opts['stopping'] == 'criteria':
            inner_opts['threshold'] = opts['threshold'] * mu
        elif inner_opts['stopping'] == 'criteria_lee':
            G_rhs = L * (w_old - prox_h(w_old - g_k / L, lam / L))
            inner_opts['threshold'] = opts['threshold'] * np.linalg.norm(G_rhs)
        else:
            inner_opts['threshold'] = opts['threshold']

        # solve inner subproblem by FISTA
        w_temp, inner_iter_nums[k] = fista(grad, L, lam, w_old, inner_opts)

        # line search
        F_current = F_logistic(w_old, X, y, mu, lam)
        p = w_temp - w_old
        f_grad_current = f_logistic_grad(w_old, X, y, mu)
        Lam = p.dot(f_grad_current) + lam * np.linalg.norm(w_old + p, ord=1) - lam * np.linalg.norm(w_old, ord=1)
        alpha = 1e-3
        rho = 0.8
        lr_test = 1
        for _ in range(20):
            w_test = w_old + lr_test * p
            if F_logistic(w_test, X, y, mu, lam) > F_current + alpha * lr_test * Lam:
                lr_test *= rho
            else:
                break
        lr = lr_test
        w_new = w_old + lr * (w_temp - w_old)
        w_old = w_new.copy()

        time_list.append(time()-tic)
    if opts['time']:
        return loss_sequence, w_sequence, inner_iter_nums, time_list
    else:
        return loss_sequence, w_sequence, inner_iter_nums