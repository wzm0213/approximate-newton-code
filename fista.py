'''fista (with information of L) for solving the subproblems (1.3) up to certain stopping critieria'''
import numpy as np
from logistic_setup import F_logistic, f_logistic_grad, f_logistic_hess_z, prox_h
from time import time

def check_stopping(grad, x, lam, eps):
    '''grad: gradient function of smooth part, take one parameter
       x: current point (1-dim array)
       lam: coefficient of L1-norm
       eps: accuracy threshold    
    '''
    # initialization
    check = False
    d = len(x)
    g = grad(x)
    I = np.nonzero(x)
    Ic = np.where(x == 0)
    r = np.zeros_like(x) # residual
    if len(I) == d:
        r = g + lam * np.sign(x)
    else:
        r[I] = g[I] + lam * np.sign(x[I])   
        for i in Ic:
            r[i] = np.clip(0, -lam+g[i], lam+g[i])
    if np.linalg.norm(r) < eps:
        check = True
    return check

def check_stopping_lee(grad, x, L, lam, eps):
    '''eps: RHS of (2.24) in Lee's paper, with \eta_k being a constant
    '''
    check = False
    g = grad(x)
    G = L * (x - prox_h(x - g/L, lam / L))
    if np.linalg.norm(G) < eps:
        check = True
    return check

def fista(grad, L, lam, w_init, options=None):
    '''FISTA for solving a general minimization problem
        problem setup:
            grad: gradient function, take one parameter w
            L: smoothness constant of the smooth part f
            lam: coefficient of L1-norm
        algorithm setup:
            w_init: initial point, should be chosen as w_k!!!
            options: a dict containing keys 'max_iter', 'stopping' and 'threshold'    
    '''
    # initialization
    options.setdefault('max_iter', 1000)
    options.setdefault('stopping', '')
    options.setdefault('threshold', 1e-8)
    options.setdefault('store_seq', False)
    options.setdefault('time', False)
    
    max_iter = options['max_iter']
    d = len(w_init)
    iter_num = 0
    x_old = w_init.copy()
    y_old = w_init.copy()
    t_old = 1

    time_list = []
    tic = time()

    if options['store_seq']:
        x_sequence = np.zeros((max_iter, d))

    # main loop
    if options['stopping'] == 'criteria':
        while iter_num < options['max_iter']:
            if options['store_seq']:
                x_sequence[iter_num] = x_old
            x_new = prox_h(y_old - (1 / L) * grad(y_old), lam / L)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
            y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)
            eps = options['threshold']*np.linalg.norm(x_new-w_init, ord=2)**2
            if check_stopping(grad, x_new, lam, eps):
                break
            x_old, y_old, t_old = x_new.copy(), y_new.copy(), t_new
            iter_num += 1
            time_list.append(time()-tic)
    elif options['stopping'] == 'criteria_lee':
        while iter_num < options['max_iter']:
            if options['store_seq']:
                x_sequence[iter_num] = x_old
            x_new = prox_h(y_old - (1 / L) * grad(y_old), lam / L)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
            y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)
            if check_stopping_lee(grad, x_new, L, lam, options['threshold']):
                break
            x_old, y_old, t_old = x_new.copy(), y_new.copy(), t_new
            iter_num += 1
            time_list.append(time()-tic)
    else:
        def error(w): # first-order optimal condition 
            return w - prox_h(w - grad(w), lam)

        while iter_num < options['max_iter']:
            if options['store_seq']:
                x_sequence[iter_num] = x_old
            x_new = prox_h(y_old - (1 / L) * grad(y_old), lam / L)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
            y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)

            e = np.linalg.norm(error(x_new)) / np.linalg.norm(error(w_init))
            if e < options['threshold']:
                break
            x_old, y_old, t_old = x_new.copy(), y_new.copy(), t_new
            iter_num += 1
            time_list.append(time()-tic)
    if options['store_seq']:
        if options['time']:
            return x_sequence, iter_num, time_list
        else:
            return x_sequence, iter_num
    else:
        if options['time']:
            return x_new, iter_num, time_list
        else:
            return x_new, iter_num