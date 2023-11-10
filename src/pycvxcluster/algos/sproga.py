import numpy as np

def sproga(X, lamb, gamma, epsilon, eta, maxiters, weight_vec, a):
    mu = 2*epsilon/(lamb * np.sum(weight_vec))
    vk = None
    L = 1 + 2*lamb*np.sum(weight_vec)/mu
    for iter in range(maxiters):
        pass