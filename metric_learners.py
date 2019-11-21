"""
    This file contains the embedding technique presented in Section 3.4

"""

import numpy as np
from scipy import linalg
from copy import copy

def empirical_loss(X, constraints, mu):
    """
    A helper function to compute empirical loss
    """
    empirical_loss = 0.

    for query in constraints:
        i, j, k = query
        a, b, c = X[i], X[j], X[k]
        
        ab_dist = np.linalg.norm(b - a)
        ac_dist = np.linalg.norm(c - a)
        
        if ab_dist > ac_dist:
            empirical_loss += 1.
    
    empirical_loss = 1. - (empirical_loss/len(constraints))

    return empirical_loss

def log_loss(X, constraints, mu):
    """
    A helper function to compute empirical log loss
    """

    log_loss = 0.

    for query in constraints:
        i, j, k = query
        a, b, c = X[i], X[j], X[k]
        
        ab_dist = np.linalg.norm(b - a)
        ac_dist = np.linalg.norm(c - a)
        log_loss -= np.log((mu+ac_dist)/(2*mu+ab_dist+ac_dist)) 
         
    log_loss = log_loss/len(constraints)
    
    return log_loss

def gradient(X, constraints, mu):
    """
    Analytic gradient calculation reliant on the response model proposed in 3.2
    """

    n, d = X.shape
    grad = np.zeros((n,d))
    
    for query in constraints:
        i,j,k   = query
        a, b, c = X[i], X[j], X[k]
        
        ab_dist = np.linalg.norm(b-a)
        ac_dist = np.linalg.norm(c-a)
        
        grad[i] += 2 * (a-b)/(2* mu + ab_dist**2 + ac_dist**2) 
        grad[j] += 2 * ((a-c)/(mu+ac_dist**2) - (a-c)/(2*mu + ab_dist**2 + ac_dist**2))
        grad[k] += 2 * ((a-c)/(mu + ac_dist**2) - (2*a - b - c)/(2*mu + ab_dist**2 + ac_dist**2))
        
    grad *= -1./len(constraints)
   
    return grad

def probabilistic_mds(X, constraints, evaluation_constraints=None, loss=empirical_loss, mu=.5, n_iterations=5000, learning_rate=1., momentum=0., verbose=True):
    """
    Inputs:
        X: initial estimate of an Nxd embedding
        constraints: List of ordinal constraints to be preserved in the final embedding
    """

    best_X = copy(X)
    best_loss = loss(best_X, constraints, mu) 
    n, d = X.shape
    curr_X = best_X
    
    decomposed_queries = []

    for query in constraints:
        for i in range(1,len(query)-1):
            pairwise_comparison = (query[0], query[i], query[i+1])
            decomposed_queries.append(pairwise_comparison) 
   
    constraints = decomposed_queries
    n_iterations = max(1, n_iterations)

    prev_grad = np.zeros((n,d))
    
    for epoch in range(n_iterations):
       
        grad = gradient(curr_X, constraints, mu)
        curr_X -= (learning_rate * grad + momentum * prev_grad)
        prev_grad = grad
        
        curr_X = curr_X / np.linalg.norm(curr_X)

        if evaluation_constraints is not None:
            evaluation_loss = loss(curr_X, evaluation_constraints, mu)
    
            if evaluation_loss < loss(curr_X, evaluation_constraints, mu):
                best_X = curr_X
                best_loss = iteration_loss

        else:
            iteration_loss = loss(curr_X, constraints, mu)
            
            if iteration_loss  < best_loss:
                best_X = curr_X
                best_loss = iteration_loss

    if verbose:
        print "loss: ", best_loss
    
    return best_X

