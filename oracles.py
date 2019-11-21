"""
    
    This file contains oracle models for dataset generation described in Section 4.1

"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
import random


def data_oracle_generator(true_points, failure_prob=0.1):
    """

    This method  returns an oracle function that responds to tuplewise ranking queries
    of the form "which of b_1, ..., b_n is closer to a in the true embedding space"
    without any additional noise.

    Input:
        true_points: ``true'' embedding coordinates for an object set
    Returns:
        oracle: A function that accepts a tuple and returns an oracle response ranking 

    """
    oracle = lambda x: [x[0]] + sorted(x[1:], key=lambda a:np.sqrt(np.sum((true_points[x[0]]-true_points[a])**2)))
    return oracle


def plackett_luce_data_oracle_generator(true_points, P=0.95, failure_prob=0.05):
    """
    
    This method  returns an oracle function that responds to tuplewise ranking queries
    of the form "which of b_1, ..., b_n is closer to a in the true embedding space"
    according to the Plackett-Luce model specified in Section 4.1
   
    Inputs:
        true_points: ``true'' embedding coordinates for an object set
        P: percentile encompassing all distances in true_points
    Returns:
        oracle: A function that accepts a tuple and returns an oracle response ranking 
    
    """
    
    if failure_prob != None:
        P = 1. - failure_prob

    all_distances = pdist(true_points)
    alpha = -np.log(1-P)/float(np.log(max(all_distances)+1))
    
    pareto = lambda x: alpha/float((x+1)**(alpha+1))
    
    def probabilistic_oracle(x):
        distances = np.sqrt([sum((true_points[x[0]] - true_points[x[i]])**2) for i in range(1, len(x))])
        pmf_unnorm = [pareto(z) for z in distances];
        
        response = [x[0]]
        candidates = list(x[1:])
        while len(candidates) > 0:
            pmf = np.array(pmf_unnorm)/float(sum(pmf_unnorm))
            close_idx = np.random.choice(range(len(candidates)),p=pmf)
            response.append(candidates.pop(close_idx))
            del pmf_unnorm[close_idx]
            
        return tuple(response)

    return probabilistic_oracle

