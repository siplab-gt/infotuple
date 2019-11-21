"""
    
    This file contains code corresponding to a generalized version of Algorithm 1 from Section 3.4    
    The parametrization of the crowd oracle, metric learner, and body selector is intended for the ease of 
    Iterating over the different selection strategies tested in this paper.

"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
from collections import defaultdict


def selection_algorithm(M, R, T, crowd_oracle, metric_learner, body_selector, tuple_size=3, verbose_output=True):

    """
    Inputs:
        M: An initial Nxd embedding from which an initial similarity matrix can be calculated
        R: An initial number of ``burn-in'' iterations to initialize the similarity matrix
        T: A number of iterations
        crowd_oracle: An oracle function that takes as input a tuple and returns a ranking.
        metric_learner: A function that takes as inputs a set of ordinal constraints and outputs an Nxd embedding
                        that implicitly corresponds to a similarity function preserving those constraints
        body_selector: A function that takes as input a set of candidate tuples and chooses one
        tuple_size: The size of tuples to be considered
    Returns:
        M: An embedding that captures the selected ordinal constraints
    """

    if verbose_output:
        Ms, selections, selection_qualities = [], [], []
        
    n = range(len(M))

    initial_constraints = []

    for _ in range(R):
        for h in range(len(M)):
            candidate_tuple = [h]+list(np.random.choice(n, tuple_size-1, replace=False))
            oracle_sorted_tuple = crowd_oracle(candidate_tuple)
            
            for i in range(len(oracle_sorted_tuple)-2):
                pairwise_comparison = (oracle_sorted_tuple[0], oracle_sorted_tuple[i+1], oracle_sorted_tuple[i+2])
                initial_constraints.append(pairwise_comparison)
        
    M_prime = metric_learner(M, initial_constraints)
    previous_selections = defaultdict(list)

    for constraint in initial_constraints:
        head, body = constraint[0], constraint[1:]
        previous_selections[head].append(constraint)

    constraints = []
    
    for i in range(T):

        if verbose_output:
            Ms.append(M_prime)

        for a in range(len(M)):
            candidates = permutations(filter(lambda x: x is not a, n), tuple_size - 1)
            tuples = map(lambda x: [a] + list(x), candidates)
            selected_tuple, tuple_qualities, tuple_probabilities, intermediate_params = body_selector(a, M, tuples, previous_selections)
            
            previous_selections[a].append(selected_tuple)
            oracle_sorted_tuple = crowd_oracle(selected_tuple)
            constraints.append(oracle_sorted_tuple)

            if verbose_output:
                selections.append(selected_tuple)
                selection_qualities.append(tuple_qualities)

        new_constraints = []
        for c in constraints:
            for ix in range(len(oracle_sorted_tuple)-2):
                pairwise_comparison = (c[0], c[ix+1], c[ix+2])
                new_constraints.append(pairwise_comparison)

        M_prime = metric_learner(M_prime, new_constraints)

    if verbose_output:
        Ms.append(M_prime)
        return Ms, (initial_constraints, selections), selection_qualities
    
    return M_prime
