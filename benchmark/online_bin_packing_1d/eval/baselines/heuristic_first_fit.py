# First Fit heuristic for Online Bin Packinng Problem 1D
# First Fit assigns the current item to the first available bin (lower index bin with bin - item >= 0)


import numpy as np

def score(item:int, bins:np.array):
    """
    Scores each feasible bin according to First Fit heuristic. 

    Returns a decreasing array with the same dimension as bins. That way, first feasible 
    bin will always be chosen by the simulator
    """
    return -np.arange(len(bins))