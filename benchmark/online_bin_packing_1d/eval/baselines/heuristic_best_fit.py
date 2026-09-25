# Best Fit heuristic for Online Bin Packing Problem 1D
# Best Fit assigns the current item to the bin with the lowest remaining free capacity after placing the item in the bin

import numpy as np

def score(item: int, bins: np.array):
    """
    Scores each feasible bin according to Best Fit heuristic. 

    Returns an array of the same dimension as bins priorizing bins with less free capacity remaining. 
    """

    # Return the negative of the (bins - item) array because simulator places the item in the bin with the highest score
    return -(bins - item)
