# FunSearch best performing heuristic for the OR dataset

import numpy as np

def score(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
        Args:
            item: Size of item to be added to the bin.
            bins: Array of capacities for each bin.
        Return:
            Array of same size as bins with priority score of each bin.
    """
    def s(bin, item):
        if bin - item <= 2:
            return 4
        elif (bin - item) <= 3:
            return 3
        elif (bin - item) <= 5:
            return 2
        elif (bin - item) <= 7:
            return 1
        elif (bin - item) <= 9:
            return 0.9
        elif (bin - item) <= 12:
            return 0.95
        elif (bin - item) <= 15:
            return 0.97
        elif (bin - item) <= 18:
            return 0.98
        elif (bin - item) <= 20:
            return 0.98
        elif (bin - item) <= 21:
            return 0.98
        else:
            return 0.99
    return np.array([s(bin, item) for bin in bins])