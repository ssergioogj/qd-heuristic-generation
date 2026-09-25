# Romera-Paredes et al. heuristic for Online Bin Packing 1D

import numpy as np

def score(item: float, bins: np.ndarray) -> np.ndarray:
    """Online bin packing heuristic discovered with FunSearch."""
    score = 1000 * np.ones(bins.shape)
    # Penalize bins with large capacities.
    score -= bins * (bins-item)
    # Extract index of bin with best fit.
    index = np.argmin(bins)
    # Scale score of best fitbin by item size.
    score[index] *= item
    # Penalize best fit bin if fitisnot tight.
    score[index] -= (bins[index] - item)**4
    return score