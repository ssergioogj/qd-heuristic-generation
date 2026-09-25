# Eoc heuristic for Online Bin Packing 1D

import numpy as np

def score(item, bins):
    scores = np.log(item) * (bins ** 2) / (
    item * np.sqrt(bins - item)) + (
    bins / item) ** 3
    scores[bins == bins.max()] = -np.inf
    return scores