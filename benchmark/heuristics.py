import numpy as np


# Heuristics

def best_fit(item: int, bins: np.ndarray) -> np.ndarray:
    """
    Minimizes the residual space
    """
    return bins.astype(float)

def first_fit(item:int, bins:np.ndarray) -> np.ndarray:
    """"
    Selects the first open bin in which it fits
    """
    n = len(bins)
    return np.arrange(n, 0, -1, dtype=float)

def worst_fit(item:int, bins:np.ndarray) -> np.ndarray:
    """
    Selects the more empty bin
    """
    return -bins.astype(float)


# LLM generated heuristics

def fs1(item:int, bins:np.ndarray) -> np.ndarray:
    """
    Discovered by FunSearch
    """
    def s(b, it):
        diff = b - it
        if diff <= 2:
            return 4.0
        elif diff <= 3:
            return 3.0
        elif diff <= 5:
            return 2.0
        elif diff <= 7:
            return 1.0
        elif diff <= 9:
            return 0.9
        elif diff <= 12:
            return 0.95
        elif diff <= 15:
            return 0.97
        elif diff <= 18:
            return 0.98
        elif diff <= 20:
            return 0.98
        elif diff <= 21:
            return 0.98     # Could be optimized
        else: 
            return 0.99

    return np.array([s(b, item) for b in bins], dtype=float)

def eoc(item:int, bins:np.ndarray) -> np.ndarray:
    """
    Discovered by EoC
    """
    # Avoid division by 0 and compute score
    scores = (
        np.log(item) * (bins**2) + (bins / item) ** 3
    )

    # Penalizes the bin with the actual maximum capacity
    scores[bins == bins.max()] = -np.inf
    return scores

def eoh(item:int, bins: np.ndarray) -> np.ndarray:
    """
    Discovered by EoH
    """
    diff = bins - item
    exp = np.exp(np.clip(diff, -50, 50)) # Avoid numeric overflow
    sqrt = np.sqrt(np.maximum(diff, 0))
    ulti = 1.0 - (diff / np.maximum(bins, 1e-6))
    comb = ulti * sqrt

    adjust = np.where(diff > (item * 3), comb + 0.8, comb + 0.3)
    hybrid_exp = bins / ((exp + 0.7) * exp)
    scores = hybrid_exp + adjust
    return scores

