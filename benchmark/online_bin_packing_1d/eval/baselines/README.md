# Baseline Heuristic Suite for Online Bin Packing 1D

The benchmark suite includes both traditional baselines and state-of-the-art LLM generated heuristics for OBP

- Classical heuristics
- LLM generated heuristics

## Interface

All the implemented heuristics follows the Romera-Paredes et al. interface model.

```python
def heuristic(item: float, bins: np.array) -> np.array:
    """
    Assigns a score to each bin for placing an item into a bin.

    Args:
        item: Size of the current item
        bins: Array of the REMAINING FREE CAPACITY of feasible bins (those which comply bins - item >= 0)

    Returns:
        scores: Array of the priority values (scores) matching the dimension of `bins`. Simulator will assign the
        item to the bin with the highest score
    """

```

> **Note:** For this implementation, `item` is of type `int`.

## Classical heuristics

### First Fit

Assigns the current item to the first available bin (lower index bin with **bin - item >= 0**).

### Best Fit

Assigns the current item to the bin with the lowest remaining free capacity after placing the item in the bin.

## LLM-generated heuristics

### FunSearch short heuristic from Romera-Paredes et al.

Assigns the current item to the bin with the lowest remaining free capacity only if the residual space is approximately 0.

### FunSearch best performing heuristic for the OR Dataset

Assigns the current item to bins that does not leave small residual space.

### Evolution od Code (EoC) heuristic

Assigns the current item to the bin that have remianing free capacity approximately 0 after placing the item, but gets out of control if not with large capacities

### Evolution of Heuristics (EoH) heuristic

Assigns the current item to the bin that have remaining free capacity approximately 0 and if not, prefers high remaining free capacity.
