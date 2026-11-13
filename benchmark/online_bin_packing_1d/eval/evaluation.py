# EoH imlpementation
# Repo: https://github.com/FeiLiu36/EoH

import numpy as np
import time
import tracemalloc


class Evaluation():
    def __init__(self):

        self.instances, self.lb = None, None


    def get_valid_bin_indices(self,item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]


    def online_binpack(self,items: tuple, bins: np.ndarray, alg):
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        # Add items to bins.
        for item in items:
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            priorities = alg.score(item, bins[valid_bin_indices])
            # Add item to bin with highest priority.
            #print(priorities)
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins


    # @funsearch.run
    def evaluateGreedy(self,instances: dict, alg, profile_memory:bool = False) -> float:

        """Evaluate heuristic function on a set of online binpacking instances."""
        # List storing number of bins used for each instance.
        num_bins = []
        # List storing execution times for each instance
        exec_times = []
        # List storing memory consumption in Kb for each instance
        peak_memories = []
        # Perform online binpacking for each instance.
        #print(len(instances))
        for name in instances:
            #print(name)
            instance = instances[name]
            capacity = instance['capacity']
            items = np.array(instance['items'])

            # Measure memory consumption
            if profile_memory:
                tracemalloc.start()
                tracemalloc.reset_peak()

            # Measure time
            tstart = time.perf_counter()


            # Create num_items bins so there will always be space for all items,
            # regardless of packing order. Array has shape (num_items,).
            bins = np.array([capacity for _ in range(instance['num_items'])])
            # Pack items into bins and return remaining capacity in bins_packed, which
            # has shape (num_items,).
            _, bins_packed = self.online_binpack(items, bins, alg)

            # Metrics
            t = time.perf_counter() - tstart

            if profile_memory:
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_memories.append(peak_mem)

            # If remaining capacity in a bin is equal to initial capacity, then it is
            # unused. Count number of used bins.
            num_bins.append((bins_packed != capacity).sum())
            exec_times.append(t)
                
            
        # Metrics per instance: score, execution time and memory consumption
        return {
                "avg_num_bins": -np.mean(num_bins),
                "avg_time_sec": np.mean(exec_times),
                "avg_peak_mem_kb": np.mean(peak_memories) if profile_memory else None,  
            }