# Compute fitness of an heuristic running its code for a specific problem in runttime

from benchmark.online_bin_packing_1d.eval.evaluation import Evaluation

import numpy as np
import types

class Evaluator:
    def __init__(self, instances):
        self.__instances = instances
        self.__evaluator = Evaluation()

    def fitness(self, code_str: str):
        """
        Creates a temporal module for executing the code and compute fitness for the heuristic given. 
        """

        try:
            # Create temporal module
            module = types.ModuleType("heuristic_module")
            module.__dict__["np"] = np

            # Execute the code
            exec(code_str, module.__dict__)

            if "score" not in module.__dict__:
                return -float("inf")

            # Evaluate the func in the benchmark
            metrics = self.__evaluator.evaluateGreedy(instances=self.__instances, alg=module)

            return float(metrics["avg_num_bins"])

        except Exception:
            return -float("inf")