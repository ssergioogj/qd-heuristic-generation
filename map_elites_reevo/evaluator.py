# Compute fitness of an heuristic running its code for a specific problem

from benchmark.online_bin_packing_1d.eval.evaluation import Evaluation

import numpy as np
import types

class Evaluator:
    def __init__(self, evaluator):
        self.__evaluator = evaluator

    def fitness(self, code_str: str):
        """
        Creates a temporal module for executing the code and compute fitness for the heuristic given. 
        """

        try:
            # Create temporal module
            module = types.ModuleType("heuristic_module")
            module.__dict__("np") = np

            # Execute the code
            exec(code_str, module.__dict__)

            if "score" not in module.__dict__:
                return -float("inf")

            score = module.score

            # Evaluate the func in the benchmark
            fitness = self.__benchmark.evaluatorGreedy()