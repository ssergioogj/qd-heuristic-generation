from map_elites_reevo.reevo.reevo_llm_operators import ReevoLLMOperators
from map_elites_reevo.gnode import GNode


reevo_op = ReevoLLMOperators()

worse_h = """
def score(item: float, bins: np.ndarray) -> np.ndarray:
    residual = bins - item
    return -residual
"""

better_h = """
def score(item: float, bins: np.ndarray) -> np.ndarray:
    diff = bins - item
    eps = 1e-6
    tight_fit_bonus = 1.0 / (diff + eps)
    waste_penalty = 0.05 * (diff ** 2)
    return tight_fit_bonus - waste_penalty
"""

elite_h = """
def score(item: float, bins: np.ndarray) -> np.ndarray:
    diff = bins - item
    eps = 1e-5
    
    exact_match = 1.0 / np.sqrt(diff + eps)
    
    capacity_ratio = item / (bins + eps)
    fill_reward = np.log1p(capacity_ratio) * np.sqrt(bins)
    
    safety_cushion = np.where(diff > 3.0 * item, 0.5 * diff, 0.0)
    
    return exact_match + fill_reward + safety_cushion
"""

# Fitness (maximize) = -AEB (minimize) for online bin packing problem
worse_node = GNode(heuristic=worse_h, fitness=-4.46)
better_node = GNode(heuristic=better_h, fitness=-2.1)
elite_node = GNode(heuristic=elite_h, fitness=-0.55)

ref = reevo_op.short_term_reflection(worse=worse_node, better=better_node)
print(f"Reflection: {ref}")

crossover = reevo_op.crossover(worse_node, better_node, ref)
print("Heuristic:")
print(crossover)