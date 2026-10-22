# Map Elites + ReEvo's Graph Node

from dataclasses import dataclass
from typing import Optional

@dataclass
class GNode:
    # TODO: Add the genotype AST
    heuristic: str
    # fitness: float = -float('inf')
    reflection: Optional[str] = None
    # generation: int = 0
    # genotype: ast.Module
