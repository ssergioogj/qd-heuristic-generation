import ast

from mutator import ASTMutator
from gnode import GNode

heuristic = """def score(item, bins):
    diff = bins - item
    return diff + 0.5
"""

fathe_node = GNode(heuristic)


mutator = ASTMutator(
    p_param_mut=0.4, 
    p_struct_mut=0.3,   
    p_wrap_mut=0.25,    
    p_expand_mut=0.3,    
    sigma=0.15,
    max_depth=4,
    available_vars=['item', 'bins', 'diff']
)


tree = ASTMutator.str_to_ast(fathe_node)

mutated_tree = mutator.visit(tree)

ast.fix_missing_locations(mutated_tree)

child = ASTMutator.ast_to_str(mutated_tree)

print("=== CÓDIGO ORIGINAL ===")
print(fathe_node.heuristic)

print("=== CÓDIGO DESCENDIENTE TRAS MUTACIÓN AST ===")
print(child)