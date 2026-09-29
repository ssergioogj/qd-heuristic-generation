# Mutation operators for exploration in Map Elite's search space
# The given heuristic is parsed into an AST and then mutated
# Then is returned to heuristic code

from node import GNode
import ast
import random as rd

class ASTMutator(ast.NodeTransformer):
    def __init__(self, p_param_mut: float = 0.3, p_struct_mut: float = 0.2, sigma: float = 0.1):
        # Mutations are not mutually exclusive
        self.__p_param_mut = p_param_mut  # Probability of suffering a parametric mutation
        self.__p_struct_mut = p_struct_mut # Probability of suffering a structural mutation
        self.__sigma = sigma  # Standard desviation of the normal distribution used to generate random numbers

        # Valid ops
        self.__VALID_OP = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv]

        # TODO: Allow all ops?
        self.__VALID_OPS = [ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.In, ast.NotIn, ast.Is, ast.IsNot]

    def visit_Constant(self, node):
        """
        Applies a parametric mutation with a ´p_param_mutation´ probability and returns an AST node
        """

        # Mutate the constant
        if rd.random() < self.__p_param_mut:
            node.value = round(float(node.value) + rd.gauss(0, self.__sigma), 3)

        return node

    def visit_BinOp(self, node):
        """
        Applies a structural mutation of operators with a ´p_struct_param´ probability and returns an AST node
        """

        # Continue visiting the tree
        self.generic_visit(node)

        # Mutate the operator
        if rd.random() < self.__p_struct_mut:
            node.op = rd.choice(self.__VALID_OP)

        return node

    def visit_Compare(self, node):
        """
        Applies a structural mutation of comparisons with a ´p_struct_param´ probability and returns an AST node
        """

        # Continue visiting the tree
        self.generic_visit(node)

        # Mutate the comparison
        if rd.random() < self.__p_struct_mut:
            node.ops = [rd.choice(self.__VALID_OPS)]

        return node

    @staticmethod
    def str_to_ast(node: GNode) -> ast.Module:
        """
        Given a node, represents its heuristic as an AST
        """
        return ast.parse(node.heuristic)

    @staticmethod
    def ast_to_string(tree: ast.Module) -> str:
        """
        Given an AST, returns its python implementation as a string
        """
        return ast.unparse(tree)

