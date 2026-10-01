# Mutation operators for exploration in Map Elite's search space
# The given heuristic is parsed into an AST and then mutated
# Then is returned to heuristic code

from node import GNode
import ast
import random as rd

class ASTMutator(ast.NodeTransformer):
    def __init__(self, p_param_mut: float = 0.3, p_struct_mut: float = 0.2, p_wrap_mut=0.2, p_expand_mut=0.2, sigma: float = 0.1, max_depth=4):
        # Mutations are not mutually exclusive
        self.__p_param_mut = p_param_mut  # Probability of suffering a parametric mutation
        self.__p_struct_mut = p_struct_mut # Probability of suffering a structural mutation
        self.__p_wrap_mut = p_wrap_mut  # Probability of suffering a wrap mutation
        self.__p_expand_mut = p_expand_mut  # Probability of suffering a expansion mutation
        self.__sigma = sigma  # Standard desviation of the normal distribution used to generate random numbers

        # Valid ops
        self.__VALID_OP = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv]

        # TODO: Allow all ops?
        self.__VALID_OPS = [ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.In, ast.NotIn, ast.Is, ast.IsNot]
        
        # Valid funcs
        self.__VALID_FUNCS = ['exp', 'log1p', 'sqrt', 'abs']

    def visit_Constant(self, node):
        """
        Applies a parametric mutation with a ´p_param_mutation´ probability and returns an AST node
        """

        # Mutate the constant
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
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
            node.op = rd.choice(self.__VALID_OP)()

        return node

    def visit_Compare(self, node):
        """
        Applies a structural mutation of comparisons with a ´p_struct_param´ probability and returns an AST node
        """

        # Continue visiting the tree
        self.generic_visit(node)

        # Mutate the comparison
        if rd.random() < self.__p_struct_mut:
            node.ops = [rd.choice(self.__VALID_OPS)()]

        return node

    def visit_Name(self, node):
        """Visita variables y opcionalmente las expande (x -> x op y)."""
        # Avoid mutating numpy operators or storing statements
        if node.id == 'np' or isinstance(node.ctx, ast.Store):
            return node

        if self.__current_depth < self.__max_depth and rd.random() < self.__p_expand_mut:
            return self.expand_tree(node)

        return node

    def wrap_expr(self, node):
        """
        Applies a mutation wrapping an expression into a non-linear function
        """

        random_func = rd.choice(self.__VALID_FUNCS)

        # Wrap the node with abs when non-linear function does not allow negative number
        arg = node if random_func not in ['log1p', 'sqrt'] else ast.Call(
            func = ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='abs', ctx=ast.Load()),
            args = [node],
            keywords = []
        )

        # Wrap the node in the chosen function
        call = ast.Call(
            func = ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr=random_func, ctx=ast.Load()),
            args = [arg],
            keywords = []
        )

        return ast.copy_location(call, node) # Copy the location to Call

    def expand_tree(self, node, available_vars):
        """
        Expands the tree applying a mutation with the form: xi -> xi (operator) (operand)
        """

        # Choose operand and operator
        op = rd.choice(self.__VALID_OP)
        operand = ast.Name(id=rd.choice(available_vars), ctx=ast.Load()) if rd.random() < 0.5 and available_vars else ast.Constant(value=round(rd.uniform(0.1, 2.0), 2))

        # Insert operation
        new_operation = ast.BinOp(
            left=node,
            op=op(),
            right=operand
        )

        return ast.copy_location(new_operation, node)



    @staticmethod
    def str_to_ast(node: GNode) -> ast.Module:
        """
        Given a node, represents its heuristic as an AST
        """
        return ast.parse(node.heuristic)

    @staticmethod
    def ast_to_str(tree: ast.Module) -> str:
        """
        Given an AST, returns its python implementation as a string
        """
        return ast.unparse(tree)

