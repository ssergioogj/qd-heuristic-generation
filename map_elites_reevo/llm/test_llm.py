from llm_client import LLMClient
from mutator import ASTMutator

system_prompt = ("You are a Python code heuristic generator."
                "Output ONLY raw Python code. "
                "Do not write markdown backticks (``` or ```python)."
                "Do not include text, markdown, comments or notes. ")

client = LLMClient()

problem = "Online Bin Packing Problem 1D"

heuristic = """def score(item, bins):
    diff = bins - item
    return diff + 0.5
"""

user_prompt = f"""Problem: {problem}
Optimize the scoring function for {problem}
Only output the function definition.

{heuristic}
"""

res = client.generate_code(system_prompt=system_prompt, user_prompt=user_prompt)

print(res)

ast = ASTMutator()

ast.visit(res)
