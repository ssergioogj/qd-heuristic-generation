# Implementation of the ReEvo operators based on LLM calls
# Short term reflection: Reflection of two heuristic (where the fitness is better in one of them, the better parent)
# Long term reflection: Reflection of an elite heuristic based on a collection of long-term reflections and newly added short ones

from llm.llm_client import LLMClient
from gnode import GNode

import re

class ReevoLLMOperators:
    REFLECTION_SYSTEM_PROMPT  = (
        "You are an expert algorithm designer. Analyze the provided heuristics "
        "for a combinatorial problem, compare their logic, and explain why the "
        "better one achieves superior performance. Provide a concise, actionable insight."
    )

    CROSSOVER_SYSTEM_PROMPT = (
        "You are an expert programmer. Write a new Python heuristic combining the "
        "strengths of both parents following the provided reflection. Return ONLY the "
        "executable Python code inside a ```python block. Must adhere to the exact signature."
    )

    SYNTHESIS_SYSTEM_PROMPT = "Summarize individual heuristic design findings into high-level algorithmic principles."

    MUTATION_SYSTEM_PROMPT = (
        "You are an expert algorithm engineer. Mutate and improve the provided elite heuristic "
        "by applying advanced statistical techniques and the accumulated long-term insights. "
        "Return ONLY the executable Python code inside a ```python block."
    )

    def __init__(self):
        self.__llm_client = LLMClient()

    def short_term_reflection(self, worse: GNode, better: GNode):
        """
        Generates a short-term reflection based on a worse parent heuristic and a better parent heuristic. 
        """

        user_prompt = f"""
            Worse Heuristic (Fitness: {worse.fitness:.4f}):
            ```python
            {worse.heuristic}
            ```
            Better Heuristic (Fitness: {better.fitness:.4f}):
            ```python
            {better.heuristic}
            ```
            Compare both algorithms and provide a short-term reflection: what logical or mathematical
            mechanism made the better one succeed? Focus on algorithmic principles.
            """

        return self.__llm_client.generate_code(system_prompt=self.REFLECTION_SYSTEM_PROMPT, user_prompt=user_prompt)

    def crossover(self, parent1: GNode, parent2: GNode, reflection: str):
        """
        Generates a child combining the parents guided by the short-term reflection
        """

        user_prompt = f"""

            Parent 1 (Fitness: {parent1.fitness:.4f}):
            ```python
            {parent1.heuristic}
            ```
            Parent 2 (Fitness: {parent2.fitness:.4f}):
            ```python
            {parent2.heuristic}
            ```
            Guidance from Reflection:
            {reflection}

            Synthesize a novel, valid Python scoring heuristic named score(item: float, bins: np.ndarray) -> np.ndarray.
            """
        raw_output = self.__llm_client.generate_code(system_prompt=self.CROSSOVER_SYSTEM_PROMPT, user_prompt=user_prompt)
        return self.__extract_code(raw_output)

    def update_long_term_reflection(self, reflections:list[str]):
        """
        Synthesizes succesfull short term reflections into a long-term general principle.
        """

        user_prompt = "Consolidate the following design insights into a single general guideline:\n" + \
                  "\n".join(f"- {r}" for r in reflections[-4:])

        return self.__llm_client.generate_code(system_prompt=self.SYNTHESIS_SYSTEM_PROMPT, user_prompt=user_prompt)



    def mutate(self, elite_heuristic: GNode, long_term_reflections:list[str]):
        """
        Mutates an elite heuristic guided by long-term reflections.
        """

        memory = "\n".join(f"- {ref}" for ref in long_term_reflections[-5:]) # Select the 5 latest reflections

        user_prompt = f"""

        Elite Heuristic to Improve (Fitness: {elite_heuristic.fitness:.4f}):
        ````python
        {elite_heuristic.code}
        ````
        Accumulated Knowledge / Long-term Reflections:
        {memory}

        Generate a substantially refined version with improved performance.
        """
        raw_output = self.__llm_client.generate_code(system_prompt=self.MUTATION_SYSTEM_PROMPT, user_prompt=user_prompt)
        return self.__extract_code(raw_output)


    @staticmethod
    def __extract_code(text: str) -> str:
        """Extracts the block of python code from markdown"""
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

