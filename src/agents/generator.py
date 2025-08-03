import subprocess
import shlex

class Generator:
    """
    This agent generates a coherent answer using a template-based approach.
    """
    def __init__(self, model_name: str = "template"):
        self.model_name = model_name

    def generate(self, query: str, documents: list[str]) -> str:
        """
        Generates an answer using a template-based approach.

        Args:
            query: The user's original query.
            documents: A list of retrieved document contents.

        Returns:
            The generated answer as a string.
        """
        print(f"Generating answer for query: '{query}' using template-based approach")

        if not documents:
            return "I don't have enough information to answer your question. Please try rephrasing your query or ask about a different topic."

        context = "\n\n".join(documents[:3])  # Use first 3 documents for context

        # Create a simple template-based response
        answer = f"""Based on the information I found, here's what I can tell you about your question: "{query}"

{context}

This information should help answer your question. If you need more specific details, please let me know!"""

        return answer
