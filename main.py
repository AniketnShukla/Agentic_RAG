from src.agents.orchestrator import Orchestrator
from src.agents.retriever import Retriever
from src.agents.generator import Generator
from src.agents.rephraser import Rephraser
from src.agents.evaluator import Evaluator
from src.tools.file_loader import load_documents
from src.tools.vector_store import get_vector_store, add_documents_to_store
import os

def setup_and_run(query: str):
    """
    Sets up the RAG system, ingests data, and runs the query using Ollama and sentence-transformers.
    """
    print("--- 1. CONFIGURATION AND SETUP ---")
    print("Using local Ollama and sentence-transformers setup.")

    print("\n--- 2. DOCUMENT INGESTION ---")
    documents = load_documents("./data")
    if not documents:
        print("No documents found in the './data' directory. Please add some .txt files and try again.")
        return

    vector_store = get_vector_store()
    if documents:
        add_documents_to_store(vector_store, documents)

    print("\n--- 3. AGENT AND ORCHESTRATOR INITIALIZATION ---")
    rephraser_agent = Rephraser()
    retriever_agent = Retriever(vector_store=vector_store)
    generator_agent = Generator()
    evaluator_agent = Evaluator()

    orchestrator = Orchestrator(
        rephraser=rephraser_agent,
        retriever=retriever_agent,
        generator=generator_agent,
        evaluator=evaluator_agent
    )

    print("\n--- 4. RUNNING THE AGENTIC RAG WORKFLOW ---")
    result_state = orchestrator.run(query)

    print("\n--- 5. WORKFLOW FINISHED ---")
    if result_state and 'final_answer' in result_state:
        print("\nFinal Answer:")
        print(result_state['final_answer'])
    else:
        print("\nCould not retrieve a final answer.")
        print("Final state:", result_state)


if __name__ == "__main__":
    # Ensure the data directory and a sample file exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./data/sample.txt"):
        with open("./data/sample.txt", "w") as f:
            f.write("The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris. The currency of Japan is the Yen.")

    # Get user input
    user_query = input("Please enter your query: ")

    if user_query and user_query.strip():
        setup_and_run(user_query)
    else:
        print("No query entered. Exiting.")
