from src.agents.orchestrator import Orchestrator
from src.agents.retriever import Retriever
from src.agents.generator import Generator
from src.agents.rephraser import Rephraser
from src.agents.evaluator import Evaluator
from src.tools.file_loader import load_documents
from src.tools.vector_store import get_vector_store, add_documents_to_store
import os

def setup_and_run(query: str, data_path: str = "./data"):
    """
    Sets up the RAG system, ingests data, and runs the query using Ollama and sentence-transformers.
    """
    print("--- 1. CONFIGURATION AND SETUP ---")
    print("Using local Ollama and sentence-transformers setup.")

    print("\n--- 2. DOCUMENT INGESTION ---")
    documents = load_documents(data_path)
    if not documents:
        print(f"No documents found in the '{data_path}' directory. Please add some .txt or .pdf files and try again.")
        return

    vector_store = get_vector_store()
    if documents:
        add_documents_to_store(vector_store, documents)

    print("\n--- 3. AGENT AND ORCHESTRATOR INITIALIZATION ---")
    rephraser_agent = Rephraser()
    retriever_agent = Retriever(vector_store=vector_store)
<<<<<<< HEAD
    # The generator is initialized with the template model
    generator_agent = Generator(model_name="template")
    orchestrator = Orchestrator(retriever=retriever_agent, generator=generator_agent)
=======
    generator_agent = Generator()
    evaluator_agent = Evaluator()

    orchestrator = Orchestrator(
        rephraser=rephraser_agent,
        retriever=retriever_agent,
        generator=generator_agent,
        evaluator=evaluator_agent
    )
>>>>>>> 93e5e8f09561a2663a73d9536d3b896a55130228

    print("\n--- 4. RUNNING THE AGENTIC RAG WORKFLOW ---")
    result_state = orchestrator.run(query)

    print("\n--- 5. WORKFLOW FINISHED ---")
    # The final state is nested under the last node that ran
    final_node_state = result_state.get('evaluator', {})
    final_answer = final_node_state.get('final_answer')

    if final_answer:
        print("\nFinal Answer:")
        print(final_answer)
    else:
        print("\nCould not retrieve a final answer.")
        print("Final state:", result_state)


if __name__ == "__main__":
    # Use the psychology PDF file path
    psychology_pdf_path = r"C:\Users\anike\OneDrive\Documents\ebooks"
    
    # Check if the PDF file exists
    pdf_file = os.path.join(psychology_pdf_path, "Psychology-A-Self-Teaching-Guide-English.pdf")
    if not os.path.exists(pdf_file):
        print(f"PDF file not found at: {pdf_file}")
        print("Please check the file path and try again.")
        exit(1)
    
    print(f"Using psychology PDF file: {pdf_file}")

    # Get user input
    user_query = input("Please enter your query about psychology: ")

    if user_query and user_query.strip():
        setup_and_run(user_query, psychology_pdf_path)
    else:
        print("No query entered. Exiting.")
