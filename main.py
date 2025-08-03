from src.agents.orchestrator import Orchestrator
from src.agents.retriever import Retriever
from src.agents.generator import Generator
from src.tools.file_loader import load_documents
from src.tools.vector_store import get_vector_store, add_documents_to_store
import os

def setup_and_run(query: str, data_path: str = "./data"):
    """
    Sets up the RAG system, ingests data, and runs the query using Ollama.
    """
    print("--- 1. CONFIGURATION AND SETUP ---")
    # No API keys needed for local Ollama setup
    print("Using local Ollama setup.")

    print("\n--- 2. DOCUMENT INGESTION ---")
    documents = load_documents(data_path)
    if not documents:
        print(f"No documents found in the '{data_path}' directory. Please add some .txt or .pdf files and try again.")
        return

    vector_store = get_vector_store()
    if documents:
        add_documents_to_store(vector_store, documents)

    print("\n--- 3. AGENT AND ORCHESTRATOR INITIALIZATION ---")
    retriever_agent = Retriever(vector_store=vector_store)
    # The generator is initialized with the template model
    generator_agent = Generator(model_name="    ")
    orchestrator = Orchestrator(retriever=retriever_agent, generator=generator_agent)

    print("\n--- 4. RUNNING THE AGENTIC RAG WORKFLOW ---")
    result_state = orchestrator.run(query)

    print("\n--- 5. WORKFLOW FINISHED ---")
    if result_state and 'final_answer' in result_state:
        print("Final Answer:")
        print(result_state['final_answer'])
    else:
        print("Could not retrieve a final answer.")
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
