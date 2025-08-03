from src.agents.orchestrator import Orchestrator
from src.tools.file_loader import load_documents
from src.tools.vector_store import get_vector_store, add_documents_to_store
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

def setup_and_run(query: str):
    """
    Sets up the RAG system, ingests data, and runs the query.
    """
    # --- Configuration ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in .env file. Please create a .env file in the root directory.")

    # --- Document Ingestion ---
    # 1. Load documents from the 'data' directory
    documents = load_documents("./data")

    if not documents:
        print("No documents found in the './data' directory. Please add some .txt files.")
        return

    # 2. Initialize the vector store
    vector_store = get_vector_store()

    # 3. Add documents to the vector store
    add_documents_to_store(vector_store, documents)

    # --- Agentic RAG Execution ---
    # 1. Initialize the LLM
    # We will need to pass this to our agents
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 2. Initialize the Orchestrator
    # Note: In a real implementation, the orchestrator would be initialized with all the agents,
    # and the agents would be initialized with the necessary tools (like the vector store).
    # For now, our orchestrator is a simple placeholder.
    orchestrator = Orchestrator(model=llm)

    # 3. Run the orchestrator with the user's query
    print("\n--- Starting Agentic RAG Workflow ---")
    result_state = orchestrator.run(query)

    print("\n--- Workflow Finished ---")
    print("Final State:")
    print(result_state)

    # In the future, we will extract the final answer from the state
    # print("\nFinal Answer:")
    # print(result_state.get('final_answer', "No final answer generated."))


if __name__ == "__main__":
    # Create a dummy data file if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./data/sample.txt"):
        with open("./data/sample.txt", "w") as f:
            f.write("The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris.")

    # Get user input
    user_query = input("Please enter your query: ")

    if user_query:
        setup_and_run(user_query)
    else:
        print("No query entered. Exiting.")
