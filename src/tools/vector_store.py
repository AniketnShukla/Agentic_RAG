from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

def get_vector_store(collection_name: str = "rag_agentic_system", persist_directory: str = "./chroma_db"):
    """
    Initializes and returns a Chroma vector store.
    """
    print(f"Initializing vector store: collection='{collection_name}', directory='{persist_directory}'")

    # Using the new ChromaDB client API
    client = chromadb.PersistentClient(path=persist_directory)

    # OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client,
        persist_directory=persist_directory,
    )

    return vector_store

def add_documents_to_store(vector_store, documents):
    """
    Adds a list of documents to the vector store.
    """
    if not documents:
        print("No documents to add to the vector store.")
        return

    print(f"Adding {len(documents)} document chunks to the vector store...")
    vector_store.add_documents(documents)
    # ChromaDB with a PersistentClient automatically persists data, so no need to call persist() explicitly
    # if you are using the client API correctly.
    print("Documents added and persisted successfully.")


if __name__ == '__main__':
    # This is for testing the vector store in isolation.
    from file_loader import load_documents
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in .env file")

    # 1. Load documents
    if not os.path.exists("./data/sample.txt"):
        if not os.path.exists("./data"):
            os.makedirs("./data")
        with open("./data/sample.txt", "w") as f:
            f.write("This is a test document about vector stores.")

    documents = load_documents("./data")

    # 2. Initialize vector store
    vector_store = get_vector_store()

    # 3. Add documents to the store
    add_documents_to_store(vector_store, documents)

    # 4. Test a similarity search
    query = "What are vector stores?"
    results = vector_store.similarity_search(query, k=1)

    print(f"\nSimilarity search for: '{query}'")
    if results:
        print("Found results:")
        for doc in results:
            print(doc.page_content)
    else:
        print("No results found.")
