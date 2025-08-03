from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

def get_vector_store(collection_name: str = "rag_agentic_system", persist_directory: str = "./chroma_db"):
    """
    Initializes and returns a Chroma vector store using HuggingFace embeddings.
    """
    print(f"Initializing vector store with HuggingFace embeddings...")

    client = chromadb.PersistentClient(path=persist_directory)

    # Use a lightweight embedding model that works well for RAG
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

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
    print("Documents added and persisted successfully.")


if __name__ == '__main__':
    from file_loader import load_documents
    import os

    if not os.path.exists("./data/sample.txt"):
        if not os.path.exists("./data"):
            os.makedirs("./data")
        with open("./data/sample.txt", "w") as f:
            f.write("This is a test document about vector stores using HuggingFace embeddings.")

    documents = load_documents("./data")

    vector_store = get_vector_store()

    if documents:
        add_documents_to_store(vector_store, documents)

        query = "What are vector stores?"
        results = vector_store.similarity_search(query, k=1)

        print(f"\nSimilarity search for: '{query}'")
        if results:
            print("Found results:")
            for doc in results:
                print(doc.page_content)
        else:
            print("No results found.")
