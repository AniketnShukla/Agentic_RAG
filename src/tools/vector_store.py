from langchain_community.vectorstores import Chroma
import chromadb
import subprocess
import json

class OllamaSubprocessEmbeddings:
    """
    A custom embedding class that uses the Ollama CLI via subprocess to generate embeddings.
    This is to comply with the user's request and avoid direct HTTP requests from the sandbox.
    """
    def __init__(self, model: str = "openchat:latest"):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents."""
        print(f"Embedding {len(texts)} documents using 'ollama embed'...")
        embeddings = []
        for text in texts:
            command = ["ollama", "embed", "-m", self.model, text]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True
                )
                embedding_data = json.loads(result.stdout)
                embeddings.append(embedding_data['embedding'])
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error getting embedding for document: {e}")
                pass
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query."""
        print(f"Embedding query using 'ollama embed'...")
        command = ["ollama", "embed", "-m", self.model, text]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            embedding_data = json.loads(result.stdout)
            return embedding_data['embedding']
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error getting embedding for query: {e}")
            return []

def get_vector_store(collection_name: str = "rag_agentic_system", persist_directory: str = "./chroma_db"):
    """
    Initializes and returns a Chroma vector store using our custom Ollama subprocess embeddings.
    """
    print(f"Initializing vector store with Ollama subprocess embeddings...")

    client = chromadb.PersistentClient(path=persist_directory)

    embeddings = OllamaSubprocessEmbeddings(model="openchat:latest")

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
            f.write("This is a test document about vector stores using Ollama subprocess.")

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
