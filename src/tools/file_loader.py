from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(directory_path: str = "./data"):
    """
    Loads documents from the specified directory, splits them into chunks, and returns them.
    Supports both .txt and .pdf files.
    """
    print(f"Loading documents from {directory_path}...")
    
    # Load text files
    text_loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
    text_documents = text_loader.load()
    
    # Load PDF files
    pdf_loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    
    # Combine all documents
    documents = text_documents + pdf_documents

    if not documents:
        print("No documents found.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)

    print(f"Loaded and split {len(documents)} documents into {len(chunked_documents)} chunks.")
    print(f"Text files: {len(text_documents)}, PDF files: {len(pdf_documents)}")

    return chunked_documents

if __name__ == "__main__":
    # This is for testing the file loader in isolation.
    # First, let's create a dummy file in the data directory.
    import os
    if not os.path.exists("./data"):
        os.makedirs("./data")
    with open("./data/sample.txt", "w") as f:
        f.write("This is a sample document about LangChain and LangGraph. " * 100)
        f.write("\n\n")
        f.write("This is another paragraph in the same document. " * 100)

    docs = load_documents()
    print(f"Successfully loaded {len(docs)} chunks.")
    print("First chunk:")
    print(docs[0].page_content)
    print("\nMetadata:")
    print(docs[0].metadata)
