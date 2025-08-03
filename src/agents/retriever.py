class Retriever:
    """
    This agent retrieves relevant documents from the vector store based on the rephrased queries.
    """
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, queries: list[str], k: int = 2) -> list[str]:
        """
        Retrieves documents for the given queries.

        Args:
            queries: A list of queries to search for.
            k: The number of documents to retrieve for each query.

        Returns:
            A list of unique retrieved document contents.
        """
        print(f"Retrieving documents for queries: {queries}")

        retrieved_docs = []
        for query in queries:
            retrieved_docs.extend(self.vector_store.similarity_search(query, k=k))

        # Get unique documents by their content
        unique_docs_by_content = {doc.page_content for doc in retrieved_docs}

        print(f"Retrieved {len(unique_docs_by_content)} unique documents.")

        return list(unique_docs_by_content)
