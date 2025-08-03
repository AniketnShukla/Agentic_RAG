from langgraph.graph import StateGraph, END
from src.schemas.state import AgentState
from src.agents.retriever import Retriever
from src.agents.generator import Generator

class Orchestrator:
    """
    The orchestrator manages the overall workflow of the agentic RAG system.
    It defines the graph of agents and the transitions between them.
    """
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        self.workflow = self._build_workflow()

    def retriever_node(self, state: AgentState) -> dict:
        """
        Node that calls the Retriever agent.
        """
        print("---CALLING RETRIEVER---")
        queries = [state['original_query']] # For now, we use the original query directly
        documents = self.retriever.retrieve(queries)
        return {"retrieved_documents": documents}

    def generator_node(self, state: AgentState) -> dict:
        """
        Node that calls the Generator agent.
        """
        print("---CALLING GENERATOR---")
        query = state['original_query']
        documents = state['retrieved_documents']
        generated_answer = self.generator.generate(query, documents)
        return {"final_answer": generated_answer}

    def _build_workflow(self):
        """
        Builds the LangGraph workflow for the agentic RAG system.
        """
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("generator", self.generator_node)

        # Define the edges for the graph
        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("generator", END)

        return workflow.compile()

    def run(self, query: str):
        """
        Runs the agentic RAG system with the given query.
        """
        initial_state = {"original_query": query}
        # The `stream` method returns an iterator of states. We'll get the final state.
        final_state = None
        for s in self.workflow.stream(initial_state):
            # Print each step's output for debugging
            print(f"---STATE UPDATE---\n{s}\n---END STATE UPDATE---")
            final_state = s
        return final_state

if __name__ == "__main__":
    # This is for testing the orchestrator in isolation
    from src.tools.vector_store import get_vector_store, add_documents_to_store
    from src.tools.file_loader import load_documents
    import os

    # Ingest documents
    if not os.path.exists("./data/sample.txt"):
        if not os.path.exists("./data"):
            os.makedirs("./data")
        with open("./data/sample.txt", "w") as f:
            f.write("The capital of France is Paris.")

    documents = load_documents("./data")
    vector_store = get_vector_store()
    if documents:
        add_documents_to_store(vector_store, documents)

    # Initialize agents
    retriever_agent = Retriever(vector_store=vector_store)
    generator_agent = Generator(model_name="openchat:latest")

    # Initialize orchestrator
    orchestrator = Orchestrator(retriever=retriever_agent, generator=generator_agent)

    # Run the orchestrator
    result = orchestrator.run("What is the capital of France?")

    print("\n---ORCHESTRATOR FINISHED---")
    if result and 'final_answer' in result:
        print("Final Answer:")
        print(result['final_answer'])
    else:
        print("Could not retrieve a final answer.")
        print("Final state:", result)
