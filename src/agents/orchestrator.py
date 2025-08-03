from langgraph.graph import StateGraph, END
from src.schemas.state import AgentState

class Orchestrator:
    """
    The orchestrator manages the overall workflow of the agentic RAG system.
    It defines the graph of agents and the transitions between them.
    """
    def __init__(self, model):
        self.model = model
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """
        Builds the LangGraph workflow for the agentic RAG system.
        """
        workflow = StateGraph(AgentState)

        # TODO: Add nodes for each agent (rephraser, retriever, generator, etc.)
        # Example:
        # workflow.add_node("rephraser", self.rephrase_query_node)
        # workflow.add_node("retriever", self.retriever_node)
        # workflow.add_node("generator", self.generator_node)

        # TODO: Define the edges and conditional logic for the graph
        # Example:
        # workflow.set_entry_point("rephraser")
        # workflow.add_edge("rephraser", "retriever")
        # workflow.add_conditional_edges(...)

        workflow.add_node("dummy_start", self._dummy_start_node)
        workflow.set_entry_point("dummy_start")
        workflow.add_edge("dummy_start", END)


        return workflow.compile()

    def _dummy_start_node(self, state: AgentState):
        """A placeholder start node."""
        print("Starting the orchestration...")
        return state

    def run(self, query: str):
        """
        Runs the agentic RAG system with the given query.
        """
        initial_state = AgentState(original_query=query)
        # The `stream` method returns an iterator of states. We'll get the final state.
        final_state = None
        for s in self.workflow.stream(initial_state):
            final_state = s
        return final_state

if __name__ == "__main__":
    # This is for testing the orchestrator in isolation
    from langchain_openai import ChatOpenAI
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in .env file")

    openai_model = ChatOpenAI(model="gpt-4o", temperature=0)

    orchestrator = Orchestrator(model=openai_model)

    result = orchestrator.run("What is the capital of France?")

    print("Orchestrator finished.")
    # We don't have a final answer yet, so we'll just print the final state
    print(result)
