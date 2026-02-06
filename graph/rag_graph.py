from langgraph.graph import StateGraph

from agents.retriever_agent import retriever_agent
from agents.generator_agent import generator_agent

class RAGState(dict):
    query: str
    documents: list
    answer: str

graph = StateGraph(RAGState)

graph.add_node("retrieve", retriever_agent)
graph.add_node("generate", generator_agent)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

rag_app = graph.compile()
