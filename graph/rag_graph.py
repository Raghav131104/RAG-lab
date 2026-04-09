from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from agents.retriever_agent import retriever_agent
from agents.generator_agent import generator_agent
from agents.document_grader import grade_documents_agent
from agents.hallucination_grader import hallucination_grader, answer_grader

class RAGState(TypedDict):
    query: str
    documents: List[str]
    answer: str
    generation_attempts: int

def decide_to_generate(state: RAGState):
    """
    Determines whether to generate an answer, or stop if no relevant docs.
    """
    print("---DECIDE TO GENERATE---")
    filtered_documents = state.get("documents", [])

    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE IRRELEVANT, SKIP GENERATION---")
        return "end"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state: RAGState):
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["query"]
    documents = state.get("documents", [])
    generation = state["answer"]
    attempts = state.get("generation_attempts", 0)

    # Allow graceful exits
    if "I don't know" in generation:
        print("---DECISION: MODEL DOES NOT KNOW, STOP---")
        return "useful"
    
    if attempts >= 2:
        print("---DECISION: TOO MANY ATTEMPTS, FORCE STOP TO AVOID LOOP---")
        return "useful"

    # 1. Check hallucination
    score = hallucination_grader.invoke(
        {"documents": "\n".join(documents), "generation": generation}
    )
    if score.binary_score.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        # 2. Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score.lower() == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not_useful"
    else:
        print("---DECISION: GENERATION IS HALLUCINATED, RETRY---")
        return "not_supported"

# Build Graph
graph = StateGraph(RAGState)

graph.add_node("retrieve", retriever_agent)
graph.add_node("grade_documents", grade_documents_agent)
graph.add_node("generate", generator_agent)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "end": END
    }
)
graph.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not_supported": "generate", 
        "not_useful": "generate",    
        "useful": END                
    }
)

rag_app = graph.compile()
