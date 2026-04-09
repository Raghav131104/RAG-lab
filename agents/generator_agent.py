from dotenv import load_dotenv
import os

load_dotenv() 

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generator_agent(state):
    print("---GENERATING ANSWER---")
    
    context = "\n-----\n".join([f"Source {i+1}: {doc}" for i, doc in enumerate(state.get("documents", []))])
    query = state["query"]
    generation_attempts = state.get("generation_attempts", 0)

    if not state.get("documents"):
        return {
            "answer": "I don't know the answer because I could not find relevant information in the provided internal documents.", 
            "generation_attempts": generation_attempts + 1
        }

    prompt = f"""
You are an expert internal AI assistant.
Answer the question using ONLY the provided context below.
- If the context provides information for only part of the question, answer what you know and explicitly state what you cannot answer.
- If the answer is completely not present in the context, strictly say "I don't know."

When answering, you MUST cite the relevant sources (e.g., "[Source 1]", "[Source 2]") to prove where you found the information.


Context:
{context}

Question:
{query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"answer": response.content, "generation_attempts": generation_attempts + 1}
