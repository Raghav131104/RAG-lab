from dotenv import load_dotenv
import os

load_dotenv() 

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generator_agent(state):
    context = "\n".join(state["documents"])
    query = state["query"]

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"answer": response.content}

# used structured retrieval patterns by constraining the LLM's prompt. In my code, I explicitly tell the model: 'Answer using ONLY the context below. If the answer is not present, say I don't know.


