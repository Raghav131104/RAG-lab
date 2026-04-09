from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

def retriever_agent(state):
    query = state["query"]
    docs = vectorstore.similarity_search(query, k=25)

    print("\nRetrieved docs:")
    for d in docs:
        print("-", d.page_content[:80])

    return {"documents": [doc.page_content for doc in docs]}
