from graph.rag_graph import rag_app
import time

if __name__ == "__main__":
    query = input("Ask a question: ")

    start = time.time()
    
    # We initialize the state with generation_attempts at 0
    result = rag_app.invoke({"query": query, "generation_attempts": 0})
    end = time.time()

    # Handling empty state returns
    answer = result.get("answer", "No answer could be generated. This information may not exist in the source documents.")

    print("\n[Final Answer]:\n", answer)
    print(f"\n[Response Time]: {end - start:.3f} seconds")
    