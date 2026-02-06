from graph.rag_graph import rag_app
import time

query = input("Ask a question: ")

start = time.time()
result = rag_app.invoke({"query": query})
end = time.time()

print("\nAnswer:\n", result["answer"])
print(f"\nResponse Time: {end - start:.3f} seconds")
