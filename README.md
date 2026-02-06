# LangGraph Multi-Agent RAG System

This repository implements a **multi-agent Retrieval-Augmented Generation (RAG) system**
using **LangGraph**, **FAISS**, and **LLMs**.  
The system is designed to answer questions **strictly based on provided documents** and
respond with **“I don’t know”** when the answer is not present, reducing hallucinations.

---

## 🧠 Architecture Overview

The system is composed of the following stages:

- **Ingestion Pipeline**
  - Loads documents
  - Splits text into semantic chunks
  - Generates embeddings
  - Stores vectors in a FAISS index

- **Retriever Agent**
  - Retrieves top-K relevant document chunks

- **Generator Agent**
  - Produces answers **only from retrieved context**
  - Explicitly refuses to answer when information is missing

- **LangGraph Orchestration**
  - Manages agent execution and state transitions

---

## 📂 Project Structure

langgraph-multi-agent-rag/
│
├── agents/
│ ├── **init**.py
│ ├── retriever_agent.py
│ └── generator_agent.py
│
├── graph/
│ ├── **init**.py
│ └── rag_graph.py
│
├── data/
│ ├── sample_docs.txt  
│ ├── eval_queries.txt  
│ └── docs.txt  
│
├── faiss_index/  
│
├── ingest.py
├── main.py
├── requirements.txt
├── .gitignore
├── .env  
├── LICENSE
└── README.md
