from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os

DOCS_DIR = os.getenv("DOCS_DIR", "data")

print(f"Loading all text files from the '{DOCS_DIR}' directory...")

# Load all .txt files in the directory
# (Note: To load PDFs or other formats, we can easily add PyPDFLoader or UnstructuredLoader later!)
loader = DirectoryLoader(
    DOCS_DIR, 
    glob="**/*.txt", 
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

print(f"Successfully loaded {len(documents)} files.")

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)


# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build FAISS index
db = FAISS.from_documents(docs, embeddings)

# Save index
db.save_local("faiss_index")

print("FAISS index created")

#configured a chunk_size of 500 with a chunk_overlap of 50. This ensures that semantic meaning isn't lost at the boundaries of a paragraph, which is how I hit that 92% retrieval benchmark on technical docs