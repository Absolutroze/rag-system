import os
import faiss
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable

app = FastAPI()

# Folder paths
DB_FOLDER = "db"
PDF_FOLDER = "pdf"

# Ensure folders exist
os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# Load LLM model
cached_llm = OllamaLLM(model="llama3")

# Embeddings system
embedding = FastEmbedEmbeddings()

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Custom Prompt Template
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information, say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# Query Request Model
class QueryRequest(BaseModel):
    query: str

from langchain_core.runnables import Runnable

class FAISSRetriever(Runnable):
    def __init__(self, faiss_index, documents, config=None):
        self.index = faiss_index
        self.documents = documents
        self.config = config or {}

    def with_config(self, **config):
        self.config.update(config)
        return self

    def retrieve(self, query, k=5):
        k = self.config.get("k", k)
        
        # Embedding the query
        query_embedding = np.array([embedding.embed_query(query)]).astype(np.float32)
        
        # Perform search in FAISS index
        D, I = self.index.search(query_embedding, k)
        
        # Retrieve documents
        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "page_number": doc.metadata.get("page", "Unknown"),
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown")
                })
        return results

    def invoke(self, inputs: dict) -> dict:
        # Ensure the input is a dictionary with "input" key
        query = inputs.get("input")
        if not query:
            raise ValueError("No query found in inputs")
        
        retrieved_docs = self.retrieve(query)
        return {"retrieved_docs": retrieved_docs}

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

# 📌 Upload PDF and Store in FAISS
@app.post("/pdf")
async def pdf(file: UploadFile = File(...)):
    file_name = file.filename
    save_path = os.path.join(PDF_FOLDER, file_name)

    # Save the PDF file
    with open(save_path, "wb") as f:
        f.write(file.file.read())

    print(f"📄 Saved PDF: {file_name}")

    # Load the document
    loader = PDFPlumberLoader(save_path)
    docs = loader.load_and_split()
    print(f"📜 Loaded {len(docs)} pages.")

    # Split text into chunks
    chunks = text_splitter.split_documents(docs)
    print(f"📝 Split into {len(chunks)} text chunks.")

    # Generate embeddings
    doc_embeddings = np.array([embedding.embed_documents([chunk.page_content]) for chunk in chunks])
    doc_embeddings = doc_embeddings.reshape(-1, doc_embeddings.shape[-1])

    # Initialize FAISS index
    dim = doc_embeddings.shape[1]  # Dimension of embeddings
    faiss_index = faiss.IndexFlatL2(dim)  # L2 distance metric
    faiss_index.add(doc_embeddings.astype(np.float32))  # Add embeddings

    # Save FAISS index
    index_path = os.path.join(DB_FOLDER, "faiss_index.index")
    faiss.write_index(faiss_index, index_path)

    # Save documents for later retrieval
    np.save(os.path.join(DB_FOLDER, "documents.npy"), chunks)

    return {
        "status": "✅ PDF Uploaded & Embedded",
        "filename": file_name,
        "pages": len(docs),
        "chunks": len(chunks)
    }

# Load FAISS index and documents
def load_faiss_index_and_docs():
    index_path = os.path.join(DB_FOLDER, "faiss_index.index")
    faiss_index = faiss.read_index(index_path)
    documents = np.load(os.path.join(DB_FOLDER, "documents.npy"), allow_pickle=True)
    return faiss_index, documents

faiss_index, documents = load_faiss_index_and_docs()
retriever = FAISSRetriever(faiss_index, documents)

@app.post("/query")
async def query(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Retrieve documents
    retrieved_docs = retriever.invoke({"input": query})["retrieved_docs"]

    if not retrieved_docs:
        return {"answer": "No relevant information found in the provided documents."}

    # Combine documents to form the context
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    # Generate answer using LLM
    answer = cached_llm.generate(
        prompts=[raw_prompt.format(input=query, context=context)]
    )

    return {
        "query": query,
        "answer": answer,
        "citations": retrieved_docs
    }