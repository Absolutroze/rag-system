# RAG System
A Retrieval-Augmented Generation (RAG) System built with FastAPI, Ollama, and FAISS for efficient document search and retrieval. The project leverages Docker for containerized deployment and supports embedding-based search using FastEmbed. 

# System Architecture Overview
The RAG (Retrieval-Augmented Generation) System is designed to enhance the capabilities of LLMs by integrating LangChain, Ollama (LLaMA 3), FAISS, and FastAPI for efficient document retrieval and API-based interaction. The system operates as follows:
1. Document Ingestion & Processing:
- Documents (PDFs) are processed using pdfplumber.
- Text embeddings are generated using fastembed.
- FAISS is used for efficient document vector storage and retrieval.

2. Query Processing & Retrieval:
- A user query is processed via FastAPI.
- FAISS retrieves relevant document embeddings.
- The LLaMA 3 model (via Ollama) generates a response using the retrieved context.

3. API Exposure:
- FastAPI exposes endpoints for querying the system.
- API documentation is available via Swagger UI (http://127.0.0.1:8000/docs).

# Setup Instructions
a. Clone the Repository
cd C:\Users\desired\location folder
git clone https://github.com/your-username/rag-system.git
cd name of the repository

b. Setup Virtual Environment & Dependencies 
python -m venv venv
venv\Scripts\activate

c. Install Required Dependencies
pip install langchain-community
pip install llama_index
pip install flask
pip install fastembed
pip install pdfplumber
pip install faiss-cpu
pip install numpy
pip install -U langchain langchain-community
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install langchain-ollama

d. Run FASTAPI Server
uvicorn app:app --reload
- API Docs: Swagger UI (http://127.0.0.1:8000/docs#/)
- API Root: http://localhost:8000

e. Docker Setup
- Build and Run Docker Containers
  docker-compose build
  docker-compose up
- Stop containers
  docker-compose down

# API Usage Examples
1. Querying the RAG System
   Send a POST request to /query with a JSON payload: (Example)
   {
  "question": "What is Retrieval-Augmented Generation?"
  }
   
3. Expected Response
   {
  "answer": "Retrieval-Augmented Generation (RAG) ...."
  }
  
# RAG Implementation Choices
- LangChain: Handles orchestration between document retrieval and LLM responses.
- Ollama (LLaMA 3): Provides a powerful open-source language model for generating answers.
- FAISS: A fast and scalable vector search engine for document retrieval.
- FastAPI: A lightweight web framework for serving the RAG model as an API.

# Future Improvements
1. Model Enhancement
- Consider using DeepSeek instead of LLaMA 3 for better domain-specific responses.
- Explore fine-tuning LLaMA 3 for more customized output.

2. Performance Optimization
- Implement GPU acceleration for faster embedding generation and retrieval.
- Optimize FAISS indexing for better search efficiency.

3. Additional Features
- Add multi-modal retrieval (support for images, audio, etc.).
- Integrate user feedback mechanisms to improve model responses.
- Deploy on cloud platforms (AWS/GCP) for better scalability.

# Contributing
Feel free to contribute by creating pull requests or raising issues for enhancements.

# Author
Developed by Yvonn. Reach out via GitHub for discussions and collaborations!

