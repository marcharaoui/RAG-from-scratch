# Chapter 2: The Technical Foundations of Text-Only RAG

This code is a functional text-based RAG pipeline to use by anyone. It is fully manageable and modular, allowing you to choose between:
- Different chunking 
- Similarity search
- Open source embedding models on Hugging Face
- Open source LLMs on Hugging Face

## Installations

Start off by installing the libraries used for this project.

```bash
pip install transformers langchain-text-splitters langchain-huggingface scikit-learn rank-bm25
```

You also install a vector database library, such as FAISS, Qdrant, or Milvus, if you wish to store and search embeddings directly in a vector DB. In this project, we store them locally.

## Usage

### General steps

The pipeline follows the following steps:
1. Choose a chunking method (fixed, recursive, document for structured formats).
2. Index documents using your preferred chunking strategy (traditional, late).
3. Select a search technique (semantic, keyword, hybrid) for optimized search.
4. Generate responses based on retrieved chunks.

### Documents structuring

All your text documents should be loaded following the following structuring:

```python
documents = [
    {"id": "doc1", "name": "doc1.txt", "text": doc1_content},
    {"id": "doc2", "name": "html_ai_doc2.txt", "text": doc2_content},
]
```

### Use the RAGPipeline class

Create your own scripts that use RAGPipeline as needed. 

```python
form utils import get_documents
from rag_pipeline import RAGPipeline

# Setup RAG pipeline
rag = RAGPipeline(embedding_model=args.embedder, generator_model=args.generator)

# Setup documents
documents = get_documents(doc_path=args.doc_path)

# Create a new knowledge base with two sample documents.
rag.create_knowledge_base(documents, chunking_method="recursive", chunk_size=150, overlap=15)

query = "How does product A work?"

# Similarity search using hybrid method
context = rag.similarity_search(query, method="hybrid", top_k=3)

# Generate final response using query and context
response = rag.generate_response(query, context)

print("User: ", query)
print("Assistant: ", response)
```

### Run locally using main.py

Use this pre-built script to have a functional RAG pipeline. Replace with your custom text documents in the 'documents' folder.

Example of usage:

```bash
python main.py --query Explain how product A works --embedder EuroBERT/EuroBERT-210m --generator HuggingFaceTB/SmolLM2-1.7B-Instruct
```
