# Chapter 3: Multimodal RAG

This code is a functional multimodal RAG pipeline to use by anyone.

## Installations

Start off by installing the libraries used for this project.

```bash
pip install -r requiremments.in
pip install git+https://github.com/huggingface/transformers.git 
```

You can also install a vector database library, such as FAISS, Qdrant, or Milvus, if you wish to store and search embeddings directly in a vector DB. In this project, we store them locally.

## Usage

### General steps

The pipeline follows the following steps:
1. Prepare and format multimodal documents (text, image, pdf).
2. Index documents using a multimodal embedding model (default: siglip 2).
3. Retrieve top-k documents using semantic search.
4. Generate responses based on retrieved documents using a vision-language model (default: smolvlm 500M).

### Documents structuring

All your documents should be placed in the documents folder, in order to create the knowledge base:

```python
documents = get_documents(doc_path='documents')
```

### Use the RAGPipeline class

Create your own scripts that use MultimodalRAGPipeline as needed. Here is an example:

```python
form utils import get_documents
from rag_pipeline import MultimodalRAGPipeline

# Setup RAG pipeline
rag = MultimodalRAGPipeline(embedding_model, generator_model)

# Setup documents 
documents = get_documents(doc_path)

# Add list of documents into the pipeline
rag.add_documents(documents)

# Save the knowledge base
rag.save_knowledge_base("rag_knowledge_base")

# Query setup
if image_paths:
    images = [load_image(img_path) for img_path in image_paths]
    query = {
        "text": text_query,
        "image": images
    }
else:
    query = {"text": text_query}

# Document retrieval
modality = "image-text" if image_paths else "text"
context = rag.retrieve_context(query, modality, top_k)

print("Retrieved documents: ", context)

response = rag.generate_response(query, context)

print("User: ", text_query)
print("Assistant: ", response)
```

### Run locally using main.py

Use this pre-built script to have a functional Multimodal RAG pipeline.

Example of usage:

```bash
python main.py --query Explain these images --image_path path/image1.png path/image2.jpg --top_k 5
```
