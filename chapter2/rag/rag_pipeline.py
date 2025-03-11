"""
=========================================== 
Text-based RAG: RAG from scratch
Pipeline script - Easily use this class in your project 
Marc Haraoui - created on 09/03/2025
=========================================== 
"""


from typing import List, Dict, Optional
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from transformers import pipeline
from datetime import date
import numpy as np
import json, os


class RAGPipeline:
    """
    A fully modular and flexible text-based RAG pipeline.

    Core functionality:
    1. Chunking: Splitting documents into retrievable pieces.
    2. Embedding: Converting text chunks into vector representations.
    3. Indexing: Storing embeddings for efficient retrieval.
    4. Retrieval: Fetching relevant chunks based on query similarity.
    5. Generation: Using retrieved context to generate responses.

    Key Features:
      * Supports multiple chunking strategies.
      * Offers keyword-based, semantic, and hybrid search.
      * Integrated with Hugging Face models for embedding and generation.
      * Supports local storage of knowledge base (JSON) and embeddings (NumPy).
      * Easily extendable to add new documents.
    """


    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-l6-v2",
                 generator_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model (str): Pretrained text embedding name.
            generator_model (str): Pretrained text generation model name.
        """
        # Setup embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model      = None # Initially set embedding_model to None; it will be loaded when creating/loading a KB.

        # Load text generation model
        self.generator = pipeline(
            "text-generation",
            model=generator_model,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

        # Storage for indexed data
        self.index          = [] # Stores chunk metadata
        self.knowledge_base = {} # Stores document metadata
        self.embeddings     = np.array([])
        self.bm25           = None


    def chunk_text(self,
                   text: str,
                   doc_name: str,
                   doc_id: str,
                   method: str = "recursive",
                   chunk_size: int = 500,
                   overlap: int = 50,
                   markdown_headers: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Splits text into smaller chunks and adds metadata to each chunk.

        Args:
            text (str): Document content.
            doc_name (str): Name of the document.
            doc_id (str): Unique document identifier.
            method (str): Chunking strategy: "fixed", "recursive", or "document".
            chunk_size (int): Size of chunks.
            overlap (int): Overlapping characters between chunks.
            markdown_headers (Optional[List[Dict[str, str]]]): Headers for Markdown/HTML chunking.

        Returns:
            List[Dict[str, str]]: List of chunks with metadata.
        """
        if method == "fixed":
          splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        elif method == "recursive":
          splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        elif method == "document":
          if markdown_headers:
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
          else:
            splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        else:
          raise ValueError("Unsupported chunking method. Choose from 'fixed', 'recursive', and 'document'.")

        chunks = splitter.split_text(text)
        return [{"id": f"{doc_id}_{i}", "text": chunk} for i, chunk in enumerate(chunks)]


    def create_knowledge_base(self,
                              documents: List[Dict[str, str]],
                              knowledge_base_path: str = f"knowledge_base_{date.today().strftime('%Y_%m_%d')}.json",
                              embeddings_path: str = f"text_embeddings_{date.today().strftime('%Y_%m_%d')}.npy",
                              chunking_method: str = "recursive",
                              chunk_size: int = 500,
                              overlap: int = 50,
                              markdown_headers: Optional[List[Dict[str, str]]] = None):
        """
        Creates a new knowledge base by processing and indexing documents.

        Args:
            documents (List[Dict[str, str]]): List of documents (each with "id", "name", and "text").
            knowledge_base_path (str): Path to save the knowledge base as JSON.
            embeddings_path (str): Path to save the embeddings as NumPy array.
            chunking_method (str): Chunking strategy.
            chunk_size (int): Size of chunks.
            overlap (int): Overlapping characters between chunks.
        """
        # Initialize knowledge base structure
        self.knowledge_base = {
            "embedding_model_name": self.embedding_model_name,
            "documents": []
        }
        self.index  = []
        chunk_texts = []

        for doc in documents:
            doc_id   = doc["id"]
            doc_name = doc["name"]
            chunks   = self.chunk_text(doc["text"], doc_name=doc_name, doc_id=doc_id, method=chunking_method,
                                       chunk_size=chunk_size, overlap=overlap, markdown_headers=markdown_headers)
            self.knowledge_base["documents"].append({
                "id": doc_id,
                "name": doc_name,
                "chunking_technique": chunking_method,
                "chunks": chunks
            })
            self.index.extend(chunks)
            chunk_texts.extend([chunk["text"] for chunk in chunks])

        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # Compute text embeddings
        self.embeddings = np.array(self.embedding_model.embed_documents(chunk_texts))

        # Store BM25 index for hybrid retrieval
        tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
        self.bm25 = BM25Okapi(tokenized_chunks)

        # Save knowledge base
        self.save_knowledge_base(knowledge_base_path=knowledge_base_path, embeddings_path=embeddings_path)


    def save_knowledge_base(self,
                            knowledge_base_path: str,
                            embeddings_path: str):
        """
        Saves the knowledge base and embeddings locally.

        Args:
            knowledge_base_path (str): Path to the knowledge base JSON file.
            embeddings_path (str): Path to the embeddings NumPy file.
        """
        with open(knowledge_base_path, "w") as f:
            json.dump(self.knowledge_base, f, indent=4)
        np.save(embeddings_path, self.embeddings)

        print("Knowledge base and embeddings saved successfully.")


    def load_knowledge_base(self,
                            knowledge_base_path: str,
                            embeddings_path: str):
        """
        Load knowledge base and embeddings from disk.

        Args:
            knowledge_base_path (str): Path to the knowledge base JSON file.
            embeddings_path (str): Path to the embeddings numpy file.
        """
        with open(knowledge_base_path, "r") as f:
            self.knowledge_base = json.load(f)
        self.embeddings = np.load(embeddings_path)

        # Load embedding model
        stored_model_name         = self.knowledge_base["embedding_model_name"]
        self.embedding_model_name = stored_model_name
        self.embedding_model      = HuggingFaceEmbeddings(model_name=stored_model_name)

        # Rebuild self.index from the loaded knowledge base documents
        self.index = []
        for doc in self.knowledge_base.get("documents", []):
            self.index.extend(doc.get("chunks", []))

        # Rebuild BM25 index from all chunk texts
        all_texts = [chunk["text"] for chunk in self.index]
        tokenized_all = [text.lower().split() for text in all_texts]
        self.bm25 = BM25Okapi(tokenized_all)

        print("Knowledge base and embeddings loaded successfully.")


    def add_documents(self,
                      documents: List[Dict[str, str]],
                      knowledge_base_path: str = f"knowledge_base_{date.today().strftime('%Y_%m_%d')}.json",
                      embeddings_path: str = f"text_embeddings_{date.today().strftime('%Y_%m_%d')}.npy",
                      chunking_method: str = "recursive",
                      chunk_size: int = 500,
                      overlap: int = 50,
                      markdown_headers: Optional[List[Dict[str, str]]] = None):
        """
        Adds additional documents to an existing knowledge base.

        Args:
            documents (List[Dict[str, str]]): List of documents (each with "id", "name", and "text").
            knowledge_base_path (str): Path to save the knowledge base as JSON.
            embeddings_path (str): Path to save the embeddings as NumPy array.
            chunking_method (str): Chunking strategy.
            chunk_size (int): Size of chunks.
            overlap (int): Overlapping characters between chunks.
        """
        # Collect new chunks and update knowledge base
        new_chunks      = []
        new_chunk_texts = []

        for doc in documents:
            doc_id   = doc["id"]
            doc_name = doc["name"]
            chunks   = self.chunk_text(doc["text"], doc_name, doc_id, method=chunking_method,
                                     chunk_size=chunk_size, overlap=overlap, markdown_headers=markdown_headers)
            self.knowledge_base["documents"].append({
                "id": doc_id,
                "name": doc_name,
                "chunking_technique": chunking_method,
                "chunks": chunks
            })
            new_chunks.extend(chunks)
            new_chunk_texts.extend([chunk["text"] for chunk in chunks])
            self.index.extend(chunks)

        # Compute embeddings for new chunks
        new_embeddings = np.array(self.embedding_model.embed_documents(new_chunk_texts))
        if self.embeddings.size:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        else:
            self.embeddings = new_embeddings

        # Update BM25 index with new chunks
        all_tokenized = [chunk.lower().split() for chunk in new_chunk_texts]
        if self.bm25 is None:
            self.bm25 = BM25Okapi(all_tokenized)
        else:
            # BM25Okapi doesn't support incremental updates directly; recreate with existing + new tokens.
            all_texts = [doc["text"] for doc in self.index]
            self.bm25 = BM25Okapi([text.split() for text in all_texts])

        # Save the updated knowledge base and embeddings
        self.save_knowledge_base(knowledge_base_path, embeddings_path)


    def semantic_search(self,
                        query: str,
                        top_k: int = 3) -> List[str]:
        """
        Performs semantic search using cosine similarity.

        Args:
            query (str): User input query.
            top_k (int): Number of relevant chunks to retrieve.

        Returns:
            List[str]: Top-k relevant text chunks.
        """
        query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.index[i]["text"] for i in top_indices]


    def keyword_search(self,
                       query: str,
                       top_k: int = 3) -> List[str]:
        """
        Performs keyword-based search using BM25 ranking.

        Args:
            query (str): User input query.
            top_k (int): Number of relevant chunks to retrieve.

        Returns:
            List[str]: Top-k relevant text chunks.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.index[i]["text"] for i in top_indices]


    def similarity_search(self,
                        query: str,
                        method: str = "semantic",
                        top_k: int = 3) -> List[str]:
        """
        Retrieves relevant chunks using the chosen search method.

        Args:
            query (str): User input query.
            method (str): Retrieval method: "semantic", "keyword", or "hybrid".
            top_k (int): Number of relevant chunks to retrieve.

        Returns:
            List[str]: Top-k relevant text chunks.
        """
        if method == "semantic":
            return self.semantic_search(query, top_k)

        elif method == "keyword":
            return self.keyword_search(query, top_k)

        elif method == "hybrid":
            semantic_results = self.semantic_search(query, top_k)
            keyword_results  = self.keyword_search(query, top_k)
            combined_results = list(set(semantic_results + keyword_results))[:top_k]
            return combined_results

        else:
            raise ValueError("Invalid retrieval method. Choose 'semantic', 'keyword', or 'hybrid'.")


    def generate_response(self,
                          query: str,
                          retrieved_chunks: List[str],
                          instructions: str = "You are a chat assistant who answers briefly using only the provided context.") -> str:
        """
        Generates a response using the retrieved chunks as context.

        Args:
            query (str): User's query.
            retrieved_chunks (List[str]): Retrieved text chunks to be used as context.
            instructions (str): System instructions for the generator.

        Returns:
            str: Generated response from the LLM.
        """
        # Combine the retrieved chunks into a single context string
        context = " ".join(retrieved_chunks)

        # Construct the prompt for the language model
        prompt = f"Context: {context}\n\nQuestion: {query}"

        # Prepare the augmented prompt
        augmented_prompt = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
        # Generate the response
        response = self.generator(augmented_prompt, max_length=200, truncation=True)
        # return response[0]["generated_text"]
        return response[0]["generated_text"][-1]["content"]