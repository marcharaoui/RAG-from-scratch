"""
=========================================== 
Text-based RAG: RAG from scratch
Main script
Marc Haraoui - created on 09/03/2025
=========================================== 
"""

import argparse
from utils import get_documents
from rag_pipeline import RAGPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAGPipelineArguments')
    parser.add_argument('--query',
                        help    = 'User''s query',
                        type    = str,
                        default=None)
    parser.add_argument('--embedder',
                        help    = 'Choose the embedding model you wish to use in the RAG Pipeline. Default: ""HuggingFaceTB/SmolLM2-360M-Instruct""',
                        type    = str,
                        default = "BAAI/llm-embedder")
    parser.add_argument('--generator',  
                        help    = 'Choose the LLM you wish to use as the generator in the RAG. Default: ""HuggingFaceTB/SmolLM2-360M-Instruct""',
                        type    = str,
                        default = "HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument('--doc_path',
                        help    = 'Path of stored text-based documents. Default: "documents"',
                        type    = str,
                        default = "documents")
    args = parser.parse_args()

    # Setup documents 
    documents = get_documents(doc_path=args.doc_path)

    # Setup RAG pipeline
    rag = RAGPipeline(embedding_model=args.embedder, generator_model=args.generator)
    
    # Now let's create a new knowledge base with two sample documents.
    rag.create_knowledge_base(documents, chunking_method="recursive", chunk_size=256, overlap=20)

    query = "What is the battery life of the AlphaTech SmartWatch X100, and does it support fast charging?" if args.query is None else args.query

    context = rag.similarity_search(query, method="semantic", top_k=3)

    print("Semantic Search Results (AI):", context)

    response = rag.generate_response(query, context)

    print("User: ", query)
    print("Assistant: ", response)