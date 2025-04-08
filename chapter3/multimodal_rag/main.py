"""
=========================================== 
Multimodal RAG: RAG from scratch
Main script
Marc Haraoui - created on 07/04/2025
=========================================== 
"""

from utils import get_documents
from rag_pipeline import MultimodalRAGPipeline
from transformers.image_utils import load_image
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Pipeline Arguments')
    parser.add_argument('--query',
                        help    = 'User''s text query',
                        required=True)
    parser.add_argument('--image_path',
                        help    = 'Visual data paths (PDF or image, one or multiple).',
                        nargs   = '+',
                        default = None)
    parser.add_argument('--top_k',
                        help    = 'Number of documents to retrieve (Top K is at 3 by default).',
                        type   = int,
                        default = 3)
    parser.add_argument('--embedder',
                        help    = 'Choose the embedding model you wish to use in the RAG Pipeline. Default: "google/siglip2-base-patch16-naflex".',
                        default = "google/siglip2-base-patch16-naflex")
    parser.add_argument('--generator',  
                        help    = 'Choose the LLM you wish to use as the generator in the RAG. Default: "HuggingFaceTB/SmolLM2-360M-Instruct".',
                        default = "HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument('--doc_path',
                        help    = 'Path of stored text-based documents. Default: "documents"',
                        default = "documents")
    args = parser.parse_args()


    # Setup the multimodal RAG pipeline
    rag = MultimodalRAGPipeline(embedding_model=args.embedder, generator_model=args.generator)
    
    # Setup documents 
    documents = get_documents(doc_path=args.doc_path)

    # Add list of documents into the pipeline
    rag.add_documents(documents)

    # Save the knowledge base
    rag.save_knowledge_base("rag_knowledge_base")

    # Query setup
    if args.image_path:
        images = [load_image(img_path) for img_path in args.image_path]
        query = {
            "text": args.query,
            "image": images
        }
    else:
        query = {"text": args.query}

    # Document retrieval
    modality = "image-text" if args.image_path else "text"
    context = rag.retrieve_context(query, modality=modality, top_k=args.top_k)

    print("Retrieved documents: ", context)

    response = rag.generate_response(query, context)

    print("User: ", args.query)
    print("Assistant: ", response)