"""
=========================================== 
Multimodal RAG: RAG from scratch
utils script - useful function(s) to use with the RAG pipeline  
Marc Haraoui - created on 07/04/2025
=========================================== 
"""

from transformers.image_utils import load_image
import os


def get_documents(doc_path:str='documents'):
    """
    Retrieves all documents from the specified directory and stores them in a list of dictionaries.

    Args:
        doc_path (str): Path to the directory containing multimodal files.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - "id": A unique identifier for the document (starting from 1).
              - "type": The name of the .txt file (without extension).
              - "content": The content of the document.
    """
    documents = []

    try:
        # List all files in the given directory
        allowed_extensions = ('.txt', '.pdf', '.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(doc_path) if f.endswith(allowed_extensions)]
        
        # Iterate through each .txt file
        for idx, file_name in enumerate(files, start=1):
            file_path = os.path.join(doc_path, file_name)

            # Text document
            if file_path.endswith('.txt'):
                # Read the content of the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add document details to the list
                documents.append({
                    "id": idx,
                    "type": "text" ,
                    "content": content
                })
            
            # Image document
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                # Handle image files
                try:
                    img = load_image(file_path)
                    documents.append({
                        "id": idx,
                        "type": "image",
                        "content": img
                    })
                except Exception as e:
                    print(f"Error loading image {file_name}: {e}")

            # PDF document
            elif file_path.endswith('.pdf'):
                # Handle PDF files
                documents.append({
                    "id": idx,
                    "type": "pdf",
                    "content": file_path
                })
    
    except FileNotFoundError:
        print(f"Error: The directory '{doc_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")     

    return documents

