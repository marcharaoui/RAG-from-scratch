"""
=========================================== 
Multimodal RAG: RAG from scratch
Pipeline script - Easily use this class in your project 
Marc Haraoui - created on 07/04/2025
=========================================== 
"""


from datetime import date
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoProcessor, AutoModel, AutoModelForVision2Seq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import pymupdf
import os
import json


class MultimodalRAGPipeline:
    """
    Multimodal RAG pipeline that supports text, images and PDFs.
    It enables advanced semantic retrieval and multimodal generative responses.
    """

    def __init__(self,
                  embedding_model: str = "google/siglip2-base-patch16-naflex",
                  generator_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct"):
        """ Initialize the RAG pipeline."""
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load multimodal embedding model
        self.embedding_processor = AutoProcessor.from_pretrained(embedding_model)
        self.embedding_model = AutoModel.from_pretrained(embedding_model).to(self.device)

        # Load the vision-language model
        self.generator_processor = AutoProcessor.from_pretrained(generator_model)
        self.generator_model = AutoModelForVision2Seq.from_pretrained(
            generator_model,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        # Initialize knowledge base
        self.knowledge_base = []
        self.embeddings = np.array([])


    def embed_multimodal(self, data, modality: str) -> np.ndarray:
        """ Generate embeddings based on modality."""
        with torch.no_grad():
            if modality == "text":
                inputs    = self.embedding_processor(text=[data], return_tensors="pt").to(self.embedding_model.device)
                embedding = self.embedding_model.get_text_features(**inputs)
            elif modality == "image":
                inputs    = self.embedding_processor(images=[data], return_tensors="pt").to(self.embedding_model.device)
                embedding = self.embedding_model.get_image_features(**inputs)
            else:
                raise ValueError(f"Unsupported modality: {modality}.")
        return embedding.cpu().numpy().flatten()


    def chunk_text(self, text, chunk_size=500, overlap=50):
        """ Recursively split text into chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text)


    def process_pdf(self, pdf_path):
        """ Convert PDF into page-wise image chunks."""
        doc = pymupdf.open(pdf_path)
        pdf_chunks = []
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf_chunks.append({
                "type": "image",
                "content": img,
                "metadata": {"page": page_num, "source": pdf_path}
            })
        return pdf_chunks


    def save_knowledge_base(self, dir_path="knowledge_base_folder"):
        """ Save knowledge base and embeddings."""
        os.makedirs(dir_path, exist_ok=True)
        kb_path = os.path.join(dir_path, f"kb_{date.today()}.json")
        embeddings_path = os.path.join(dir_path, f"embeddings_{date.today()}.npy")

        kb_serializable = []
        for item in self.knowledge_base:
            data = {
                "type": item["type"],
                "id": item["id"],
                "chunk_id": item["chunk_id"],
            }
            if item["type"] == "text":
                data["content"] = item["content"]
            elif item["type"] == "image":
                image_path = os.path.join(dir_path, f"{item['id']}_{item['chunk_id']}.png")
                item["content"].save(image_path)
                data["content"] = image_path
            kb_serializable.append(data)

        with open(kb_path, "w") as f:
            json.dump(kb_serializable, f, indent=2)
        np.save(embeddings_path, self.embeddings)


    def load_knowledge_base(self, kb_path, embeddings_path):
        """ Load knowledge base and embeddings."""
        with open(kb_path, "r") as f:
            kb_data = json.load(f)

        self.knowledge_base = []
        for item in kb_data:
            if item["type"] == "image":
                item["content"] = Image.open(item["content"])
            self.knowledge_base.append(item)

        self.embeddings = np.load(embeddings_path)


    def add_documents(self, docs):
        """ Add documents to knowledge base and update embeddings."""
        new_embeddings = []
        for doc in docs:
            doc_id = doc["id"]
            modality = doc["type"]

            if modality == "text":
                text_chunks = self.chunk_text(doc["content"])
                for idx, chunk in enumerate(text_chunks):
                    embedding = self.embed_multimodal(chunk, "text")
                    self.knowledge_base.append({
                        "type": "text", "id": doc_id, "chunk_id": idx, "content": chunk
                    })
                    new_embeddings.append(embedding)

            elif modality == "image":
                embedding = self.embed_multimodal(doc["content"], "image")
                self.knowledge_base.append({
                    "type": "image", "id": doc_id, "chunk_id": 0, "content": doc["content"]
                })
                new_embeddings.append(embedding)

            elif modality == "pdf":
                chunks = self.process_pdf(doc["content"])
                for chunk in chunks:
                    embedding = self.embed_multimodal(chunk["content"], "image")
                    self.knowledge_base.append({
                        "type": "image",
                        "id": doc_id,
                        "chunk_id": chunk["metadata"]["page"],
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"]
                    })
                    new_embeddings.append(embedding)
            else:
                raise ValueError("Unsupported document type: text, image, pdf")

        self.embeddings = np.vstack([self.embeddings, new_embeddings]) if self.embeddings.size else np.array(new_embeddings)


    def retrieve_context(self, query, modality, top_k=3):
        """ Perform semantic search and retrieve context."""
        if modality == "text":
            query_emb = self.embed_multimodal(query["text"], "text")
        elif modality == "image":
            query_emb = self.embed_multimodal(query["image"], "image")
        elif modality == "image-text":
            text_emb = self.embed_multimodal(query["text"], "text")
            img_emb = self.embed_multimodal(query["image"], "image")
            query_emb = (text_emb + img_emb) / 2
        else:
            raise ValueError("Unsupported query modality")

        similarities = cosine_similarity(query_emb.reshape(1, -1), self.embeddings)[0]
        top_idxs = similarities.argsort()[::-1][:top_k]

        return [self.knowledge_base[i] for i in top_idxs]


    def generate_response(self, query, retrieved_context, instructions="Answer using the provided multimodal context."):
        """ Generate response using retrieved context and a VLM."""
        context_text = ""
        images       = []

        for chunk in retrieved_context:
            if chunk["type"] == "text":
                context_text += f"Text: {chunk['content']}\n"
            elif chunk["type"] == "image":
                # context_text += "<image>\n"
                images.append(chunk["content"])

        # Construct the augmented prompt with placeholders
        augmented_prompt = f"{instructions}\n\nContext:\n{context_text}\n\nQuery: {query['text']}\n\nAnswer:"

        # Add query image if exists
        if 'image' in query:
            if type(query['image']) == list:
                images.extend(query['image'])
            else:
                images.append(query['image'])

        if images:
            messages = [{
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": augmented_prompt}]
            }]

            # Prepare inputs
            prompt = self.generator_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.generator_processor(text=prompt, images=images, return_tensors="pt").to(self.device)

        else:
            messages = [{
                  "role": "user",
                  "content": [{"type": "text", "text": augmented_prompt}]
              }]

            # Prepare inputs
            prompt = self.generator_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.generator_processor(text=prompt, return_tensors="pt").to(self.device)


        # Generate outputs
        generated_ids = self.generator_model.generate(**inputs, max_new_tokens=500)
        response_text = self.generator_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return response_text