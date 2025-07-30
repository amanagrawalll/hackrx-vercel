# api/index.py

import os
import requests
import numpy as np
import faiss
import groq
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --- Global Objects (loaded once) ---
# This is a key optimization for serverless functions
print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# Initialize the Groq client, getting the key from environment variables
try:
    client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("Groq client initialized.")
except Exception as e:
    client = None
    print(f"Failed to initialize Groq client: {e}")

# --- Pydantic Models for Request/Response ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Initialize FastAPI App ---
app = FastAPI()

# --- Helper Functions (logic from your notebook) ---
def process_document_from_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        
        chunk_size = 1500
        chunk_overlap = 200
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        
        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        print(f"Error processing document: {e}")
        return []

def create_vector_store(chunks: list):
    if not chunks: return None
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def generate_answer_with_groq(question: str, context: str):
    if not client: return "Groq client not initialized."
    prompt = f"""
    Answer the following question based ONLY on the provided context. If the answer is not in the context, say "Answer not found in the provided context."
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:
    """
    try:
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return "Error generating answer from Groq API."

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    # This endpoint is simplified for deployment.
    # You should add the Bearer token authentication as per the hackathon rules.
    
    chunks = process_document_from_url(request.documents)
    if not chunks: raise HTTPException(status_code=500, detail="Failed to process document.")

    vector_index = create_vector_store(chunks)
    if not vector_index: raise HTTPException(status_code=500, detail="Failed to create vector store.")

    all_answers = []
    for question in request.questions:
        question_embedding = embedding_model.encode([question])
        k = 5
        _, indices = vector_index.search(np.array(question_embedding).astype('float32'), k)
        retrieved_context = "\n\n---\n\n".join([chunks[i] for i in indices[0]])
        answer = generate_answer_with_groq(question, retrieved_context)
        all_answers.append(answer)
        
    return HackRxResponse(answers=all_answers)

# Optional: Add a root endpoint for simple health checks
@app.get("/")
def read_root():
    return {"status": "API is running"}