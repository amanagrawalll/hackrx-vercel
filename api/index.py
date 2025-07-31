# api/index.py

import os
import requests
import numpy as np
import groq
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from io import BytesIO
from pypdf import PdfReader
from openai import OpenAI

# --- Global Clients (loaded once) ---
# Initialize the Groq and OpenAI clients from environment variables
try:
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Clients initialized successfully.")
except Exception as e:
    groq_client = None
    openai_client = None
    print(f"Failed to initialize clients: {e}")

# --- Pydantic Models for Request/Response ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Initialize FastAPI App ---
app = FastAPI()

# --- Helper Functions ---
def process_document_from_url(url: str):
    """Downloads and chunks the document."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        
        chunk_size = 2000  # Increased chunk size as we are not using a local model
        chunk_overlap = 300
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

def get_embedding(text: str, model="text-embedding-3-small"):
   """Generates an embedding for a given text using OpenAI's API."""
   text = text.replace("\n", " ")
   return openai_client.embeddings.create(input=[text], model=model).data[0].embedding

def generate_answer_with_groq(question: str, context: str):
    """Generates an answer using Groq."""
    if not groq_client: return "Groq client not initialized."
    prompt = f"""
    Answer the following question based ONLY on the provided context. If the answer is not in the context, say "Answer not found in the provided context."
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return "Error generating answer from Groq API."

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest):
    if not openai_client or not groq_client:
        raise HTTPException(status_code=500, detail="API clients not initialized. Check environment variables.")

    chunks = process_document_from_url(request.documents)
    if not chunks: raise HTTPException(status_code=500, detail="Failed to process document.")

    # Create embeddings for all chunks via API
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    all_answers = []
    for question in request.questions:
        question_embedding = get_embedding(question)

        # Perform simple semantic search (cosine similarity)
        similarities = [np.dot(question_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
        
        # Get top 5 most relevant chunks
        top_k = 5
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        retrieved_context = "\n\n---\n\n".join([chunks[i] for i in top_indices])
        
        answer = generate_answer_with_groq(question, retrieved_context)
        all_answers.append(answer)
        
    return HackRxResponse(answers=all_answers)

@app.get("/")
def read_root():
    return {"status": "API is running"}