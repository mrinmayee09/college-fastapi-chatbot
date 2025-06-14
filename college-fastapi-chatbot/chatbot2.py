from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Q&A dataset
df = pd.read_csv("umit_faq_cleaned.csv")
questions = df['question'].tolist()
answers = df['answer'].tolist()

# Load sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Embed all questions
embeddings = model.encode(questions, convert_to_tensor=False)

# Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# QA Retriever
def retrieve_answer(user_query: str, top_k: int = 1) -> str:
    query_embedding = model.encode([user_query])[0]
    distances, indices = index.search(np.array([query_embedding]), k=top_k)
    return answers[indices[0][0]]  # Return top-1 answer

# Define chat endpoint
@app.get("/chat")
async def chat(question: str = Query(..., description="Ask a question about UMIT.")):
    answer = retrieve_answer(question)
    return {"answer": answer}
