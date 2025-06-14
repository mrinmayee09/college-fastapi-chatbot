# Chatbot for student query assistance

This chatbot was developed as part of a mini project to assist students with common queries related to Usha Mittal Institute of Technology (UMIT).

## Description

The system uses FastAPI for the backend and SentenceTransformers with FAISS for retrieving the most relevant answer based on user input. A simple HTML and JavaScript interface is provided for interaction.

## Features

* Fast and accurate semantic search
* Simple web-based chat interface
* Based on a custom Q\&A dataset

## Technologies Used

* FastAPI
* SentenceTransformers (all-MiniLM-L6-v2)
* FAISS
* Pandas, NumPy
* HTML, CSS, JavaScript

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Place `umit_qa_enhanced.csv` in the project folder.
3. Run the backend:

   ```bash
   uvicorn chatbot2:app --reload
   ```
4. Open `index.html` in a web browser.



This project was created as part of a mini project submission.
