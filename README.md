# Admission Query Chatbot

This is a simple chatbot built with **FastAPI** and **Falcon-7B-Instruct**, designed to answer admission-related queries using a combination of keyword-based matching and LLM-generated responses.

## Features

- Keyword-based responses for common admission questions  
- Integration with Falcon-7B-Instruct via HuggingFace Transformers  
- FastAPI backend with CORS enabled  
- Contextual responses generated using a local knowledge base (`data.txt`)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

Make sure you’re using Python 3.8+ and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Prepare the knowledge base

Add your `data.txt` file in the root directory. This file should contain text entries (FAQs or reference info) separated by new lines.

### 4. Run the server

```bash
uvicorn main:app --reload
```

This will start the FastAPI server at `http://localhost:8000`.

### 5. Test the chatbot

Send a GET request to the `/chat` endpoint with your question:

```
http://localhost:8000/chat?question=What%20documents%20are%20required%20for%20admission
```

## Example Questions Handled

- What is the DTE code?
- What documents are needed for admission?
- Does UMIT offer hostel facilities?
- Which B.Tech programs are available?

## Notes

- For context-based answers, the entire `data.txt` content is passed as background context to the Falcon model.
- The chatbot attempts a direct keyword match before falling back to LLM-generated responses.

## License

This project is for academic and learning purposes.
