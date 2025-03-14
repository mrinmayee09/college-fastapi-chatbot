from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
origins = [
    "http://localhost",  # Frontend URL
    "http://localhost:8000",  # Backend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load Falcon-7B-Instruct model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"  # Lightweight but powerful
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Function to load the knowledge base
def load_knowledge_base(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
        return [line.strip() for line in content if line.strip()]
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

# Direct keyword matching for specific questions
def keyword_match(question, knowledge_base):
    question = question.lower()
    
    # Match for documents required
    if "documents required" in question or "documents for admission" in question:
        return """
        Required documents for admission include:
        1. FC acknowledgement letter
        2. CAP allotment letter
        3. Seat Acceptance Letter
        4. MH-CET/JEE score card
        5. S.S.C. Marksheet
        6. H.S.C. Marksheet
        7. Jr.College Leaving Certificate
        8. GAP certificate (if any) on Rs. 100 stamp paper
        9. Migration certificate (for CBSE/ICSE/OMS/J & K students)
        10. Domicile certificate
        11. Age and Nationality certificate
        12. Birth certificate
        13. Aadhar card photocopy-2
        14. Ration Card photocopy-2 (Front and last page only)
        15. Two passport size photos
        16. Caste certificate (If applicable)
        17. Caste Validity Certificate (If applicable)
        18. Non Creamy Layer Certificate (for OBC/SBC/VJ/NT)
        19. Income Certificate issued by competent authority of Government of Maharashtra
        20. EWS certificate-Issued by competent authority of Government of Maharashtra.
        """
    
    # Match for DTE code
    if "dte code" in question:
        return "The DTE Code for UMIT is 3035."
    
    # Match for programs offered
    if "b.tech programs" in question or "what are the courses offered" in question:
        return "UMIT offers six B.Tech programs: Computer Science, Information Technology, Electronics & Communication, Computer Engineering, Artificial Intelligence, and Data Science."
    
    # Match for hostel
    if "hostel" in question:
        return "Hostel allotment is strictly done on merit due to limited availability of seats."
    
    # Match for admission
    if "admission" in question:
        return "Admission is done through CAP (Centralized Admission Process) conducted by DTE. The process is online through the DTE website."
    
    return None  # If no match is found

# Use Falcon-7B for generating answers
def generate_answer_with_falcon(question, context):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the knowledge base (faq.txt file)
file_path = "data.txt"  # Ensure the path to your file is correct
knowledge_base = load_knowledge_base(file_path)

@app.get("/chat")
async def chat(question: str = Query(..., description="Ask a question.")):
    # First check for keyword-based direct answers
    direct_answer = keyword_match(question, knowledge_base)
    
    if direct_answer:
        return {"answer": direct_answer}

    # Combine the entire knowledge base as context for Falcon
    context = " ".join(knowledge_base)
    falcon_answer = generate_answer_with_falcon(question, context)
    
    # Default fallback if Falcon doesn't produce a useful answer
    if not falcon_answer.strip():
        falcon_answer = "Sorry, I don't have an answer for your query. Please contact xxxxxxxxxx for further assistance from our Admission Committee."
    
    return {"answer": falcon_answer}
