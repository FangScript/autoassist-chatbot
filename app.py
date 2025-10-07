from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# --- Flask Setup ---
app = Flask(__name__)
chat_started = False

# --- Environment & API Key ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Paths ---
PDF_FOLDER = "dataset/"
VECTOR_STORE_PATH = "embeddings/faiss_index"

# --- URLs & Contact Info ---
SERVICES_URL = "https://www.dpfspecialist.co.uk/our-services/"
BOOKING_URL = "https://www.dpfspecialist.co.uk/book-your-dpf-service/"
PHONE_NUMBER = "0330 010 0366"
INSTAGRAM_URL = "https://www.instagram.com/dpf_specialist/"
EMAIL = "info@dpfspecialist.co.uk"

# --- Common Questions / Answers (Inline JSON) ---
COMMON_QA = [
    {
        "question": "services",
        "answer": "<h5>Our Services</h5><ul><li>Mobile DPF Cleaning Service</li><li>DPF Fleet Cleaning Packages</li><li>Particle Filter Additive</li><li>Pressure Sensor Replacement</li><li>Fuel Vaporizer Replacement</li><li>Industrial Off-Car DPF Cleaning</li><li>Land Rover Exhaust Filter Specialist</li></ul>Check <a href='https://www.dpfspecialist.co.uk/our-services/' target='_blank'>our services</a> for more details."
    },
    {
        "question": "book",
        "answer": f"You can easily book your service <a href='{BOOKING_URL}' target='_blank'>here</a> 📅"
    },
    {
        "question": "contact",
        "answer": f"You can reach us at <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a> or email <a href='mailto:{EMAIL}'>{EMAIL}</a>."
    },
    {
        "question": "social",
        "answer": f"Follow us on Instagram <a href='{INSTAGRAM_URL}' target='_blank'>@dpf_specialist</a> for updates and demos."
    },
    {
        "question": "hi",
        "answer": f"Hey there! 👋 Ready to dive into DPF solutions? Ask about cleaning, services, or book a slot at <a href='{BOOKING_URL}' target='_blank'>our booking page</a>!"
    },
    {
        "question": "hello",
        "answer": f"Yo! I'm your DPF Specialist assistant. What can I help you with today? 🚗"
    }
]

# --- Initialize Embeddings & LLM ---
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.6, max_tokens=500)

# --- PDF Loading ---
def load_pdfs(folder):
    text = ""
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf = PdfReader(os.path.join(folder, file))
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    return text

# --- Create FAISS Vectorstore ---
def create_vectorstore():
    print("📄 Creating embeddings from PDFs...")
    text = load_pdfs(PDF_FOLDER)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    os.makedirs("embeddings", exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print("✅ Vectorstore built successfully!")

# --- Load FAISS Vectorstore ---
def load_vectorstore():
    if not os.path.exists(VECTOR_STORE_PATH):
        create_vectorstore()
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# --- Build Conversational Retrieval QA ---
def init_chatbot():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    prompt_template = """You are a chill, expert support agent for **DPF Specialist**, the UK's top Diesel Particulate Filter experts.  
Provide clear, short (2–4 sentence) answers using headings or bullet points when needed.  

**Your Rules:**
1. Use the context if available.  
2. If context doesn’t cover the answer, rely on your automotive expertise.  
3. If still unsure, provide contact info: <a href='tel:{phone}'>{phone}</a> or <a href='mailto:{email}'>{email}</a>.

Tone: professional, confident, and approachable.  
Format replies in clean HTML (use <h5>, <ul><li>, <strong> as needed).

Context: {{context}}
Chat History: {{chat_history}}
Question: {{question}}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template.format(
            phone=PHONE_NUMBER, email=EMAIL
        ),
        input_variables=["context", "chat_history", "question"]
    )

    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    print("🤖 Chatbot initialized successfully!")
    return qa_chain

# Initialize RAG chatbot
qa_chain = init_chatbot()

# --- Helper: Find match in common Q&A ---
def check_common_question(message):
    msg = message.lower()
    for faq in COMMON_QA:
        if faq["question"].lower() in msg:
            return faq["answer"]
    return None

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_started
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please type something 🙂"})

    # Step 1: Greeting logic (simple)
    if not chat_started and user_message.lower() in ["hi", "hello", "hey", "yo", "greetings"]:
        chat_started = True
        return jsonify({"response": check_common_question("hi")})

    chat_started = True

    # Step 2: Check for hardcoded common Q&A
    answer = check_common_question(user_message)
    if answer:
        return jsonify({"response": answer})

    # Step 3: Fallback to RAG / memory
    try:
        response = qa_chain({"question": user_message})
        final = response.get("answer", f"Oops, something went wrong! 😅 Reach out at {EMAIL} or call {PHONE_NUMBER}.")
        return jsonify({"response": final})
    except Exception as e:
        print(f"❌ Error in RAG pipeline: {e}")
        return jsonify({"response": f"Oops, something went wrong! 😅 Reach out at {EMAIL} or call {PHONE_NUMBER}."})

@app.route("/rebuild", methods=["GET"])
def rebuild():
    create_vectorstore()
    return jsonify({"status": "✅ Embeddings rebuilt successfully!"})

# --- Main ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG", "0") == "1")
