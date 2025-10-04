# app.py
import os
import re
from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from pinecone import Pinecone

# Suppress Pydantic warning
from pydantic import BaseModel

# Load API keys from environment variables
# Set these in your environment or .env file:
# OPENAI_API_KEY=your_openai_api_key_here
# PINECONE_API_KEY=your_pinecone_api_key_here

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")
if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY environment variable is required")

# Pinecone index details
INDEX_NAME = "dpf-chatbot"
NAMESPACE = "qa-namespace"

# Hardcoded data
SERVICES_URL = "https://www.dpfspecialist.co.uk/our-services/"
BOOKING_URL = "https://www.dpfspecialist.co.uk/book-your-dpf-service/"
PHONE_NUMBER = "0330 162 8424"
INSTAGRAM_URL = "https://www.instagram.com/dpf_specialist/"
EMAIL = "info@dpfspecialist.co.uk"

# Hardcoded greeting responses
GREETINGS = {
    "hi": f"Hey there! Ready to dive into DPF solutions? 😎 Ask about cleaning, services, or book a slot at <a href='{BOOKING_URL}' target='_blank'>our booking page</a>!",
    "hello": f"Yo! I'm your DPF Specialist assistant. What's up? Check out <a href='{SERVICES_URL}' target='_blank'>our services</a> or ask me anything!"
}
GREETING_KEYWORDS = ["hi", "hello", "hey", "greetings", "yo"]

# Fallback service list (used if RAG context is insufficient)
FALLBACK_SERVICES = """
<h3>Our Services</h3>
<ul>
    <li>DPF Cleaning</li>
    <li>DPF Regeneration</li>
    <li>DPF Replacement</li>
    <li>DPF Removal</li>
    <li>DPF Diagnostics</li>
</ul>
Check <a href='https://www.dpfspecialist.co.uk/our-services/' target='_blank'>our services</a> or <a href='https://www.instagram.com/dpf_specialist/' target='_blank'>Instagram</a> for more!
"""

# Initialize Pinecone client
print("Initializing Pinecone client for chatbot...")
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    print("Pinecone client initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone client: {str(e)}")
    exit(1)

# Embeddings and LLM
print("Initializing OpenAI embeddings and LLM...")
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    print("Embeddings and LLM initialized successfully")
except Exception as e:
    print(f"Error initializing OpenAI models: {str(e)}")
    exit(1)

# Connect to Pinecone vector store
print(f"Connecting to Pinecone index: {INDEX_NAME}")
try:
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE
    )
    print(f"Connected to Pinecone index '{INDEX_NAME}' in namespace '{NAMESPACE}'")
except Exception as e:
    print(f"Error connecting to Pinecone vector store: {str(e)}")
    exit(1)

# Conversation memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,  # Keep last 5 exchanges
    return_messages=True,
    output_key="answer"
)

# Custom prompt
prompt_template = """
You're a chill, expert support agent for DPF Specialist, the go-to for diesel particulate filter fixes. Keep answers short (2-3 sentences), engaging, and use bullet points or headings for clarity. Use chat history to avoid repetitive answers and stay context-aware. Follow these rules:

- **Tone**: Cool, friendly, professional.
- **Links**: Only include relevant links:
  - For service/cleaning details (e.g., "what services," "how do you clean"): Use <a href='{SERVICES_URL}' target='_blank'>our services</a> and <a href='{INSTAGRAM_URL}' target='_blank'>Instagram</a>.
  - For contact queries: Use <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a> or <a href='mailto:{EMAIL}'>{EMAIL}</a>.
  - For booking interest (new customers, e.g., general DPF questions): Suggest <a href='{BOOKING_URL}' target='_blank'>book an appointment</a>.
- **Intent**:
  - New customers (general queries, pricing, DPF basics): Suggest booking.
  - Returning customers (mention past services, issues): Skip booking, offer help or contact info.
- **Specifics**:
  - Avoid generic "visit our website" since users are on the site.
  - For unknown info (e.g., location, hours), say "I don’t have that info" once; for repeated queries, suggest contact without repeating.
  - For introductions (e.g., "my name is X"), personalize the response.
  - For service queries, use context; if insufficient, use this fallback: {FALLBACK_SERVICES}
  - Don’t repeat service lists unless explicitly asked.

**Chat History**: {chat_history}
**Context**: {context}
**User Query**: {question}

Answer concisely with relevant links only. Use HTML for formatting (e.g., <h3> for headings, <ul><li> for lists, <strong> for bold). Debug: Log retrieved context for analysis.
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"],
    partial_variables={
        "SERVICES_URL": SERVICES_URL,
        "BOOKING_URL": BOOKING_URL,
        "PHONE_NUMBER": PHONE_NUMBER,
        "INSTAGRAM_URL": INSTAGRAM_URL,
        "EMAIL": EMAIL,
        "FALLBACK_SERVICES": FALLBACK_SERVICES
    }
)

# Conversational RAG chain
print("Setting up ConversationalRetrievalChain...")
try:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True  # Return source docs for debugging
    )
    print("ConversationalRetrievalChain set up successfully")
except Exception as e:
    print(f"Error setting up ConversationalRetrievalChain: {str(e)}")
    exit(1)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": f"Oops, type something! 😎 Need DPF help? Check <a href='{SERVICES_URL}' target='_blank'>our services</a> or hit us up at <a href='mailto:{EMAIL}'>{EMAIL}</a>."}), 400
    
    print(f"Processing user query: {user_message}")
    
    # Stricter greeting check
    user_message_lower = user_message.lower().strip()
    for keyword in GREETING_KEYWORDS:
        if re.fullmatch(rf"{keyword}[\s!?]*", user_message_lower):
            response = GREETINGS.get(keyword, GREETINGS["hi"])
            print(f"Hardcoded greeting response: {response}")
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return jsonify({"response": response})
    
    # Handle introductions
    name_match = re.search(r"(?:my name is|im|i'm)\s+([a-zA-Z]+)", user_message_lower, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).capitalize()
        response = f"Nice to meet you, {name}! 😎 How can I help with your DPF needs? Check <a href='{SERVICES_URL}' target='_blank'>our services</a> or ask away!"
        print(f"Introduction response: {response}")
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(response)
        return jsonify({"response": response})

    # RAG for other queries
    try:
        result = qa_chain.invoke({"question": user_message})
        response = result["answer"]
        # Debug: Log retrieved context
        print(f"Retrieved context: {[doc.page_content for doc in result['source_documents']]}")
        print(f"Generated RAG response: {response}")
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({"response": f"Whoa, something broke! 😅 Try again or reach out at <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a> or <a href='mailto:{EMAIL}'>{EMAIL}</a>."}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)