import os
import re
import json
from flask import Flask, request, jsonify, render_template_string
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from pinecone import Pinecone

# Suppress Pydantic warning
from pydantic import BaseModel

# Load API keys from environment variables
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
  - For unknown info (e.g., location, hours), say "I don't have that info" once; for repeated queries, suggest contact without repeating.
  - For introductions (e.g., "my name is X"), personalize the response.
  - For service queries, use context; if insufficient, use this fallback: {FALLBACK_SERVICES}
  - Don't repeat service lists unless explicitly asked.

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

# HTML template for the chat interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DPF Specialist Chatbot Widget</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Chat Widget Button */
        .chat-widget-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #ff9966 0%, #ff7733 100%);
            border-radius: 50%;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(255, 119, 51, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            z-index: 1000;
        }

        .chat-widget-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 25px rgba(255, 119, 51, 0.6);
        }

        .chat-widget-btn svg {
            width: 30px;
            height: 30px;
            fill: white;
        }

        .chat-widget-btn.active {
            background: #1a1a1a;
        }

        /* Notification Badge */
        .notification-badge {
            position: absolute;
            top: -5px;
            right: -5px;
            background: #ff3333;
            color: white;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.1);
            }
        }

        /* Chat Container */
        .chat-container {
            position: fixed;
            bottom: 100px;
            right: 30px;
            width: 380px;
            height: 550px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            display: none;
            flex-direction: column;
            overflow: hidden;
            z-index: 999;
            animation: slideUp 0.3s ease;
        }

        .chat-container.active {
            display: flex;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Chat Header */
        .chat-header {
            background: linear-gradient(135deg, #ff9966 0%, #F09301 100%);
            color: white;
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .bot-avatar {
            width: 45px;
            height: 45px;
            background: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            flex-shrink: 0;
            border: 2px solid #1a1a1a;
        }

        .bot-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .chat-header-info h3 {
            font-size: 18px;
            margin-bottom: 3px;
        }

        .status {
            font-size: 13px;
            opacity: 0.9;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            animation: blink 2s infinite;
        }

        @keyframes blink {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }

        /* Chat Messages */
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #fafafa;
        }

        /* Welcome Screen */
        .welcome-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            gap: 30px;
            animation: fadeInUp 0.6s ease;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .logo-container {
            animation: logoFadeIn 0.6s ease;
            position: relative;
        }

        @keyframes logoFadeIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .chat-logo {
            max-width: 280px;
            height: auto;
            filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.08));
        }

        .welcome-message {
            text-align: center;
            animation: slideInBottom 0.8s ease 0.3s both;
        }

        @keyframes slideInBottom {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .bot-icon-large {
            width: 80px;
            height: 80px;
            margin: 0 auto 15px;
            animation: fadeInSimple 0.8s ease 0.6s both;
            border: 3px solid #1a1a1a;
            border-radius: 50%;
            overflow: hidden;
            background: white;
        }

        .bot-icon-large img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        @keyframes fadeInSimple {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        .welcome-text {
            background: white;
            padding: 20px 25px;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            max-width: 280px;
            font-size: 15px;
            line-height: 1.6;
            position: relative;
        }

        .welcome-text::before {
            content: '';
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 10px solid transparent;
            border-right: 10px solid transparent;
            border-bottom: 10px solid white;
        }

        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.bot {
            display: flex;
            gap: 10px;
        }

        .message.user {
            display: flex;
            justify-content: flex-end;
        }

        .message-avatar {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #ff9966 0%, #ff7733 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            overflow: hidden;
            border: 2px solid #1a1a1a;
        }

        .message-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .message-content {
            background: white;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 70%;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .message.user .message-content {
            background: #1a1a1a;
            color: white;
        }

        /* Markdown Rendering Styles */
        .message-content h5 {
            margin: 0 0 10px;
            font-size: 1.1em;
            color: #2c3e50;
        }
        .message-content ul {
            margin: 5px 0;
            padding-left: 20px;
            list-style-type: disc;
        }
        .message-content li {
            margin-bottom: 5px;
        }
        .message-content strong {
            font-weight: 600;
            color: #ff7733;
        }
        .message-content br {
            line-height: 1.2;
        }

        /* Typing Indicator */
        .typing-indicator {
            display: none;
            gap: 10px;
            margin-bottom: 15px;
        }

        .typing-indicator.active {
            display: flex;
        }

        .typing-dots {
            background: white;
            padding: 12px 16px;
            border-radius: 18px;
            display: flex;
            gap: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: #ff7733;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }

        /* Chat Input */
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e5e5e5;
            display: flex;
            gap: 10px;
        }

        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e5e5e5;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
            transition: border-color 0.3s;
        }

        .chat-input input:focus {
            border-color: #ff7733;
        }

        .send-btn {
            width: 45px;
            height: 45px;
            background: linear-gradient(135deg, #ff9966 0%, #F09301 100%);
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s;
        }

        .send-btn:hover:not(:disabled) {
            transform: scale(1.1);
        }

        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .send-btn svg {
            width: 20px;
            height: 20px;
            fill: white;
        }

        /* Scrollbar */
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: #ff7733;
            border-radius: 3px;
        }

        @media (max-width: 400px) {
            .chat-container {
                width: calc(100vw - 60px);
                height: calc(100vh - 120px);
                bottom: 60px;
                right: 30px;
            }
        }
    </style>
</head>
<body>
    <!-- Chat Widget Button -->
    <button class="chat-widget-btn" id="chatBtn">
        <svg viewBox="0 0 24 24" id="chatIcon">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
        </svg>
        <svg viewBox="0 0 24 24" id="closeIcon" style="display: none;">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
        <span class="notification-badge">1</span>
    </button>

    <!-- Chat Container -->
    <div class="chat-container" id="chatContainer">
        <div class="chat-header">
            <div class="bot-avatar">
                <img src="/static/logo2.jpg" alt="Support">
            </div>
            <div class="chat-header-info">
                <h3>DPF Specialist Bot</h3>
                <div class="status">
                    <span class="status-dot"></span>
                    Online
                </div>
            </div>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="welcome-screen" id="welcomeScreen">
                <div class="logo-container">
                    <img src="/static/image.png" alt="DPF Specialist" class="chat-logo">
                </div>
                <div class="welcome-message">
                    <div class="bot-icon-large">
                        <img src="/static/logo2.jpg" alt="Bot">
                    </div>
                    <div class="welcome-text">
                        Yo! Welcome to DPF Specialist! 😎 What's on your mind? Ask about DPF fixes, call us at <a href='tel:0330 029 9561'>0330 029 9561</a>, or check out <a href='https://www.dpfspecialist.co.uk/our-services/' target='_blank'>our services</a>!
                    </div>
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <div class="message-avatar">
                <img src="/static/logo2.jpg" alt="Bot">
            </div>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>

        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="Type your message..." maxlength="500">
            <button class="send-btn" id="sendBtn" disabled>
                <svg viewBox="0 0 24 24">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
            </button>
        </div>
    </div>

    <script>
        const chatBtn = document.getElementById('chatBtn');
        const chatContainer = document.getElementById('chatContainer');
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const chatIcon = document.getElementById('chatIcon');
        const closeIcon = document.getElementById('closeIcon');
        const notificationBadge = document.querySelector('.notification-badge');
        const typingIndicator = document.getElementById('typingIndicator');
        const welcomeScreen = document.getElementById('welcomeScreen');

        // Parse Markdown-like syntax to HTML
        function parseMarkdown(text) {
            // Convert ### to <h5>
            text = text.replace(/^###\s*(.+)$/gm, '<h5>$1</h5>');
            // Convert **text** to <strong>text</strong>
            text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            // Convert - item to <ul><li>item</li></ul>
            text = text.replace(/^- (.+)$/gm, '<li>$1</li>');
            text = text.replace(/(<li>.+<\/li>)/g, '<ul>$1</ul>');
            // Convert newlines to <br> (except within lists)
            text = text.replace(/\\n(?!(<ul>|<\/ul>|<li>))/g, '<br>');
            return text;
        }

        // Toggle chat
        chatBtn.addEventListener('click', () => {
            chatContainer.classList.toggle('active');
            chatBtn.classList.toggle('active');
            
            if (chatContainer.classList.contains('active')) {
                chatIcon.style.display = 'none';
                closeIcon.style.display = 'block';
                notificationBadge.style.display = 'none';
                messageInput.focus();
            } else {
                chatIcon.style.display = 'block';
                closeIcon.style.display = 'none';
            }
        });

        // Send message to backend
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (message === '') return;

            // Hide welcome screen on first message
            if (welcomeScreen) {
                welcomeScreen.style.display = 'none';
            }

            // Add user message
            addMessage(message, 'user');
            messageInput.value = '';
            sendBtn.disabled = true;

            // Show typing indicator
            typingIndicator.classList.add('active');
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                let botResponse = data.response || 'No response received.';

                // Replace old phone number with new one in bot response
                botResponse = botResponse.replace(/0330 162 8424/g, '0330 029 9561');

                // Hide typing indicator and add bot message
                typingIndicator.classList.remove('active');
                addMessage(botResponse, 'bot');
            } catch (error) {
                console.error('Error:', error);
                typingIndicator.classList.remove('active');
                addMessage(`Oops, something went wrong! 😅 Reach out at <a href='mailto:info@dpfspecialist.co.uk'>info@dpfspecialist.co.uk</a> or call <a href='tel:0330 029 9561'>0330 029 9561</a>.`, 'bot');
            } finally {
                sendBtn.disabled = false;
            }
        }

        // Add message to chat
        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;

            if (sender === 'bot') {
                // Parse Markdown for bot responses
                const parsedText = parseMarkdown(text);
                messageDiv.innerHTML = `
                    <div class="message-avatar">
                        <img src="/static/logo2.jpg" alt="Bot">
                    </div>
                    <div class="message-content">${parsedText}</div>
                `;
            } else {
                messageDiv.innerHTML = `
                    <div class="message-content">${text}</div>
                `;
            }

            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Enable send button when typing
        messageInput.addEventListener('input', () => {
            sendBtn.disabled = messageInput.value.trim() === '';
        });

        // Initial notification badge (remove after first open)
        setTimeout(() => {
            notificationBadge.style.display = 'none';
        }, 5000); // Hide after 5 seconds or on open
    </script>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

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

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda *args: None)
