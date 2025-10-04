import os
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import re

# Load API keys from environment variables
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")
if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY environment variable is required")

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

# Fallback service list
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

# Simple response generator (without LangChain for now)
def generate_response(user_message):
    user_message_lower = user_message.lower().strip()
    
    # Check for greetings
    for keyword in GREETING_KEYWORDS:
        if re.fullmatch(rf"{keyword}[\s!?]*", user_message_lower):
            return GREETINGS.get(keyword, GREETINGS["hi"])
    
    # Handle introductions
    name_match = re.search(r"(?:my name is|im|i'm)\s+([a-zA-Z]+)", user_message_lower, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).capitalize()
        return f"Nice to meet you, {name}! 😎 How can I help with your DPF needs? Check <a href='{SERVICES_URL}' target='_blank'>our services</a> or ask away!"
    
    # Handle common DPF queries
    if any(word in user_message_lower for word in ["dpf", "diesel", "filter", "particulate"]):
        return f"Great question about DPF systems! 🔧 We specialize in DPF cleaning, regeneration, and replacement. Check out <a href='{SERVICES_URL}' target='_blank'>our services</a> or <a href='{BOOKING_URL}' target='_blank'>book an appointment</a> for expert help!"
    
    if any(word in user_message_lower for word in ["clean", "cleaning", "service"]):
        return f"Our DPF cleaning service is top-notch! 🧽 We use professional equipment and techniques. <a href='{BOOKING_URL}' target='_blank'>Book your service</a> or call us at <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a> for more info!"
    
    if any(word in user_message_lower for word in ["price", "cost", "how much"]):
        return f"Pricing varies based on your specific DPF needs! 💰 For accurate quotes, <a href='{BOOKING_URL}' target='_blank'>book a consultation</a> or call <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a>. We offer competitive rates!"
    
    if any(word in user_message_lower for word in ["location", "where", "address"]):
        return f"We're located in the UK and serve customers nationwide! 📍 For specific location details, call <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a> or email <a href='mailto:{EMAIL}'>{EMAIL}</a>."
    
    if any(word in user_message_lower for word in ["contact", "phone", "call"]):
        return f"Get in touch! 📞 Call <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a> or email <a href='mailto:{EMAIL}'>{EMAIL}</a>. We're here to help with all your DPF needs!"
    
    # Default response
    return f"Thanks for reaching out! 😊 I'm here to help with DPF questions. Check <a href='{SERVICES_URL}' target='_blank'>our services</a>, <a href='{BOOKING_URL}' target='_blank'>book an appointment</a>, or call <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a> for expert assistance!"

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DPF Specialist Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
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
        .chat-widget-btn:hover { transform: scale(1.1); }
        .chat-widget-btn svg { width: 30px; height: 30px; fill: white; }
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
        }
        .chat-container.active { display: flex; }
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
        .bot-avatar img { width: 100%; height: 100%; object-fit: cover; }
        .bot-avatar { background: linear-gradient(135deg, #ff9966 0%, #ff7733 100%); }
        .message-avatar img { width: 100%; height: 100%; object-fit: cover; }
        .message-avatar { background: linear-gradient(135deg, #ff9966 0%, #ff7733 100%); }
        .chat-header-info h3 { font-size: 18px; margin-bottom: 3px; }
        .status { font-size: 13px; opacity: 0.9; display: flex; align-items: center; gap: 5px; }
        .status-dot {
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            animation: blink 2s infinite;
        }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #fafafa;
        }
        .welcome-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            gap: 30px;
        }
        .welcome-text {
            background: white;
            padding: 20px 25px;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            max-width: 280px;
            font-size: 15px;
            line-height: 1.6;
            text-align: center;
        }
        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.bot { display: flex; gap: 10px; }
        .message.user { display: flex; justify-content: flex-end; }
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
        .message-avatar img { width: 100%; height: 100%; object-fit: cover; }
        .message-content {
            background: white;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 70%;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .message.user .message-content { background: #1a1a1a; color: white; }
        .message-content h5 { margin: 0 0 10px; font-size: 1.1em; color: #2c3e50; }
        .message-content ul { margin: 5px 0; padding-left: 20px; list-style-type: disc; }
        .message-content li { margin-bottom: 5px; }
        .message-content strong { font-weight: 600; color: #ff7733; }
        .typing-indicator { display: none; gap: 10px; margin-bottom: 15px; }
        .typing-indicator.active { display: flex; }
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
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-10px); } }
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
        .chat-input input:focus { border-color: #ff7733; }
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
        .send-btn:hover:not(:disabled) { transform: scale(1.1); }
        .send-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .send-btn svg { width: 20px; height: 20px; fill: white; }
        .chat-messages::-webkit-scrollbar { width: 6px; }
        .chat-messages::-webkit-scrollbar-track { background: #f1f1f1; }
        .chat-messages::-webkit-scrollbar-thumb { background: #ff7733; border-radius: 3px; }
        @media (max-width: 400px) {
            .chat-container { width: calc(100vw - 60px); height: calc(100vh - 120px); bottom: 60px; right: 30px; }
        }
    </style>
</head>
<body>
    <button class="chat-widget-btn" id="chatBtn">
        <svg viewBox="0 0 24 24" id="chatIcon">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
        </svg>
        <svg viewBox="0 0 24 24" id="closeIcon" style="display: none;">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
    </button>

    <div class="chat-container" id="chatContainer">
        <div class="chat-header">
            <div class="bot-avatar">
                <img src="/static/logo2.jpg" alt="Support" onerror="this.style.display='none'; this.parentElement.innerHTML='🤖';">
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
                <div class="welcome-text">
                    Yo! Welcome to DPF Specialist! 😎 What's on your mind? Ask about DPF fixes, call us at <a href='tel:0330 029 9561'>0330 029 9561</a>, or check out <a href='https://www.dpfspecialist.co.uk/our-services/' target='_blank'>our services</a>!
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <div class="message-avatar">
                <img src="/static/logo2.jpg" alt="Bot" onerror="this.style.display='none'; this.parentElement.innerHTML='🤖';">
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
        const typingIndicator = document.getElementById('typingIndicator');
        const welcomeScreen = document.getElementById('welcomeScreen');

        function parseMarkdown(text) {
            text = text.replace(/^###\\s*(.+)$/gm, '<h5>$1</h5>');
            text = text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
            text = text.replace(/^- (.+)$/gm, '<li>$1</li>');
            text = text.replace(/(<li>.+<\\/li>)/g, '<ul>$1</ul>');
            text = text.replace(/\\n(?!(<ul>|<\\/ul>|<li>))/g, '<br>');
            return text;
        }

        chatBtn.addEventListener('click', () => {
            chatContainer.classList.toggle('active');
            chatBtn.classList.toggle('active');
            
            if (chatContainer.classList.contains('active')) {
                chatIcon.style.display = 'none';
                closeIcon.style.display = 'block';
                messageInput.focus();
            } else {
                chatIcon.style.display = 'block';
                closeIcon.style.display = 'none';
            }
        });

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (message === '') return;

            if (welcomeScreen) {
                welcomeScreen.style.display = 'none';
            }

            addMessage(message, 'user');
            messageInput.value = '';
            sendBtn.disabled = true;

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
                botResponse = botResponse.replace(/0330 162 8424/g, '0330 029 9561');

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

        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;

            if (sender === 'bot') {
                const parsedText = parseMarkdown(text);
                messageDiv.innerHTML = `
                    <div class="message-avatar">
                        <img src="/static/logo2.jpg" alt="Bot" onerror="this.style.display='none'; this.parentElement.innerHTML='🤖';">
                    </div>
                    <div class="message-content">${parsedText}</div>
                `;
            } else {
                messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
            }

            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        messageInput.addEventListener('input', () => {
            sendBtn.disabled = messageInput.value.trim() === '';
        });
    </script>
</body>
</html>
"""

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path.startswith('/static/'):
            # Let Vercel handle static files
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_message = data.get('message', '')
                
                if not user_message:
                    response = f"Oops, type something! 😎 Need DPF help? Check <a href='{SERVICES_URL}' target='_blank'>our services</a> or hit us up at <a href='mailto:{EMAIL}'>{EMAIL}</a>."
                else:
                    response = generate_response(user_message)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'response': response}).encode())
                
            except Exception as e:
                print(f"Error processing chat request: {str(e)}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'response': f"Whoa, something broke! 😅 Try again or reach out at <a href='tel:{PHONE_NUMBER}'>{PHONE_NUMBER}</a> or <a href='mailto:{EMAIL}'>{EMAIL}</a>."}).encode())
        else:
            self.send_response(404)
            self.end_headers()