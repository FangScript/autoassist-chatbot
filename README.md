# DPF Specialist Chatbot - Vercel Deployment Ready

A conversational AI chatbot for DPF (Diesel Particulate Filter) Specialist services, built with Flask, LangChain, and Pinecone, optimized for Vercel deployment.

## 🚀 Quick Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/FangScript/autoassist-chatbot)

## Features

- **Conversational AI**: Powered by OpenAI GPT-4o-mini
- **RAG (Retrieval-Augmented Generation)**: Uses Pinecone vector database for document retrieval
- **Professional UI**: Modern, responsive chat widget interface
- **Business Integration**: Hardcoded service information and contact details
- **Memory**: Maintains conversation context (last 5 exchanges)
- **Serverless Ready**: Optimized for Vercel deployment

## 🛠️ Manual Setup

### Prerequisites

- Vercel account
- OpenAI API key
- Pinecone API key
- Pinecone index with DPF documentation

### Environment Variables

Set these in your Vercel dashboard:

```bash
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Pinecone Setup

1. Create a Pinecone index named `dpf-chatbot`
2. Set namespace to `qa-namespace`
3. Use `text-embedding-3-small` model (1536 dimensions)
4. Upload your DPF documentation using the `embeddings.py` script locally

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/FangScript/autoassist-chatbot.git
cd autoassist-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export PINECONE_API_KEY="your_pinecone_api_key_here"
```

4. Run locally:
```bash
vercel dev
```

## 📁 Project Structure

```
├── api/
│   └── index.py          # Main Flask app (Vercel serverless function)
├── static/               # Static assets (logos, images)
├── templates/           # HTML templates (legacy, now embedded)
├── dataset/             # DPF documentation PDFs
├── embeddings.py        # Document indexing script (run locally)
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel configuration
├── .vercelignore        # Files to exclude from deployment
└── README.md           # This file
```

## 🔧 Vercel Configuration

The `vercel.json` file configures:
- Python runtime for API functions
- Static file serving for assets
- Route handling for chat and static files
- Environment variable mapping

## 📊 Business Information

- **Services**: DPF Cleaning, Regeneration, Replacement, Removal, Diagnostics
- **Website**: https://www.dpfspecialist.co.uk/
- **Phone**: 0330 162 8424
- **Email**: info@dpfspecialist.co.uk
- **Instagram**: https://www.instagram.com/dpf_specialist/

## 🚀 Deployment Steps

1. **Fork/Clone** this repository
2. **Connect** to Vercel
3. **Set Environment Variables** in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
4. **Deploy** - Vercel will automatically build and deploy

## 🔒 Security

- ✅ No hardcoded API keys
- ✅ Environment variables for sensitive data
- ✅ Proper `.vercelignore` configuration
- ✅ Clean git history

## 📝 Usage

1. Open the deployed URL
2. Click the floating chat widget
3. Ask questions about DPF services
4. The bot provides contextual responses based on indexed documentation

## 🛠️ Customization

- **Styling**: Modify CSS in `api/index.py` HTML template
- **Business Info**: Update hardcoded URLs and contact details
- **Prompts**: Modify the prompt template for different responses
- **Memory**: Adjust conversation memory settings

## 📈 Performance

- **Serverless**: Scales automatically with Vercel
- **Edge**: Global CDN for fast loading
- **Caching**: Optimized for repeated requests
- **Cold Start**: Minimal initialization time

## 🐛 Troubleshooting

- **API Keys**: Ensure environment variables are set correctly
- **Pinecone**: Verify index exists and has data
- **Memory**: Check conversation memory limits
- **Logs**: Use Vercel dashboard for debugging

## 📄 License

This project is for DPF Specialist business use.

---

**Ready to deploy?** Click the Vercel button above or follow the manual setup instructions!
