## DPF Specialist Chatbot

Flask + LangChain RAG chatbot using FAISS and OpenAI.

### Local setup
1. Create and activate a Python 3.10+ venv.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
4. Run: `python app.py` then open `http://localhost:5000`.

### Railway deployment
- Railway detects Python and uses the `Procfile`.
- Ensure `OPENAI_API_KEY` is set in Railway project variables.
- The app binds to `0.0.0.0:${PORT}` automatically.

#### Steps
1. Push this repo to GitHub.
2. Create a new Railway project and deploy from the repo.
3. Add environment variable `OPENAI_API_KEY` in Railway.
4. Deploy. The service starts via `web: gunicorn app:app`.

### Rebuilding embeddings
Open `/rebuild` endpoint to regenerate FAISS index from PDFs in `dataset/`.

