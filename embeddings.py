# index_data_with_debug.py
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
DIMENSION = 1536  # Matches text-embedding-3-small
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

# Initialize Pinecone client
print("Initializing Pinecone client...")
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    print("Pinecone client initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone client: {str(e)}")
    exit(1)

# Check if index exists, create if it doesn't
print(f"Checking for Pinecone index: {INDEX_NAME}")
try:
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index {INDEX_NAME} does not exist. Creating index with dimension {DIMENSION}, metric {METRIC}, cloud {CLOUD}, region {REGION}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
        print(f"Index {INDEX_NAME} created successfully with dimension {DIMENSION}, metric {METRIC}, cloud {CLOUD}, region {REGION}")
    else:
        print(f"Index {INDEX_NAME} already exists")
except Exception as e:
    print(f"Error checking/creating Pinecone index: {str(e)}")
    exit(1)

# Load PDFs
print("Starting PDF loading process...")
pdf_files = ["dataset/DS-1.pdf", "dataset/DS-2.pdf"]
docs = []
for file in pdf_files:
    print(f"Loading PDF: {file}")
    try:
        loader = PyPDFLoader(file)
        loaded_docs = loader.load()
        print(f"Successfully loaded {len(loaded_docs)} pages from {file}")
        docs.extend(loaded_docs)
    except Exception as e:
        print(f"Error loading {file}: {str(e)}")

if not docs:
    print("Error: No documents loaded from PDFs. Check file paths and formats.")
    exit(1)

print(f"Total pages loaded from all PDFs: {len(docs)}")

# Split documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)
print(f"Created {len(split_docs)} chunks from {len(docs)} pages")

# Initialize embeddings model
print("Initializing OpenAI embeddings model (text-embedding-3-small)...")
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    print("Embeddings model initialized successfully")
except Exception as e:
    print(f"Error initializing embeddings model: {str(e)}")
    exit(1)

# Upsert dense vectors to Pinecone
print(f"Upserting dense vectors to Pinecone index: {INDEX_NAME}")
try:
    vectorstore = PineconeVectorStore.from_documents(
        split_docs,
        embeddings,
        index_name=INDEX_NAME,
        namespace="qa-namespace"
    )
    print(f"Successfully upserted {len(split_docs)} dense vectors to Pinecone index '{INDEX_NAME}' in namespace 'qa-namespace'")
except Exception as e:
    print(f"Error upserting to Pinecone: {str(e)}")
    exit(1)

print("Indexing complete! Dense embeddings successfully upserted to Pinecone.")