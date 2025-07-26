from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pdfplumber
import re
from typing import List
import os
from dotenv import load_dotenv

import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")



if not all([
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX_NAME,

]):
    raise RuntimeError("Missing environment variables. Check your .env file.")

# Pinecone init
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)

app = FastAPI(title="PDF Embedder with Pinecone and Azure OpenAI Embeddings")

class EmbedRequest(BaseModel):
    pdf_path: str

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^অ-হ০-৯a-zA-Z0-9.,;:?!\'"()\- ]+', '', text)
    return text

def load_pdf_text(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading PDF: {e}")

def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    return docs

# Azure OpenAI Embeddings init
embeddings = AzureOpenAIEmbeddings(
    deployment=os.getenv("azure_deployment"),
    api_key=os.getenv("api_key"),
    AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("azure_endpoint"),
    
)

def embed_and_store(docs: List[Document]):
    texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)

    ids = [f"doc-{i}" for i in range(len(vectors))]

    to_upsert = []
    for i, vector in enumerate(vectors):
        meta = {"text": texts[i]}
        to_upsert.append((ids[i], vector, meta))

    index.upsert(vectors=to_upsert)

@app.post("/embed/")
async def embed_document(req: EmbedRequest):
    raw_text = load_pdf_text(req.pdf_path)
    cleaned_text = clean_text(raw_text)
    docs = chunk_text(cleaned_text)
    embed_and_store(docs)
    return {
        "message": f"Embedding and upload completed successfully. Total chunks: {len(docs)}",
        "cleaned_text_sample": cleaned_text[:500]
    }
