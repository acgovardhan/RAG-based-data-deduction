# backend/main.py
import os
import io
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
import pdfplumber

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# ----------------------------
# Paths and config
# ----------------------------
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K_DEFAULT = 4

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY") 

if not OPENROUTER_KEY:
    raise ValueError("OPENROUTER_KEY not set!")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "xiaomi/mimo-v2-flash:free"

# ----------------------------
# FastAPI app + CORS
# ----------------------------
app = FastAPI(title="Kerala Farmer Trainings - RAG Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo. For prod, restrict origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Data models
# ----------------------------
class IngestItem(BaseModel):
    url: str
    source: Optional[str] = None
    training_type: Optional[str] = None
    district: Optional[str] = None
    date: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    district: Optional[str] = None
    training_type: Optional[str] = None
    recent_days: Optional[int] = None
    top_k: Optional[int] = TOP_K_DEFAULT
    language: Optional[str] = "ml"

# ----------------------------
# Globals: vector index & chunks
# ----------------------------
chunks: List[Dict[str, Any]] = []
index: Optional[faiss.IndexFlatL2] = None
print("Loading embedding model (this may download on first run)...")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

# ----------------------------
# Utilities: persist / chunk / scrape
# ----------------------------
def save_state():
    global index, chunks
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    if index is not None:
        faiss.write_index(index, INDEX_PATH)
    print("Saved state: chunks:", len(chunks))

def load_state():
    global index, chunks
    try:
        if os.path.exists(CHUNKS_PATH) and os.path.exists(INDEX_PATH):
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
            index = faiss.read_index(INDEX_PATH)
            print("Loaded state: chunks:", len(chunks))
        else:
            chunks = []
            index = faiss.IndexFlatL2(EMBED_DIM)
            print("Initialized new index")
    except Exception as e:
        print("Failed to load state, initializing new index:", e)
        chunks = []
        index = faiss.IndexFlatL2(EMBED_DIM)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    text = text.replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            out.append(p)
        else:
            start = 0
            while start < len(p):
                end = start + chunk_size
                out.append(p[start:end])
                start = end - overlap
    return out

def scrape_html_text(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        s.extract()
    text = soup.get_text(separator="\n\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n\n".join([ln for ln in lines if ln])

def extract_pdf_text_from_bytes(b: bytes):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text()
                if ptext:
                    text += "\n\n" + ptext
    except Exception as e:
        print("pdfplumber failed:", e)
    return text.strip()

def download_url(url: str) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.content

def add_document_text(text: str, metadata: dict):
    """
    Chunk -> embed -> add to faiss & chunks list
    """
    global index, chunks, embedding_model
    texts = chunk_text(text)
    if not texts:
        return 0
    embs = embedding_model.encode(texts, show_progress_bar=False)
    if index is None:
        index = faiss.IndexFlatL2(len(embs[0]))
    index.add(np.array(embs, dtype="float32"))
    base = len(chunks)
    now = datetime.utcnow().isoformat()
    for i, t in enumerate(texts):
        chunks.append({
            "id": base + i,
            "text": t,
            "metadata": {**metadata, "scraped_at": now}
        })
    return len(texts)

def ensure_index_ok():
    global index
    if index is None:
        index = faiss.IndexFlatL2(EMBED_DIM)

def filter_metadata_item(meta: dict, filters: dict):
    if not filters:
        return True
    if filters.get("district"):
        if not meta.get("district") or filters["district"].strip().lower() not in str(meta.get("district")).lower():
            return False
    if filters.get("training_type"):
        if not meta.get("training_type") or filters["training_type"].strip().lower() not in str(meta.get("training_type")).lower():
            return False
    if filters.get("recent_days"):
        recent_days = filters["recent_days"]
        date_str = meta.get("date") or meta.get("scraped_at")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                cutoff = datetime.utcnow() - timedelta(days=recent_days)
                if dt < cutoff:
                    return False
            except Exception:
                pass
    return True

def retrieve_relevant(question: str, top_k: int = 4, district: Optional[str] = None, training_type: Optional[str] = None, recent_days: Optional[int] = None):
    ensure_index_ok()
    if len(chunks) == 0:
        return []
    q_emb = embedding_model.encode([question])
    qv = np.array(q_emb, dtype="float32")
    search_k = max(top_k * 6, top_k + 10)
    n_vectors = index.ntotal
    if n_vectors == 0:
        return []
    k = min(search_k, n_vectors)
    distances, indices = index.search(qv, k)
    selected = []
    filters = {"district": district, "training_type": training_type, "recent_days": recent_days}
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0: 
            continue
        c = chunks[idx]
        if filter_metadata_item(c["metadata"], filters):
            selected.append({"chunk": c, "distance": float(dist)})
            if len(selected) >= top_k:
                break
    return selected

def build_prompt_with_context(chunks_list: List[Dict[str, Any]], question: str, language: str = "ml"):
    header = "താഴെ കൊടുത്ത വിവരങ്ങൾ മാത്രം ഉപയോഗിച്ച് ചോദ്യത്തിനു മറുപടി പറയുക. വിവരങ്ങൾ ഇല്ലെങ്കിൽ 'No training found.' എന്ന് മറുപടി നൽകുക. ചോദ്യം ചുരുക്കമായി മറുപടി ചെയ്യുക." if language=="ml" else "Use only the provided information to answer the question. If answer not found, reply 'No training found.'"
    context_parts = []
    sources = []
    for i, it in enumerate(chunks_list):
        c = it["chunk"]
        meta = c["metadata"]
        src = {
            "source": meta.get("source"),
            "url": meta.get("url"),
            "training_type": meta.get("training_type"),
            "district": meta.get("district"),
            "date": meta.get("date") or meta.get("scraped_at")
        }
        sources.append(src)
        context_parts.append(f"--- SOURCE {i+1} ---\n{c['text']}\n[meta: {json.dumps(src, ensure_ascii=False)}]\n")
    prompt = f"{header}\n\nCONTEXT:\n{'\n\n'.join(context_parts)}\n\nQUESTION:\n{question}\n\nAnswer shortly."
    return prompt, sources

def call_openrouter(prompt: str, language: str = "ml"):
    if not OPENROUTER_KEY:
        raise Exception("OPENROUTER_KEY not set in environment variables.")
    messages = [
        {"role": "system", "content": "You are an expert farmer in Kerala who answers in Malayalam." if language=="ml" else "You are an expert farmer."},
        {"role": "user", "content": prompt}
    ]
    payload = {"model": OPENROUTER_MODEL, "messages": messages, "reasoning": {"enabled": True}}
    headers = {"Authorization": "Bearer " + OPENROUTER_KEY, "Content-Type": "application/json"}
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"OpenRouter error: {resp.status_code} {resp.text}")
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(data, ensure_ascii=False)[:2000]
    return text, data

# ----------------------------
# Startup load
# ----------------------------
@app.on_event("startup")
def startup_load():
    load_state()

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), source: Optional[str] = Form(None), training_type: Optional[str] = Form(None), district: Optional[str] = Form(None)):
    """
    Upload a PDF or text file. Will extract text and ingest into FAISS.
    """
    content = await file.read()
    if file.filename.lower().endswith(".pdf") or b"%PDF" in content[:4]:
        text = extract_pdf_text_from_bytes(content)
    else:
        text = content.decode("utf-8", errors="ignore")
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from file")
    meta = {
        "source": source or file.filename,
        "url": "",
        "training_type": training_type or "Unknown",
        "district": district or "",
        "date": datetime.utcnow().isoformat()
    }
    n = add_document_text(text, meta)
    save_state()
    return {"message": f"{file.filename} ingested -> {n} chunks", "total_chunks": len(chunks)}

@app.post("/ingest_url")
def ingest_url(item: IngestItem):
    try:
        content = download_url(item.url)
        if item.url.lower().endswith(".pdf") or b"%PDF" in content[:1024]:
            text = extract_pdf_text_from_bytes(content)
        else:
            text = scrape_html_text(content.decode("utf-8", errors="ignore"))
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from URL")
        meta = {"source": item.source or item.url.split("/")[2], "url": item.url, "training_type": item.training_type or "Unknown", "district": item.district or "", "date": item.date or ""}
        n = add_document_text(text, meta)
        save_state()
        return {"added": n, "total_chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/create_dummy")
def create_dummy():
    """
    Quickly create and ingest a small fake training document to test RAG.
    """
    text = (
        "Kerala Farmer Training Programs:\n\n"
        "1) Krishi Bhavan seasonal workshops: free short workshops run seasonally across districts.\n"
        "2) KVK Skill Development: three-day hands-on programs on vegetable cultivation and pest management.\n"
        "3) NGO organic training: community workshops for small farmers on organic methods.\n"
        "4) Paid certification: private providers offering online & offline certification courses.\n"
    )
    meta = {"source": "DemoDoc", "url": "", "training_type": "General", "district": "Kerala", "date": datetime.utcnow().isoformat()}
    n = add_document_text(text, meta)
    save_state()
    return {"message": f"Dummy doc added with {n} chunks", "total_chunks": len(chunks)}

@app.post("/ask")
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question required")
    top_k = req.top_k or TOP_K_DEFAULT
    selected = retrieve_relevant(req.question, top_k=top_k, district=req.district, training_type=req.training_type, recent_days=req.recent_days)
    if not selected:
        return {"answer": "No training found.", "sources": [], "used_chunks_count": 0}
    prompt, sources = build_prompt_with_context(selected, req.question, language=(req.language or "ml"))
    # call LLM
    try:
        assistant_text, openrouter_raw = call_openrouter(prompt, language=(req.language or "ml"))
    except Exception as e:
        # helpful fallback: return context text for debugging
        debug_text = "\n\n".join([c["chunk"]["text"] for c in selected])
        return {"answer": f"OpenRouter error: {e}. Context:\n\n{debug_text}", "sources": sources, "used_chunks_count": len(selected)}
    return {"answer": assistant_text, "sources": sources, "used_chunks_count": len(selected), "openrouter_raw": openrouter_raw}

@app.get("/stats")
def stats():
    return {"chunks": len(chunks), "index_vectors": index.ntotal if index is not None else 0}

@app.get("/save")
def save_endpoint():
    save_state()
    return {"status": "saved", "chunks": len(chunks)}
