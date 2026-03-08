from fastapi import FastAPI
from app.api.routes import router, set_service

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import VectorStore
from app.cache.semantic_cache import SemanticCache
from app.services.query_service import QueryService

import numpy as np
import faiss
import pickle
import subprocess
import sys
from pathlib import Path

app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

# check if model artifacts exist
models_exist = (
    (MODELS_DIR / "embeddings.npy").exists() and
    (MODELS_DIR / "faiss.index").exists() and
    (MODELS_DIR / "cluster_model.pkl").exists() and
    (MODELS_DIR / "documents.pkl").exists()
)

# if not present, build them automatically
if not models_exist:
    print("Model files not found. Running build_index.py...")
    subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "build_index.py")], cwd=ROOT_DIR, check=True)

print("Loading saved models...")

embeddings = np.load(MODELS_DIR / "embeddings.npy")

index = faiss.read_index(str(MODELS_DIR / "faiss.index"))

with open(MODELS_DIR / "cluster_model.pkl", "rb") as f:
    cluster = pickle.load(f)

with open(MODELS_DIR / "documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("Models loaded")

embedder = Embedder()

vector_store = VectorStore(len(embeddings[0]))
vector_store.index = index
vector_store.documents = documents

cache = SemanticCache()

service = QueryService(embedder, vector_store, cache, cluster)

set_service(service, cache)

app.include_router(router)

print("Application startup complete")
