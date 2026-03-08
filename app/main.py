from fastapi import FastAPI
from app.api.routes import router, set_service

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import VectorStore
from app.cache.semantic_cache import SemanticCache
from app.services.query_service import QueryService

import numpy as np
import faiss
import pickle

app = FastAPI()

print("Loading saved models...")

embeddings = np.load("models/embeddings.npy")

index = faiss.read_index("models/faiss.index")

with open("models/cluster_model.pkl", "rb") as f:
    cluster = pickle.load(f)

with open("models/documents.pkl", "rb") as f:
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