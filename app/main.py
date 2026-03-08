from fastapi import FastAPI
from app.api.routes import router, set_service

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import VectorStore
from app.cache.semantic_cache import SemanticCache
from app.clustering.fuzzy_cluster import FuzzyCluster
from app.services.query_service import QueryService

import pandas as pd

app = FastAPI()

print("Loading dataset...")

df = pd.read_csv("data/processed/corpus.csv")

# Remove NaN values
df = df.dropna(subset=["text"])

# Ensure all values are strings
df["text"] = df["text"].astype(str)

documents = df["text"].tolist()

print("Total documents:", len(documents))

embedder = Embedder()

print("Generating embeddings...")

embeddings = embedder.embed_documents(documents)

print("Building vector store...")

vector_store = VectorStore(len(embeddings[0]))
vector_store.add(embeddings, documents)

print("Training clustering model...")

cluster = FuzzyCluster(n_clusters=20)
cluster.fit(embeddings)

cache = SemanticCache()

service = QueryService(embedder, vector_store, cache, cluster)

# pass objects to API routes
set_service(service, cache)

app.include_router(router)

print("Application startup complete.")