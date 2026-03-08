import numpy as np
import pandas as pd
import faiss
import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_PATH = ROOT_DIR / "data" / "processed" / "corpus.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from app.embeddings.embedder import Embedder
from app.clustering.fuzzy_cluster import FuzzyCluster

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

documents = df["text"].tolist()

print("Total documents:", len(documents))

embedder = Embedder()

print("Generating embeddings...")

embeddings = embedder.embed_documents(documents)

embeddings = np.array(embeddings).astype("float32")
np.save(MODELS_DIR / "embeddings.npy", embeddings)

print("Embeddings saved")

print("Building FAISS index...")

dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, str(MODELS_DIR / "faiss.index"))

print("FAISS index saved")

print("Training clustering model...")

cluster = FuzzyCluster(n_clusters=20)
cluster.fit(embeddings)

with open(MODELS_DIR / "cluster_model.pkl", "wb") as f:
    pickle.dump(cluster, f)

print("Cluster model saved")

with open(MODELS_DIR / "documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Documents saved")

print("Index build complete")
