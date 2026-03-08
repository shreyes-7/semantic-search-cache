import numpy as np
import pandas as pd
import faiss
import pickle

from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import VectorStore
from app.clustering.fuzzy_cluster import FuzzyCluster

print("Loading dataset...")

df = pd.read_csv("data/processed/corpus.csv")

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

documents = df["text"].tolist()

print("Total documents:", len(documents))

embedder = Embedder()

print("Generating embeddings...")

embeddings = embedder.embed_documents(documents)

embeddings = np.array(embeddings).astype("float32")

np.save("models/embeddings.npy", embeddings)

print("Embeddings saved")

print("Building FAISS index...")

dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "models/faiss.index")

print("FAISS index saved")

print("Training clustering model...")

cluster = FuzzyCluster(n_clusters=20)
cluster.fit(embeddings)

with open("models/cluster_model.pkl", "wb") as f:
    pickle.dump(cluster, f)

print("Cluster model saved")

with open("models/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Documents saved")

print("Index build complete")