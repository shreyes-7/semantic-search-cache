import faiss
import numpy as np


class VectorStore:

    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add(self, embeddings, docs):

        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)

        self.documents.extend(docs)

    def search(self, vector, k=5):

        D, I = self.index.search(vector.reshape(1, -1).astype("float32"), k)

        results = []

        for idx in I[0]:
            results.append(self.documents[idx])

        return results