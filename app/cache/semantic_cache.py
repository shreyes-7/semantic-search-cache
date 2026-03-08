import numpy as np


class SemanticCache:

    def __init__(self, threshold=0.85):

        # cache grouped by cluster
        self.cache = {}

        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0

    def cosine(self, a, b):

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def lookup(self, query_embedding, cluster_id):

        # if cluster has no cache entries
        if cluster_id not in self.cache:
            self.miss_count += 1
            return False, None, 0

        entries = self.cache[cluster_id]

        best = None
        best_score = 0

        for entry in entries:

            score = self.cosine(query_embedding, entry["embedding"])

            if score > best_score:
                best_score = score
                best = entry

        if best_score > self.threshold:

            self.hit_count += 1
            return True, best, best_score

        self.miss_count += 1
        return False, None, best_score

    def store(self, query, embedding, result, cluster_id):

        if cluster_id not in self.cache:
            self.cache[cluster_id] = []

        self.cache[cluster_id].append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

    def stats(self):

        total_entries = sum(len(v) for v in self.cache.values())

        hits = self.hit_count
        misses = self.miss_count

        rate = hits / (hits + misses) if hits + misses > 0 else 0

        return {
            "total_entries": total_entries,
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate": rate
        }

    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0