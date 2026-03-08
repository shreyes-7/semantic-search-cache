import numpy as np

class SemanticCache:

    def __init__(self, threshold=0.85):
        self.entries = []
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0

    def cosine(self,a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def lookup(self, query_embedding):

        best = None
        best_score = 0

        for entry in self.entries:
            score = self.cosine(query_embedding, entry["embedding"])

            if score > best_score:
                best_score = score
                best = entry

        if best_score > self.threshold:
            self.hit_count += 1
            return True, best, best_score

        self.miss_count += 1
        return False, None, best_score

    def store(self, query, embedding, result):

        self.entries.append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

    def stats(self):

        total = len(self.entries)
        hits = self.hit_count
        misses = self.miss_count

        rate = hits / (hits + misses) if (hits+misses)>0 else 0

        return {
            "total_entries": total,
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate": rate
        }

    def clear(self):
        self.entries = []
        self.hit_count = 0
        self.miss_count = 0