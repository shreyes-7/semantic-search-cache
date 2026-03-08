class QueryService:

    def __init__(self, embedder, store, cache, cluster):

        self.embedder = embedder
        self.store = store
        self.cache = cache
        self.cluster = cluster

    def process_query(self, query):

        embedding = self.embedder.embed_query(query)

        hit, entry, score = self.cache.lookup(embedding)

        dominant_cluster = self.cluster.dominant_cluster(embedding)

        if hit:

            return {
                "query": query,
                "cache_hit": True,
                "matched_query": entry["query"],
                "similarity_score": float(score),
                "result": entry["result"],
                "dominant_cluster": int(dominant_cluster)
            }

        results = self.store.search(embedding)

        result = results[0]

        self.cache.store(query, embedding, result)

        return {
            "query": query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": float(score),
            "result": result,
            "dominant_cluster": int(dominant_cluster)
        }