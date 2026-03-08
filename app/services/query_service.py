class QueryService:

    def __init__(self, embedder, store, cache, cluster):

        self.embedder = embedder
        self.store = store
        self.cache = cache
        self.cluster = cluster

    def process_query(self, query):

        q_emb = self.embedder.embed_query(query)

        hit, entry, score = self.cache.lookup(q_emb)

        if hit:

            return {
                "query": query,
                "cache_hit": True,
                "matched_query": entry["query"],
                "similarity_score": float(score),
                "result": entry["result"]
            }

        results = self.store.search(q_emb)

        result_text = results[0][0]

        self.cache.store(query, q_emb, result_text)

        return {
            "query": query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": float(score),
            "result": result_text
        }