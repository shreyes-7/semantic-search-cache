class QueryService:

    def __init__(self, embedder, store, cache, cluster):

        self.embedder = embedder
        self.store = store
        self.cache = cache
        self.cluster = cluster

    def process_query(self, query):

        # embed user query
        embedding = self.embedder.embed_query(query)

        # determine dominant cluster
        dominant_cluster = self.cluster.dominant_cluster(embedding)

        # check semantic cache within this cluster
        hit, entry, score = self.cache.lookup(embedding, dominant_cluster)

        if hit:

            return {
                "query": query,
                "cache_hit": True,
                "matched_query": entry["query"],
                "similarity_score": float(score),
                "result": entry["result"],
                "dominant_cluster": int(dominant_cluster)
            }

        # perform vector search if cache miss
        results = self.store.search(embedding)

        result = results[0]

        # store result in cluster-aware cache
        self.cache.store(query, embedding, result, dominant_cluster)

        return {
            "query": query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": float(score),
            "result": result,
            "dominant_cluster": int(dominant_cluster)
        }