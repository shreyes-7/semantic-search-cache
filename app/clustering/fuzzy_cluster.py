from sklearn.mixture import GaussianMixture


class FuzzyCluster:

    def __init__(self, n_clusters=20):

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied"
        )

    def fit(self, embeddings):

        self.model.fit(embeddings)

    def get_distribution(self, embedding):

        return self.model.predict_proba([embedding])[0]

    def dominant_cluster(self, embedding):

        probs = self.get_distribution(embedding)

        return probs.argmax()