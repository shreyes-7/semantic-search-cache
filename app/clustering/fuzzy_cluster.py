from sklearn.mixture import GaussianMixture

class FuzzyCluster:

    def __init__(self, n_clusters=20):
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied"
        )

    def fit(self, embeddings):
        self.model.fit(embeddings)

    def predict_distribution(self, embedding):
        return self.model.predict_proba([embedding])[0]