# statistical_anomaly.py
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import mahalanobis
import numpy as np

def statistical_novelty_score(reference_texts, test_text):
    texts = reference_texts + [test_text]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray()

    X_train = vectors[:-1]
    X_test = vectors[-1]

    mean_vec = np.mean(X_train, axis=0)
    cov = np.cov(X_train, rowvar=False)
    try:
        inv_cov = np.linalg.pinv(cov)
        dist = mahalanobis(X_test, mean_vec, inv_cov)
    except np.linalg.LinAlgError:
        dist = 0
    return dist

if __name__ == "__main__":
    corpus = [
        "Quantum entanglement connects particles over distance.",
        "Fourier transforms decompose signals into frequencies.",
        "Black holes bend spacetime with intense gravity."
    ]
    test = "Drifting deltas in a fractaverse make a dynamic spacetime."
    score = statistical_novelty_score(corpus, test)
    print("Statistical novelty score:", score)
