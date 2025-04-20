# autoencoder_novelty.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
import numpy as np

def autoencoder_novelty_score(reference_texts, test_text):
    texts = reference_texts + [test_text]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray()

    X_train = vectors[:-1]
    X_test = vectors[-1].reshape(1, -1)

    model = MLPRegressor(hidden_layer_sizes=(20,), max_iter=300, random_state=0)
    model.fit(X_train, X_train)
    reconstruction = model.predict(X_test)
    error = np.linalg.norm(X_test - reconstruction)
    return error

if __name__ == "__main__":
    corpus = [
        "Quantum entanglement connects particles over distance.",
        "Fourier transforms decompose signals into frequencies.",
        "Black holes bend spacetime with intense gravity."
    ]
    test = "Drifting deltas in a fractaverse make a dynamic spacetime."
    score = autoencoder_novelty_score(corpus, test)
    print("Autoencoder novelty score:", score)
