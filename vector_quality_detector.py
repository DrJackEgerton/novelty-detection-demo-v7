# vector_quality_detector.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def vector_novelty_score(reference_texts, test_text):
    texts = reference_texts + [test_text]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(vectors[-1], vectors[:-1])
    max_sim = max(similarities[0])
    novelty_score = 1 - max_sim
    return novelty_score

if __name__ == "__main__":
    corpus = [
        "Quantum entanglement connects particles over distance.",
        "Fourier transforms decompose signals into frequencies.",
        "Black holes bend spacetime with intense gravity."
    ]
    test = "Drifting deltas in a fractaverse make a dynamic spacetime."
    score = vector_novelty_score(corpus, test)
    print("Vector novelty score:", score)
