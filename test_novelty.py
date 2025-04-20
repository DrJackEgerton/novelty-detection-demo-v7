# test_novelty.py
from vector_quality_detector import vector_novelty_score
from autoencoder_novelty import autoencoder_novelty_score
from statistical_anomaly import statistical_novelty_score

def compare_novelty_methods(reference_texts, test_text):
    print("Test text:", test_text)
    print("")

    vector_score = vector_novelty_score(reference_texts, test_text)
    print("?? Vector Similarity Novelty Score:       {:.4f}".format(vector_score))

    auto_score = autoencoder_novelty_score(reference_texts, test_text)
    print("?? Autoencoder Reconstruction Error:       {:.4f}".format(auto_score))

    stat_score = statistical_novelty_score(reference_texts, test_text)
    print("?? Statistical (Mahalanobis) Distance:     {:.4f}".format(stat_score))

if __name__ == "__main__":
    corpus = [
        "Quantum entanglement connects particles over distance.",
        "Fourier transforms decompose signals into frequencies.",
        "Black holes bend spacetime with intense gravity."
    ]
    test_sentence = "Drifting deltas in a fractaverse make a dynamic spacetime."

    compare_novelty_methods(corpus, test_sentence)
