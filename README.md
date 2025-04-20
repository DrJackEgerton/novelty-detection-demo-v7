# Simple Novelty Detection Toolkit (Python 3.6 Compatible)

This repository provides lightweight, interpretable novelty (or "quality") detection tools, implemented with Python 3.6 compatibility in mind. The accompanying website (hosted via GitHub Pages) showcases simple methods to help demonstrate functionality to non-specialists.

## ?? Features

- **PCA Reconstruction Error**: Measures how well data can be represented in a lower-dimensional space.
- **One-Class SVM**: Learns the frontier of "normal" data in a high-dimensional space.
- **Isolation Forest**: Detects anomalies by how easily data points are isolated in decision trees.

Each method is implemented in a minimal, self-contained Python file with clear comments.

## ?? File Structure

- `pca_quality.py`: PCA-based novelty detection.
- `svm_quality.py`: One-Class SVM novelty detection.
- `isolation_forest_quality.py`: Isolation Forest novelty detection.
- `index.html`: Minimal static website for demonstration or visualization.
- `README.md`: This file.

## ?? Requirements

- Python 3.6
- `scikit-learn` (<= version 0.22 recommended for 3.6)
- `numpy`
- `matplotlib` (optional, for plots)

To install dependencies:
```bash
pip install scikit-learn==0.22 numpy matplotlib
