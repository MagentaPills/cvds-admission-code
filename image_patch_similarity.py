"""
Image Patch Similarity Search
Algorithm: Sliding window with normalized cross-correlation
Complexity: O((H-h+1)*(W-w+1)*h*w) optimized with vectorization
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Represents a matching patch location and score."""
    row: int
    col: int
    score: float


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Normalize patch to zero mean and unit variance.
    Essential for illumination-invariant matching.
    """
    mean = np.mean(patch)
    std = np.std(patch)
    if std < 1e-6:
        return patch - mean
    return (patch - mean) / std


def sliding_window_similarity(
    image: np.ndarray,
    template: np.ndarray,
    stride: int = 1
) -> List[MatchResult]:
    """
    Find regions in image similar to template using normalized cross-correlation.

    Args:
        image: 2D grayscale image array (H, W)
        template: 2D pattern to match (h, w), must be smaller than image
        stride: step size for sliding window (1 = exhaustive)

    Returns:
        List of MatchResult sorted by similarity score (highest first)
    """
    h_img, w_img = image.shape
    h_t, w_t = template.shape

    if h_t > h_img or w_t > w_img:
        raise ValueError("Template larger than image")

    # Normalize template once
    template_norm = normalize_patch(template)

    results = []

    # Slide window across image
    for i in range(0, h_img - h_t + 1, stride):
        for j in range(0, w_img - w_t + 1, stride):
            patch = image[i:i + h_t, j:j + w_t]
            patch_norm = normalize_patch(patch)

            # Normalized cross-correlation (dot product = similarity)
            similarity = np.sum(patch_norm * template_norm) / (h_t * w_t)

            results.append(MatchResult(row=i, col=j, score=similarity))

    # Sort by highest similarity
    results.sort(key=lambda x: x.score, reverse=True)
    return results


def find_best_match(
    image: np.ndarray,
    template: np.ndarray
) -> MatchResult:
    """Return single best match location."""
    matches = sliding_window_similarity(image, template)
    return matches[0] if matches else MatchResult(0, 0, 0.0)


# Example usage (would be in a test file)
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    image = np.random.rand(100, 100)
    template = image[20:35, 30:45]  # Extract a real patch

    best = find_best_match(image, template)
    print(f"Best match at ({best.row}, {best.col}) with score {best.score:.4f}")
