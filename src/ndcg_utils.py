"""
This script contains utility functions for computing nDCG.

To run the script, use the following command:

```bash
python ndcg_utils.py
```
"""

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg_score


def dcg_at_k(r, k):
    """
    Compute Discounted Cumulative Gain (DCG) at rank k.

    Parameters:
    r (array-like): Relevance scores in rank order
    k (int): Number of results to consider

    Returns:
    float: DCG@k
    """
    r = np.asarray(r, dtype=float)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def custom_ndcg_score(y_true, y_score, k=None):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) score.

    Parameters:
    y_true (array-like): True relevance scores
    y_score (array-like): Predicted scores
    k (int, optional): Number of results to consider. If None, uses all results.

    Returns:
    float: nDCG score
    """
    if k is None:
        k = len(y_true)
    # Sort the predicted scores in descending order to get the ranking
    order = np.argsort(y_score)[::-1]
    # Take the top k elements from the true relevance scores
    y_true = np.take(y_true, order[:k])
    # Calculate the best possible DCG at rank k
    best = dcg_at_k(sorted(y_true, reverse=True), k)
    actual = dcg_at_k(y_true, k)
    return actual / best if best > 0 else 0


# Example case
relevance_scores = np.array([1, 2, 3])  # True relevance scores (higher is more relevant)
predicted_scores = np.array([3, 10, 5])  # Predicted relevance scores by the model
ndcg_k: int = 3

# Explanation of the example:
# We have 3 items with true relevance scores [1, 2, 3]
# The model predicted scores [3, 10, 5] for these items
# The goal is to see how well the predicted scores rank the items compared to
# their true relevance

# Calculate nDCG score using custom implementation
custom_result = custom_ndcg_score(
    y_true=relevance_scores,
    y_score=predicted_scores,
    k=ndcg_k,
)
print(f"Custom nDCG score: {custom_result:.3f}")

# Calculate nDCG score using sklearn implementation
sklearn_result = sklearn_ndcg_score(
    y_true=[relevance_scores],
    y_score=[predicted_scores],
    k=ndcg_k,
)
print(f"sklearn nDCG score: {sklearn_result:.3f}")

# Print intermediate steps for clarity
ranking_order = np.argsort(predicted_scores)[::-1]
print(f"Ranking order based on predicted scores: {ranking_order}")
print(
    "True relevance scores in predicted ranking order: " +
    f"{relevance_scores[ranking_order]}"
)

# Verify that both implementations give the same result
assert np.isclose(custom_result, sklearn_result), (
    "Custom and sklearn implementations differ"
)
print("Custom and sklearn implementations give equivalent results.")
