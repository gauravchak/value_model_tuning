"""
This script contains the implementation of the nDCG targeting algorithm.

To run the script, use the following command:

```bash
python ndcg_targeting_v1.py
```
"""

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg_score


# pylint: disable=W0621
def optimize_weights(
    data: list[list[float]],
    desired_ndcg_fractions: list[float],
    max_iterations: int = 1000,
    learning_rate: float = 0.1,
    tolerance: float = 1e-6,
    verbose: int = 1,
    initial_weights: list[float] | None = None
):
    """
    Optimize weights for task predictions to achieve desired nDCG fractions.

    :param data: List of M arrays, each array of size T (predictions for T tasks for M items)
    :param desired_ndcg_fractions: List of T values summing up to 1 (desired nDCG fractions)
    :param max_iterations: Maximum number of iterations for the optimization
    :param learning_rate: Learning rate for weight updates
    :param tolerance: Convergence tolerance
    :return: Optimized weights of size T
    """
    data = np.array(data)
    _, T = data.shape
    desired_ndcg_fractions = np.array(desired_ndcg_fractions)

    assert len(desired_ndcg_fractions) == T, (
        "Number of tasks and desired nDCG fractions must match"
    )
    assert np.isclose(sum(desired_ndcg_fractions), 1), (
        "Desired nDCG fractions must sum to 1"
    )

    if initial_weights is None:
        weights = np.ones(T) / T  # Start with equal weights
    else:
        weights = np.array(initial_weights)

    for iteration in range(max_iterations):
        # Compute base ranking
        base_scores = np.dot(data, weights)
        base_ranking = np.argsort(base_scores)[::-1]
        # print(f"  Top 3 indices: {base_ranking[:3]}")

        # Compute nDCG between ranking with each task ignored and the base
        # ranking
        ndcgs = []
        for j in range(T):
            task_ignored_weights = weights.copy()
            task_ignored_weights[j] = 0
            task_ignored_scores = np.dot(data, task_ignored_weights)
            task_ignored_ranking = np.argsort(task_ignored_scores)[::-1]
            # print(f" Ignoring {j}:  Top 3 indices: {task_ignored_ranking[:3]}")
            ndcg = sklearn_ndcg_score(
                y_true=[base_scores[base_ranking]],
                y_score=[base_scores[task_ignored_ranking]]
            )
            ndcgs.append(ndcg)

        # Compute nDCG regrets
        ndcg_regrets = 1 - np.array(ndcgs)
        # Compute current fractions based on regrets
        current_ndcg_fractions = ndcg_regrets / np.maximum(
            0.001, np.sum(ndcg_regrets))

        # Compute the difference and update weights
        diff = desired_ndcg_fractions - current_ndcg_fractions
        # If the difference w.r.t. a task is positive, that means the current
        # ndcg gap is less than desired. So we increase the weight for the task
        # since we want to increase the change in rankings that happens when we
        # zero-out the weight for that task.
        weights += learning_rate * diff

        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(weights, 0)
        weights /= np.sum(weights)

        # Print debug information every 1000 iterations
        if verbose >= 2:
            if iteration % max(1, (max_iterations // 10)) == 0:
                print(f"Iteration {iteration}:")
                print("  Current fractions:",
                      [f"{frac:.3f}" for frac in current_ndcg_fractions])
                print("  Updated weights:",
                      [f"{weight:.3f}" for weight in weights])
                print("  Differences:",
                      [f"{frac:.3f}" for frac in diff])

        # Check for convergence
        if np.all(np.abs(diff) < tolerance):
            print(f"Converged after {iteration + 1} iterations")
            break

    # Final debug information
    if verbose >= 1:
        print("\nFinal results:")
        print("  nDCGs for each task ignored:",
              [f"{ndcg:.3f}" for ndcg in ndcgs])
        print("  Current fractions:",
              [f"{frac:.3f}" for frac in current_ndcg_fractions])
        print("  Desired fractions:",
              [f"{frac:.3f}" for frac in desired_ndcg_fractions])
        print("  Optimized weights:",
              [f"{weight:.3f}" for weight in weights])

    return weights


# Example usage:
np.random.seed(42)  # for reproducibility
means = [0.01, 0.02, 0.05, 0.10, 0.15]
data = np.clip(
    np.abs(np.random.normal(loc=means, scale=0.01, size=(100, 5))), 0, 0.99
)
desired_ndcg_fractions = [0.3, 0.5, 0.07, 0.07, 0.06]
initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Start with equal weights
print("Initial weights:",
      [f"{weight:.3f}" for weight in initial_weights])
optimized_weights = optimize_weights(
    data=data,
    desired_ndcg_fractions=desired_ndcg_fractions,
    max_iterations=1000,
    learning_rate=0.05,
    tolerance=1e-5,
    verbose=1,
    initial_weights=initial_weights
)
print("Optimized weights:",
      [f"{weight:.3f}" for weight in optimized_weights])
