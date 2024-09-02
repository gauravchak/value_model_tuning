"""
This script contains the implementation of the regret targeting algorithm.

To run the script, use the following command:

```bash
python regret_targeting.py
```
"""

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg_score


# pylint: disable=W0621
def optimize_weights(
    data: list[list[float]],
    desired_regret_fractions: list[float],
    max_iterations: int = 1000,
    learning_rate: float = 0.1,
    tolerance: float = 1e-6,
    verbose: int = 0,
    initial_weights: list[float] | None = None
):
    """
    Optimize weights to achieve desired regret fractions across multiple tasks.

    This function uses an iterative approach to find weights that produce a ranking
    with regret fractions (1 - nDCG) matching the desired fractions for each task.

    Parameters:
    data (list[list[float]]): A 2D list or array where each row represents an item
                              and each column represents a task. Values are scores
                              or relevance measures for each item-task pair.
    desired_regret_fractions (list[float]): A list of desired regret fractions for
                                            each task. Must sum to 1.
    max_iterations (int, optional): Maximum number of optimization iterations.
                                    Defaults to 1000.
    learning_rate (float, optional): Step size for weight updates in each iteration.
                                     Defaults to 0.1.
    tolerance (float, optional): Convergence threshold for differences between
                                 current and desired regret fractions.
                                 Defaults to 1e-6.
    initial_weights (list[float] | None, optional): Initial weights for tasks.
                                                    If None, starts with equal weights.
                                                    Defaults to None.

    Returns:
    numpy.ndarray: Optimized weights for each task.

    Raises:
    AssertionError: If the number of tasks doesn't match the number of desired
                    regret fractions, or if desired regret fractions don't sum to 1.

    Notes:
    - The function prints intermediate results every 10% of max_iterations iterations
      and final results at the end of optimization.
    - Optimization stops if max_iterations is reached or if the difference between
      current and desired regret fractions is below the tolerance for all tasks.
    """
    data = np.array(data)
    N, T = data.shape
    desired_regret_fractions = np.array(desired_regret_fractions)

    assert len(desired_regret_fractions) == T, (
        "Number of tasks and desired regret fractions must match"
    )
    assert np.isclose(sum(desired_regret_fractions), 1), (
        "Desired regret fractions must sum to 1"
    )

    if initial_weights is None:
        weights = np.ones(T) / T  # Start with equal weights
    else:
        weights = np.array(initial_weights)

    # Compute base rankings for each task
    base_rankings = [np.argsort(data[:, j])[::-1] for j in range(T)]

    for iteration in range(max_iterations):
        # Compute current ranking
        current_scores = np.dot(data, weights)
        current_ranking = np.argsort(current_scores)[::-1]

        # Compute regrets (1 - ndcg) for each task
        regrets = []
        for j in range(T):
            ndcg = sklearn_ndcg_score(
                y_true=[data[:, j][base_rankings[j]]],
                y_score=[data[:, j][current_ranking]]
            )
            regrets.append(1 - ndcg)

        # Compute current regret fractions
        current_regret_fractions = np.array(regrets) / np.maximum(
            0.001, np.sum(regrets)
        )

        # Compute the difference and update weights
        diff = desired_regret_fractions - current_regret_fractions
        # If the difference is negative, the current regret fraction is
        # too high. Hence the current ranking is not giving enough weight
        # to this particular task. So we need to increase the weight of
        # this task.
        # Hence we subtract the difference from the current weights.
        weights -= learning_rate * diff

        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(weights, 0)
        weights /= np.sum(weights)

        # Print debug information every 100 iterations
        if verbose >= 2:
            if iteration % max(1, max_iterations // 10) == 0:
                print(f"Iteration {iteration}:")
                print(
                    "  Current regret fractions:",
                    [f"{frac:.3f}" for frac in current_regret_fractions]
                )
                print("  Updated weights:",
                      [f"{weight:.3f}" for weight in weights])
                print("  Differences:", [f"{d:.3f}" for d in diff])

        # Check for convergence
        if np.all(np.abs(diff) < tolerance):
            print(f"Converged after {iteration + 1} iterations")
            break

    # Final debug information
    if verbose >= 1:
        print("\nFinal results:")
        print(
            "Regrets for each task:",
            [f"{regret:.3f}" for regret in regrets]
        )
        print(
            "Current regret fractions:",
            [f"{frac:.3f}" for frac in current_regret_fractions]
        )
        print(
            "Desired regret fractions:",
            [f"{frac:.3f}" for frac in desired_regret_fractions]
        )
        print("Optimized weights:", [f"{weight:.3f}" for weight in weights])

    return weights


# Example usage:
np.random.seed(42)  # for reproducibility
means = [0.01, 0.02, 0.05, 0.10, 0.15]
data = np.clip(
    np.abs(np.random.normal(loc=means, scale=0.01, size=(100, 5))), 0, 0.99
)
desired_regret_fractions = [0.3, 0.5, 0.07, 0.07, 0.06]
initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Start with equal weights
print("Initial weights:",
      [f"{weight:.3f}" for weight in initial_weights])
optimized_weights = optimize_weights(
    data=data,
    desired_regret_fractions=desired_regret_fractions,
    max_iterations=1000,
    learning_rate=0.03,
    verbose=1,
    initial_weights=initial_weights
)
print("Optimized weights:", [f"{weight:.3f}" for weight in optimized_weights])
