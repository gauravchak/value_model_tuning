"""
This script contains the implementation of the nDCG targeting algorithm.

To run the script, use the following command:

```bash
python ndcg_targeting.py
```
"""

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg_score
from scipy.optimize import minimize


# pylint: disable=W0621

def objective_function(
    weights: np.ndarray,
    data: np.ndarray,
    desired_ndcg_fractions: np.ndarray
) -> float:
    T = len(weights)
    base_scores = np.dot(data, weights)
    base_ranking = np.argsort(base_scores)[::-1]

    ndcgs = []
    for j in range(T):
        task_ignored_weights = weights.copy()
        task_ignored_weights[j] = 0
        task_ignored_scores = np.dot(data, task_ignored_weights)
        task_ignored_ranking = np.argsort(task_ignored_scores)[::-1]
        ndcg = sklearn_ndcg_score(
            y_true=[base_scores[base_ranking]],
            y_score=[task_ignored_scores[task_ignored_ranking]]
        )
        ndcgs.append(ndcg)

    ndcg_regrets = 1 - np.array(ndcgs)
    current_ndcg_fractions = ndcg_regrets / np.maximum(
        1e-4, np.sum(ndcg_regrets))

    return np.sum((desired_ndcg_fractions - current_ndcg_fractions) ** 2)


def optimize_weights_slsqp(
    data: list[list[float]],
    desired_ndcg_fractions: list[float],
    initial_weights: list[float] | None = None
):
    """
    Optimize weights for multiple tasks to achieve desired nDCG fractions using SLSQP.

    This function uses the Sequential Least Squares Programming (SLSQP) algorithm to
    optimize task weights. The goal is to minimize the difference between the current
    nDCG fractions and the desired nDCG fractions across all tasks.

    Args:
        data (list[list[float]]): A 2D list where each inner list represents a data point,
                                  and each element represents the score for a task.
        desired_ndcg_fractions (list[float]): A list of desired nDCG fractions for each task.
                                              Must sum to 1.
        initial_weights (list[float] | None, optional): Initial weights for optimization.
                                                        If None, equal weights are used.

    Returns:
        numpy.ndarray: Optimized weights for each task.

    Raises:
        AssertionError: If the number of tasks doesn't match the number of desired nDCG fractions,
                        or if the desired nDCG fractions don't sum to 1.

    Note:
        - The function prints final results including nDCG regrets, current fractions,
          desired fractions, optimized weights, and optimization status.
        - The optimization is constrained so that weights sum to 1 and are between 0 and 1.

    Example:
        >>> data = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.1], [0.3, 0.1, 0.2]]
        >>> desired_fractions = [0.4, 0.3, 0.3]
        >>> optimize_weights_slsqp(data=data, desired_fractions=desired_fractions)
    """
    data = np.array(data)
    _, num_tasks = data.shape
    desired_ndcg_fractions = np.array(desired_ndcg_fractions)

    assert len(desired_ndcg_fractions) == num_tasks, (
        "Number of tasks and desired nDCG fractions must match"
    )
    assert np.isclose(sum(desired_ndcg_fractions), 1), (
        "Desired nDCG fractions must sum to 1"
    )

    if initial_weights is None:
        initial_weights = np.ones(num_tasks) / num_tasks  # equal weights
    else:
        initial_weights = np.array(initial_weights)

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # sum of weights = 1
    )
    bounds = [(0, 1) for _ in range(num_tasks)]  # weights between 0 and 1

    result = minimize(
        objective_function,
        initial_weights,
        args=(data, desired_ndcg_fractions),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-10,
        options={'maxiter': 1000}
    )

    # pylint: disable=W0621
    optimized_weights = result.x

    # Compute final nDCG regrets and fractions
    base_scores = np.dot(data, optimized_weights)
    base_ranking = np.argsort(base_scores)[::-1]
    ndcgs = []
    for j in range(num_tasks):
        task_ignored_weights = optimized_weights.copy()
        task_ignored_weights[j] = 0
        task_ignored_scores = np.dot(data, task_ignored_weights)
        task_ignored_ranking = np.argsort(task_ignored_scores)[::-1]
        ndcg = sklearn_ndcg_score(
            y_true=[base_scores[base_ranking]],
            y_score=[base_scores[task_ignored_ranking]]
        )
        ndcgs.append(ndcg)

    ndcg_regrets = 1 - np.array(ndcgs)
    current_ndcg_fractions = ndcg_regrets / np.maximum(
        1e-4, np.sum(ndcg_regrets))

    # Print final results
    print("\nFinal results:")
    print("nDCG regrets for each task ignored:",
          [f"{regret:.3f}" for regret in ndcg_regrets])
    print("Current fractions:",
          [f"{frac:.3f}" for frac in current_ndcg_fractions])
    print("Desired fractions:",
          [f"{frac:.3f}" for frac in desired_ndcg_fractions])
    print("Optimized weights:",
          [f"{weight:.3f}" for weight in optimized_weights])
    print("Optimization success:", result.success)
    print("Optimization message:", result.message)

    return optimized_weights


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
optimized_weights = optimize_weights_slsqp(
    data=data,
    desired_ndcg_fractions=desired_ndcg_fractions,
    initial_weights=initial_weights
)
print("Optimized weights:",
      [f"{weight:.3f}" for weight in optimized_weights])
