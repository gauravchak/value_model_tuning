"""
This script contains the implementation of the nDCG targeting algorithm.
"""

import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg_score
from scipy.optimize import minimize


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
            [base_scores[base_ranking]], [base_scores[task_ignored_ranking]])
        ndcgs.append(ndcg)

    ndcg_regrets = 1 - np.array(ndcgs)
    current_ndcg_fractions = ndcg_regrets / np.maximum(
        0.001, np.sum(ndcg_regrets))

    return np.sum((desired_ndcg_fractions - current_ndcg_fractions) ** 2)


def optimize_weights_slsqp(
    data: list[list[float]],
    desired_ndcg_fractions: list[float],
    initial_weights: list[float] | None = None
):
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
        initial_weights = np.ones(T) / T  # Start with equal weights
    else:
        initial_weights = np.array(initial_weights)

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # sum of weights = 1
    )
    bounds = [(0, 1) for _ in range(T)]  # weights between 0 and 1

    result = minimize(
        objective_function,
        initial_weights,
        args=(data, desired_ndcg_fractions),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'maxiter': 1000}
    )

    optimized_weights = result.x

    # Compute final nDCG regrets and fractions
    base_scores = np.dot(data, optimized_weights)
    base_ranking = np.argsort(base_scores)[::-1]
    ndcgs = []
    for j in range(T):
        task_ignored_weights = optimized_weights.copy()
        task_ignored_weights[j] = 0
        task_ignored_scores = np.dot(data, task_ignored_weights)
        task_ignored_ranking = np.argsort(task_ignored_scores)[::-1]
        ndcg = sklearn_ndcg_score(
            [base_scores[base_ranking]], [base_scores[task_ignored_ranking]])
        ndcgs.append(ndcg)

    ndcg_regrets = 1 - np.array(ndcgs)
    current_ndcg_fractions = ndcg_regrets / np.maximum(
        0.001, np.sum(ndcg_regrets))

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
optimized_weights = optimize_weights_slsqp(
    data=data,
    desired_ndcg_fractions=desired_ndcg_fractions,
    initial_weights=[0.2, 0.2, 0.2, 0.2, 0.2]  # Start with equal weights
)
print("Optimized weights:",
      [f"{weight:.3f}" for weight in optimized_weights])
