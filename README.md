# value_model_tuning
nDCG and regret based tools to find value model weights

Value models (also known as multi-task fusion models) are a common component in modern recommender systems. They take estimates from the final ranking model for different tasks and combine them to create the final ranked list of recommendations.

For more background on value models in recommender systems, see:
- [Multi-Task Fusion via Reinforcement Learning for Long-Term User Satisfaction in Recommender Systems](https://arxiv.org/abs/2208.04560)
- [Ranking model calibration in recommender systems](https://recsysml.substack.com/p/ranking-model-calibration-in-recommender)

This repository provides tools to compute optimal weights for a linear value model based on intended importance of different tasks.

## Contents

- [sample_vm.json](./src/sample_vm.json): Example value model configuration
- [ndcg_targeting_v1.py](./src/ndcg_targeting_v1.py): NDCG-based weight optimization 
- [regret_targeting.py](./src/regret_targeting.py): Regret-based weight optimization

## NDCG Targeting

The `ndcg_targeting_v1.py` script optimizes value model weights to target specific NDCG (Normalized Discounted Cumulative Gain) values for each task. It uses gradient descent to find weights that produce NDCG scores close to the desired targets.

Key features:
- Supports multiple tasks with individual NDCG targets
- Configurable learning rate and number of iterations
- Outputs optimized weights for each task

### Supports both Black-box and Explicit Optimizations
[ndcg_targeting_v1.py](./src/ndcg_targeting_v1.py) uses a white-box optimization approach whereas [ndcg_targeting.py](./src/ndcg_targeting.py) uses SLSQP to optimize the weights. I believe the latter should be more accurate, but for whatever reason it's not working as well as I'd hoped. Currently, the SLSQP approach is terminating without converging.
```
Initial weights: ['0.200', '0.200', '0.200', '0.200', '0.200']

Final results:
  nDCG regrets for each task ignored: ['0.003', '0.002', '0.002', '0.002', '0.004']
  Current fractions: ['0.207', '0.183', '0.115', '0.180', '0.315']
  Desired fractions: ['0.300', '0.500', '0.070', '0.070', '0.060']
  Optimization success: True
  Optimization message: Optimization terminated successfully
Optimized weights: ['0.200', '0.200', '0.200', '0.200', '0.200']
```
whereas the white-box approach works as expected:
```
Initial weights: ['0.200', '0.200', '0.200', '0.200', '0.200']

Final results:
  nDCGs for each task ignored: ['0.994', '0.990', '0.999', '0.999', '0.999']
  Current fractions: ['0.296', '0.504', '0.069', '0.073', '0.059']
  Desired fractions: ['0.300', '0.500', '0.070', '0.070', '0.060']
Optimized weights: ['0.269', '0.354', '0.137', '0.143', '0.098']
```

## Regret Targeting 

The [regret_targeting.py](./src/regret_targeting.py) script takes a different approach. For each task, it looks at the ranking to optimize that task. Then it defines regret as the nDCG between the optimized weights ranking and the per-rask optimal ranking. The script has been provided a target regret ratio/fraction between tasks, and it finds the weights that produce this regret. This can be useful for balancing multiple competing objectives.

Key features:
- Minimizes regret across multiple tasks
- Configurable regularization to prevent overfitting
- Outputs optimized weights that balance task importance

## Usage

There are examples of usage in each script's docstrings and at the bottom of each script. Commands to run each script are provided in the docstrings.

## Contributing

Contributions to improve and extend these tools are welcome! Please submit issues and pull requests.
