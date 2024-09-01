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

## Regret Targeting 

The `regret_targeting.py` script takes a different approach, optimizing weights to minimize regret - the difference between the optimal and actual selections. This can be useful for balancing multiple competing objectives.

Key features:
- Minimizes regret across multiple tasks
- Configurable regularization to prevent overfitting
- Outputs optimized weights that balance task importance

## Usage

[Add specific usage instructions for running the scripts]

## Contributing

Contributions to improve and extend these tools are welcome! Please submit issues and pull requests.
