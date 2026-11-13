# Greedy initialization plus prefix training

One pretrained full GP per draw, then 40 prefix steps. Data seeds [0]. Ordering and acceptance use the selection split. The test split is reported only here.

A fixes a random permutation. B stops after the greedy order. C trains prefixes on that fixed greedy order. D replaces the order every 200 steps only when the average selection NLPD at budgets 4, 8, and 16 improves. Random-subset training starts from the same checkpoint and receives one greedy order at the end.

↓ lower is better. **Bold** is the best mean in that column. For a difference, negative favors the first method.

## Test NLPD ↓

### k = 8

| Method | EQ ↓ |
|---|:---:|
| B greedy, no training | 1.131 ± 0.000 |
| A random, then prefix | 1.272 ± 0.000 |
| C greedy, then prefix | 1.030 ± 0.000 |
| D greedy, updated | 1.030 ± 0.000 |
| Random-subset + greedy | **0.978 ± 0.000** |

### k = 32

| Method | EQ ↓ |
|---|:---:|
| B greedy, no training | 0.826 ± 0.000 |
| A random, then prefix | 0.307 ± 0.000 |
| C greedy, then prefix | **0.229 ± 0.000** |
| D greedy, updated | **0.229 ± 0.000** |
| Random-subset + greedy | 0.401 ± 0.000 |

## Latent RMSE ↓

### k = 8

| Method | EQ ↓ |
|---|:---:|
| B greedy, no training | 0.842 ± 0.000 |
| A random, then prefix | 0.968 ± 0.000 |
| C greedy, then prefix | 0.859 ± 0.000 |
| D greedy, updated | 0.859 ± 0.000 |
| Random-subset + greedy | **0.682 ± 0.000** |

### k = 32

| Method | EQ ↓ |
|---|:---:|
| B greedy, no training | 0.653 ± 0.000 |
| A random, then prefix | 0.335 ± 0.000 |
| C greedy, then prefix | **0.309 ± 0.000** |
| D greedy, updated | **0.309 ± 0.000** |
| Random-subset + greedy | 0.381 ± 0.000 |

## Paired differences

| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|:---:|
| NLPD ↓ | | | | | |
| C − A  (initial order) | 8 | −0.242 [−0.242, −0.242] |
| C − A  (initial order) | 32 | −0.078 [−0.078, −0.078] |
| C − B  (prefix training) | 8 | −0.101 [−0.101, −0.101] |
| C − B  (prefix training) | 32 | −0.597 [−0.597, −0.597] |
| D − C  (updating the order) | 8 | 0.000 [0.000, 0.000] |
| D − C  (updating the order) | 32 | 0.000 [0.000, 0.000] |
| C − subset | 8 | 0.052 [0.052, 0.052] |
| C − subset | 32 | −0.173 [−0.173, −0.173] |
| RMSE ↓ | | | | | |
| C − A  (initial order) | 8 | −0.108 [−0.108, −0.108] |
| C − A  (initial order) | 32 | −0.026 [−0.026, −0.026] |
| C − B  (prefix training) | 8 | 0.017 [0.017, 0.017] |
| C − B  (prefix training) | 32 | −0.344 [−0.344, −0.344] |
| D − C  (updating the order) | 8 | 0.000 [0.000, 0.000] |
| D − C  (updating the order) | 32 | 0.000 [0.000, 0.000] |
| C − subset | 8 | 0.178 [0.178, 0.178] |
| C − subset | 32 | −0.072 [−0.072, −0.072] |

