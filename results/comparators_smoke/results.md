# Comparator table for budgeted inducing points

Data seeds [0], initialisation seeds [0], 80 steps. Values average initialisations within a function draw.

Nested GP trains leading blocks with marginal NLPD and the full set with the ELBO. Greedy variance orders a standard full sparse GP by pivoted Cholesky of its inducing kernel. Validation greedy builds one forward-selection order on that same full model. Random-subset training uses the same budgets, weights, and full ELBO term as the nested model, then one validation-greedy order. Inclusion-probability order trains the Uhrenholt Bernoulli point process with squared-count rate 5.0, then ranks points by inclusion probability and reuses the variational marginal. That ranking is an evaluation adaptation. The adaptive-size rows below use the paper's own set, not a forced budget.

↓ lower is better. ↑ higher is better. → 0.95 means closer to nominal coverage. **Bold** is the best mean in that column.

## Test NLPD ↓

### k = 8

| Method | EQ ↓ |
|---|:---:|
| Nested GP | 1.182 ± 0.000 |
| Full GP + greedy variance | 1.187 ± 0.000 |
| Full GP + validation greedy | 1.099 ± 0.000 |
| Random-subset + greedy | **1.058 ± 0.000** |
| Inclusion-probability order | 1.755 ± 0.000 |

### k = 32

| Method | EQ ↓ |
|---|:---:|
| Nested GP | **0.388 ± 0.000** |
| Full GP + greedy variance | 0.432 ± 0.000 |
| Full GP + validation greedy | 0.432 ± 0.000 |
| Random-subset + greedy | 0.429 ± 0.000 |
| Inclusion-probability order | 0.720 ± 0.000 |

## Latent RMSE ↓

### k = 8

| Method | EQ ↓ |
|---|:---:|
| Nested GP | 0.791 ± 0.000 |
| Full GP + greedy variance | 0.755 ± 0.000 |
| Full GP + validation greedy | **0.718 ± 0.000** |
| Random-subset + greedy | 0.727 ± 0.000 |
| Inclusion-probability order | 0.991 ± 0.000 |

### k = 32

| Method | EQ ↓ |
|---|:---:|
| Nested GP | **0.411 ± 0.000** |
| Full GP + greedy variance | 0.431 ± 0.000 |
| Full GP + validation greedy | 0.431 ± 0.000 |
| Random-subset + greedy | 0.429 ± 0.000 |
| Inclusion-probability order | 0.537 ± 0.000 |

## Noisy-observation coverage → 0.95

### k = 8

| Method | EQ → 0.95 |
|---|:---:|
| Nested GP | **1.000 ± 0.000** |
| Full GP + greedy variance | **1.000 ± 0.000** |
| Full GP + validation greedy | **1.000 ± 0.000** |
| Random-subset + greedy | **1.000 ± 0.000** |
| Inclusion-probability order | 0.730 ± 0.000 |

### k = 32

| Method | EQ → 0.95 |
|---|:---:|
| Nested GP | 0.995 ± 0.000 |
| Full GP + greedy variance | 1.000 ± 0.000 |
| Full GP + validation greedy | 1.000 ± 0.000 |
| Random-subset + greedy | 0.990 ± 0.000 |
| Inclusion-probability order | **0.935 ± 0.000** |

## Latent-function coverage → 0.95

### k = 8

| Method | EQ → 0.95 |
|---|:---:|
| Nested GP | **1.000 ± 0.000** |
| Full GP + greedy variance | **1.000 ± 0.000** |
| Full GP + validation greedy | **1.000 ± 0.000** |
| Random-subset + greedy | **1.000 ± 0.000** |
| Inclusion-probability order | 0.710 ± 0.000 |

### k = 32

| Method | EQ → 0.95 |
|---|:---:|
| Nested GP | 1.000 ± 0.000 |
| Full GP + greedy variance | 1.000 ± 0.000 |
| Full GP + validation greedy | 1.000 ± 0.000 |
| Random-subset + greedy | **0.970 ± 0.000** |
| Inclusion-probability order | 0.925 ± 0.000 |

## Paired NLPD ↓ differences

Nested GP minus the comparator, paired by function draw. The bracket is a 95% Student-t interval across 1 draws. A negative value favors the nested GP.

| Nested GP minus | k | EQ |
|---|:---:|:---:|
| Full GP + greedy variance | 8 | −0.005 [−0.005, −0.005] |
| Full GP + greedy variance | 16 | 0.060 [0.060, 0.060] |
| Full GP + greedy variance | 32 | −0.044 [−0.044, −0.044] |
| Full GP + validation greedy | 8 | 0.083 [0.083, 0.083] |
| Full GP + validation greedy | 16 | 0.172 [0.172, 0.172] |
| Full GP + validation greedy | 32 | −0.044 [−0.044, −0.044] |
| Random-subset + greedy | 8 | 0.124 [0.124, 0.124] |
| Random-subset + greedy | 16 | 0.242 [0.242, 0.242] |
| Random-subset + greedy | 32 | −0.040 [−0.040, −0.040] |
| Inclusion-probability order | 8 | −0.573 [−0.573, −0.573] |
| Inclusion-probability order | 16 | 0.373 [0.373, 0.373] |
| Inclusion-probability order | 32 | −0.332 [−0.332, −0.332] |

## Paired latent RMSE ↓ differences

| Nested GP minus | k | EQ |
|---|:---:|:---:|
| Full GP + greedy variance | 8 | 0.037 [0.037, 0.037] |
| Full GP + greedy variance | 16 | 0.122 [0.122, 0.122] |
| Full GP + greedy variance | 32 | −0.020 [−0.020, −0.020] |
| Full GP + validation greedy | 8 | 0.074 [0.074, 0.074] |
| Full GP + validation greedy | 16 | 0.075 [0.075, 0.075] |
| Full GP + validation greedy | 32 | −0.020 [−0.020, −0.020] |
| Random-subset + greedy | 8 | 0.064 [0.064, 0.064] |
| Random-subset + greedy | 16 | 0.180 [0.180, 0.180] |
| Random-subset + greedy | 32 | −0.018 [−0.018, −0.018] |
| Inclusion-probability order | 8 | −0.200 [−0.200, −0.200] |
| Inclusion-probability order | 16 | 0.246 [0.246, 0.246] |
| Inclusion-probability order | 32 | −0.126 [−0.126, −0.126] |

## Ordering cost

Seconds to build one order reused at every budget. Prediction itself is not included. Inclusion ranking is an argsort of the learned probabilities.

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Full GP + greedy variance | 0.001 ± 0.000 |
| Full GP + validation greedy | 0.115 ± 0.000 |
| Random-subset + greedy | 0.113 ± 0.000 |
| Inclusion-probability order | 0.000 ± 0.000 |

## Adaptive-size inclusion model

MAP keeps every point whose inclusion probability is at least 1/2. Sampled sets draw the Bernoulli process eight times. Neither result is placed on the fixed-budget curves.

| Task | MAP k | MAP NLPD ↓ | Sampled k | Sampled NLPD ↓ | Expected k |
|---|:---:|:---:|:---:|:---:|:---:|
| EQ | 6.000 ± 0.000 | 1.951 ± 0.000 | 10.250 ± 0.000 | 1.098 ± 0.000 | 12.223 ± 0.000 |

