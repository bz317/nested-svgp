# Objective control for nested inducing points

Data seeds [0], initialisation seeds [0], 30 steps, gamma 1.0, inducing-point jitter 0.05. Each reported value averages the initialisations inside a function draw, then reports the mean and standard deviation across draws.

The four joint losses use the same full-set ELBO term and the same uniform average over budgets 4, 8, and 16. Prefix losses read the leading block. Subset losses draw one fresh subset of that size. Independent NLPD+KL replaces the ELBO data term with the marginal predictive log likelihood and keeps the KL.

↓ lower is better. ↑ higher is better. → 0.95 means closer to nominal coverage. **Bold** is the best mean in that column.

## Does the prefix do more than the predictive loss?

Test NLPD ↓. Subset NLPD is the matched control for Prefix NLPD. Subset ELBO is the matched control for Prefix ELBO.

### k = 8

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 1.312 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, greedy | 1.229 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, random mask | 1.401 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Prefix ELBO | 1.142 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset ELBO, greedy | 1.338 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent NLPD+KL | 0.891 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent ELBO | **0.695 ± 0.000** | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |

### k = 32

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 0.987 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, greedy | 1.011 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, random mask | 1.011 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Prefix ELBO | **0.842 ± 0.000** | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset ELBO, greedy | 1.289 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent NLPD+KL | 1.493 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent ELBO | 1.008 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |

## Paired differences on the same function draws

Each difference is averaged over initialisations within a draw. The interval is a 95% Student-t interval across the five draws. For ↓ metrics, a negative difference favors the first method.

### NLPD ↓

| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|:---:|
| Prefix NLPD − subset NLPD greedy | 4 | 0.081 [0.081, 0.081] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD greedy | 8 | 0.083 [0.083, 0.083] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD greedy | 12 | 0.193 [0.193, 0.193] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD greedy | 16 | 0.206 [0.206, 0.206] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD greedy | 32 | −0.025 [−0.025, −0.025] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 4 | −0.032 [−0.032, −0.032] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 8 | −0.089 [−0.089, −0.089] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 12 | 0.003 [0.003, 0.003] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 16 | 0.044 [0.044, 0.044] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 32 | −0.025 [−0.025, −0.025] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 4 | −0.165 [−0.165, −0.165] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 8 | −0.196 [−0.196, −0.196] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 12 | −0.233 [−0.233, −0.233] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 16 | −0.212 [−0.212, −0.212] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 32 | −0.447 [−0.447, −0.447] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 4 | 0.160 [0.160, 0.160] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 8 | 0.170 [0.170, 0.170] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 12 | 0.225 [0.225, 0.225] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 16 | 0.211 [0.211, 0.211] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 32 | 0.145 [0.145, 0.145] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 4 | 0.303 [0.303, 0.303] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 8 | 0.421 [0.421, 0.421] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 12 | 0.456 [0.456, 0.456] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 16 | 0.366 [0.366, 0.366] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 32 | −0.506 [−0.506, −0.506] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 4 | 0.366 [0.366, 0.366] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 8 | 0.616 [0.616, 0.616] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 12 | 0.415 [0.415, 0.415] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 16 | 0.581 [0.581, 0.581] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 32 | −0.021 [−0.021, −0.021] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |

### RMSE ↓

| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|:---:|
| Prefix NLPD − subset NLPD greedy | 4 | 0.118 [0.118, 0.118] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD greedy | 8 | −0.017 [−0.017, −0.017] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD greedy | 12 | 0.060 [0.060, 0.060] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD greedy | 16 | 0.065 [0.065, 0.065] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD greedy | 32 | 0.008 [0.008, 0.008] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 4 | 0.013 [0.013, 0.013] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 8 | −0.122 [−0.122, −0.122] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 12 | −0.053 [−0.053, −0.053] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 16 | −0.006 [−0.006, −0.006] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − subset NLPD random mask | 32 | 0.008 [0.008, 0.008] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 4 | −0.149 [−0.149, −0.149] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 8 | −0.209 [−0.209, −0.209] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 12 | −0.238 [−0.238, −0.238] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 16 | −0.216 [−0.216, −0.216] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix ELBO − subset ELBO greedy | 32 | −0.292 [−0.292, −0.292] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 4 | 0.212 [0.212, 0.212] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 8 | 0.150 [0.150, 0.150] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 12 | 0.143 [0.143, 0.143] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 16 | 0.125 [0.125, 0.125] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − prefix ELBO | 32 | 0.137 [0.137, 0.137] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 4 | 0.269 [0.269, 0.269] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 8 | 0.394 [0.394, 0.394] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 12 | 0.188 [0.188, 0.188] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 16 | 0.198 [0.198, 0.198] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent NLPD+KL | 32 | −0.367 [−0.367, −0.367] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 4 | 0.385 [0.385, 0.385] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 8 | 0.523 [0.523, 0.523] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 12 | 0.280 [0.280, 0.280] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 16 | 0.345 [0.345, 0.345] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |
| Prefix NLPD − independent ELBO | 32 | −0.000 [−0.000, −0.000] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |

## Latent-function uncertainty

Intervals below use the latent predictive variance, not the noisy-observation variance. Coverage → 0.95. Width ↓ is the mean 95% interval length.

### k = 2

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 1.000 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, greedy | 0.985 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent NLPD+KL | 1.000 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent ELBO | **0.975 ± 0.000** | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 4.373 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, greedy | 4.376 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent NLPD+KL | 3.916 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent ELBO | **2.994 ± 0.000** | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |

### k = 8

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 1.000 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, greedy | **0.995 ± 0.000** | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent NLPD+KL | 1.000 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent ELBO | 1.000 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 3.656 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, greedy | 3.538 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent NLPD+KL | 3.467 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent ELBO | **2.687 ± 0.000** | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |

### k = 32

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | **0.950 ± 0.000** | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, greedy | 0.935 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent NLPD+KL | 1.000 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent ELBO | 0.925 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 2.655 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Subset NLPD, greedy | 2.682 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent NLPD+KL | 3.942 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| Independent ELBO | **2.651 ± 0.000** | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |

## Independent EQ models by budget

Validation NLPD is the quantity used to pick the checkpoint. A smaller model with better validation NLPD than a larger one is an optimization or bias result, not a test-set accident.

| Objective | k | Test NLPD ↓ | Val NLPD ↓ | RMSE ↓ | Steps |
|---|:---:|:---:|:---:|:---:|:---:|
| ELBO | 2 | 1.187 ± 0.000 | 1.205 ± 0.000 | 0.772 ± 0.000 | 30.000 ± 0.000 |
| ELBO | 4 | 1.115 ± 0.000 | 1.216 ± 0.000 | 0.707 ± 0.000 | 30.000 ± 0.000 |
| ELBO | 6 | 0.814 ± 0.000 | 0.855 ± 0.000 | 0.477 ± 0.000 | 30.000 ± 0.000 |
| ELBO | 8 | 0.695 ± 0.000 | 0.694 ± 0.000 | 0.367 ± 0.000 | 30.000 ± 0.000 |
| ELBO | 12 | 0.888 ± 0.000 | 0.941 ± 0.000 | 0.583 ± 0.000 | 30.000 ± 0.000 |
| ELBO | 16 | 0.663 ± 0.000 | 0.685 ± 0.000 | 0.484 ± 0.000 | 30.000 ± 0.000 |
| ELBO | 24 | 0.783 ± 0.000 | 0.841 ± 0.000 | 0.625 ± 0.000 | 30.000 ± 0.000 |
| ELBO | 32 | 1.008 ± 0.000 | 1.081 ± 0.000 | 0.781 ± 0.000 | 30.000 ± 0.000 |
| NLPD+KL | 2 | 1.272 ± 0.000 | 1.279 ± 0.000 | 0.823 ± 0.000 | 30.000 ± 0.000 |
| NLPD+KL | 4 | 1.177 ± 0.000 | 1.237 ± 0.000 | 0.823 ± 0.000 | 30.000 ± 0.000 |
| NLPD+KL | 6 | 0.970 ± 0.000 | 0.988 ± 0.000 | 0.596 ± 0.000 | 30.000 ± 0.000 |
| NLPD+KL | 8 | 0.891 ± 0.000 | 0.895 ± 0.000 | 0.496 ± 0.000 | 30.000 ± 0.000 |
| NLPD+KL | 12 | 0.846 ± 0.000 | 0.860 ± 0.000 | 0.676 ± 0.000 | 30.000 ± 0.000 |
| NLPD+KL | 16 | 0.878 ± 0.000 | 0.841 ± 0.000 | 0.632 ± 0.000 | 30.000 ± 0.000 |
| NLPD+KL | 24 | 1.377 ± 0.000 | 1.385 ± 0.000 | 1.080 ± 0.000 | 30.000 ± 0.000 |
| NLPD+KL | 32 | 1.493 ± 0.000 | 1.482 ± 0.000 | 1.148 ± 0.000 | 30.000 ± 0.000 |

## Initialisation spread

Mean, across function draws, of the standard deviation across the two initialisations. This is the scale of initialisation noise, not a confidence interval.

| Method | k | EQ NLPD | EQ RMSE |
|---|:---:|:---:|:---:|
| Prefix NLPD | 8 | 0.000 | 0.000 |
| Prefix NLPD | 32 | 0.000 | 0.000 |
| Subset NLPD, greedy | 8 | 0.000 | 0.000 |
| Subset NLPD, greedy | 32 | 0.000 | 0.000 |
| Independent ELBO | 8 | 0.000 | 0.000 |
| Independent ELBO | 32 | 0.000 | 0.000 |
| Independent NLPD+KL | 8 | 0.000 | 0.000 |
| Independent NLPD+KL | 32 | 0.000 | 0.000 |

