# Comparator table for budgeted inducing points

Data seeds [0, 1, 2, 3, 4], initialisation seeds [0], 1000 steps. Values average initialisations within a function draw.

Nested GP trains leading blocks with marginal NLPD and the full set with the ELBO. Greedy variance orders a standard full sparse GP by pivoted Cholesky of its inducing kernel. Validation greedy builds one forward-selection order on that same full model. Random-subset training uses the same budgets, weights, and full ELBO term as the nested model, then one validation-greedy order. Inclusion-probability order trains the Uhrenholt Bernoulli point process with squared-count rate 5.0, then ranks points by inclusion probability and reuses the variational marginal. That ranking is an evaluation adaptation. The adaptive-size rows below use the paper's own set, not a forced budget.

↓ lower is better. ↑ higher is better. → 0.95 means closer to nominal coverage. **Bold** is the best mean in that column.

## Test NLPD ↓

### k = 8

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Nested GP | **0.002 ± 0.170** | **−0.011 ± 0.217** | **0.035 ± 0.225** | 0.576 ± 0.210 |
| Full GP + greedy variance | 0.433 ± 0.163 | 0.506 ± 0.075 | 0.518 ± 0.250 | 0.734 ± 0.205 |
| Full GP + validation greedy | 0.065 ± 0.111 | 0.286 ± 0.087 | 0.165 ± 0.215 | 0.622 ± 0.129 |
| Random-subset + greedy | 0.014 ± 0.126 | 0.127 ± 0.165 | 0.145 ± 0.193 | **0.549 ± 0.108** |
| Inclusion-probability order | 0.789 ± 0.322 | 0.714 ± 0.265 | 1.183 ± 0.259 | 0.868 ± 0.009 |

### k = 32

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Nested GP | **−0.773 ± 0.059** | −0.640 ± 0.029 | −0.700 ± 0.091 | 0.137 ± 0.078 |
| Full GP + greedy variance | −0.764 ± 0.060 | −0.652 ± 0.037 | **−0.714 ± 0.076** | 0.154 ± 0.074 |
| Full GP + validation greedy | −0.764 ± 0.060 | −0.652 ± 0.037 | **−0.714 ± 0.076** | 0.154 ± 0.074 |
| Random-subset + greedy | −0.758 ± 0.054 | **−0.656 ± 0.044** | −0.711 ± 0.082 | **0.093 ± 0.083** |
| Inclusion-probability order | 0.901 ± 0.285 | 1.673 ± 0.707 | 1.117 ± 0.187 | 1.019 ± 0.065 |

## Latent RMSE ↓

### k = 8

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Nested GP | 0.261 ± 0.071 | **0.242 ± 0.085** | **0.302 ± 0.110** | **0.451 ± 0.083** |
| Full GP + greedy variance | 0.455 ± 0.157 | 0.439 ± 0.043 | 0.476 ± 0.142 | 0.500 ± 0.053 |
| Full GP + validation greedy | 0.236 ± 0.068 | 0.388 ± 0.092 | 0.341 ± 0.141 | 0.472 ± 0.049 |
| Random-subset + greedy | **0.207 ± 0.067** | 0.313 ± 0.125 | 0.342 ± 0.135 | 0.458 ± 0.058 |
| Inclusion-probability order | 0.391 ± 0.126 | 0.431 ± 0.147 | 0.640 ± 0.183 | 0.558 ± 0.012 |

### k = 32

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Nested GP | **0.048 ± 0.003** | 0.078 ± 0.006 | 0.065 ± 0.009 | 0.268 ± 0.027 |
| Full GP + greedy variance | 0.051 ± 0.004 | **0.068 ± 0.008** | **0.061 ± 0.005** | **0.266 ± 0.023** |
| Full GP + validation greedy | 0.051 ± 0.004 | **0.068 ± 0.008** | **0.061 ± 0.005** | **0.266 ± 0.023** |
| Random-subset + greedy | 0.052 ± 0.006 | 0.074 ± 0.009 | 0.061 ± 0.006 | 0.269 ± 0.021 |
| Inclusion-probability order | 0.623 ± 0.341 | 0.965 ± 0.322 | 0.672 ± 0.131 | 0.636 ± 0.076 |

## Noisy-observation coverage → 0.95

### k = 8

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Nested GP | 0.991 ± 0.004 | 0.980 ± 0.023 | **0.977 ± 0.028** | **0.954 ± 0.014** |
| Full GP + greedy variance | **0.977 ± 0.034** | **0.950 ± 0.008** | 0.903 ± 0.062 | 0.975 ± 0.016 |
| Full GP + validation greedy | 0.994 ± 0.004 | 0.988 ± 0.014 | 0.981 ± 0.014 | 0.983 ± 0.009 |
| Random-subset + greedy | 0.995 ± 0.005 | 0.989 ± 0.011 | 0.987 ± 0.008 | 0.973 ± 0.011 |
| Inclusion-probability order | 1.000 ± 0.000 | 0.986 ± 0.012 | 0.996 ± 0.009 | 0.998 ± 0.003 |

### k = 32

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Nested GP | 0.964 ± 0.025 | **0.965 ± 0.011** | **0.961 ± 0.014** | 0.932 ± 0.009 |
| Full GP + greedy variance | 0.958 ± 0.013 | 0.978 ± 0.014 | 0.962 ± 0.013 | **0.943 ± 0.010** |
| Full GP + validation greedy | 0.958 ± 0.013 | 0.978 ± 0.014 | 0.962 ± 0.013 | **0.943 ± 0.010** |
| Random-subset + greedy | **0.954 ± 0.018** | 0.972 ± 0.021 | 0.962 ± 0.018 | 0.941 ± 0.008 |
| Inclusion-probability order | 0.966 ± 0.076 | 0.797 ± 0.146 | 0.994 ± 0.004 | 0.987 ± 0.013 |

## Latent-function coverage → 0.95

### k = 8

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Nested GP | 0.997 ± 0.007 | 0.980 ± 0.023 | **0.982 ± 0.025** | **0.939 ± 0.015** |
| Full GP + greedy variance | **0.979 ± 0.041** | **0.939 ± 0.020** | 0.895 ± 0.080 | 0.970 ± 0.017 |
| Full GP + validation greedy | 0.999 ± 0.002 | 0.986 ± 0.012 | 0.991 ± 0.012 | 0.977 ± 0.010 |
| Random-subset + greedy | 0.997 ± 0.007 | 0.991 ± 0.015 | 0.990 ± 0.010 | 0.964 ± 0.019 |
| Inclusion-probability order | 0.835 ± 0.147 | 0.834 ± 0.078 | 0.735 ± 0.240 | 0.263 ± 0.115 |

### k = 32

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Nested GP | 0.960 ± 0.033 | 0.915 ± 0.021 | 0.955 ± 0.014 | 0.882 ± 0.027 |
| Full GP + greedy variance | **0.948 ± 0.034** | 0.971 ± 0.022 | 0.945 ± 0.029 | **0.915 ± 0.031** |
| Full GP + validation greedy | **0.948 ± 0.034** | 0.971 ± 0.022 | 0.945 ± 0.029 | **0.915 ± 0.031** |
| Random-subset + greedy | 0.954 ± 0.028 | **0.937 ± 0.030** | **0.946 ± 0.020** | 0.901 ± 0.030 |
| Inclusion-probability order | 0.696 ± 0.152 | 0.579 ± 0.168 | 0.735 ± 0.191 | 0.639 ± 0.103 |

## Paired NLPD ↓ differences

Nested GP minus the comparator, paired by function draw. The bracket is a 95% Student-t interval across 5 draws. A negative value favors the nested GP.

| Nested GP minus | k | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|:---:|
| Full GP + greedy variance | 8 | −0.431 [−0.770, −0.092] | −0.517 [−0.799, −0.234] | −0.483 [−0.659, −0.306] | −0.158 [−0.322, 0.006] |
| Full GP + greedy variance | 16 | −0.180 [−0.249, −0.110] | −0.121 [−0.241, 0.000] | −0.034 [−0.155, 0.087] | −0.190 [−0.386, 0.006] |
| Full GP + greedy variance | 32 | −0.008 [−0.024, 0.007] | 0.012 [−0.067, 0.091] | 0.014 [−0.014, 0.042] | −0.017 [−0.110, 0.076] |
| Full GP + validation greedy | 8 | −0.064 [−0.199, 0.072] | −0.297 [−0.545, −0.049] | −0.130 [−0.229, −0.031] | −0.046 [−0.267, 0.176] |
| Full GP + validation greedy | 16 | −0.117 [−0.282, 0.047] | −0.094 [−0.176, −0.012] | −0.058 [−0.161, 0.046] | −0.103 [−0.306, 0.099] |
| Full GP + validation greedy | 32 | −0.008 [−0.024, 0.007] | 0.012 [−0.067, 0.091] | 0.014 [−0.014, 0.042] | −0.017 [−0.110, 0.076] |
| Random-subset + greedy | 8 | −0.013 [−0.161, 0.136] | −0.138 [−0.249, −0.026] | −0.110 [−0.225, 0.005] | 0.027 [−0.123, 0.177] |
| Random-subset + greedy | 16 | −0.050 [−0.141, 0.041] | −0.067 [−0.113, −0.020] | −0.095 [−0.183, −0.007] | −0.028 [−0.189, 0.133] |
| Random-subset + greedy | 32 | −0.015 [−0.039, 0.009] | 0.016 [−0.018, 0.050] | 0.011 [−0.007, 0.030] | 0.044 [0.004, 0.083] |
| Inclusion-probability order | 8 | −0.787 [−1.248, −0.327] | −0.724 [−1.066, −0.383] | −1.148 [−1.438, −0.858] | −0.291 [−0.544, −0.039] |
| Inclusion-probability order | 16 | −1.584 [−1.840, −1.329] | −1.646 [−2.608, −0.684] | −1.676 [−1.905, −1.446] | −0.578 [−0.808, −0.347] |
| Inclusion-probability order | 32 | −1.674 [−1.970, −1.378] | −2.313 [−3.176, −1.449] | −1.817 [−2.063, −1.571] | −0.882 [−1.024, −0.739] |

## Paired latent RMSE ↓ differences

| Nested GP minus | k | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|:---:|
| Full GP + greedy variance | 8 | −0.194 [−0.404, 0.016] | −0.197 [−0.313, −0.082] | −0.174 [−0.343, −0.005] | −0.049 [−0.092, −0.007] |
| Full GP + greedy variance | 16 | −0.034 [−0.066, −0.002] | −0.014 [−0.059, 0.031] | −0.017 [−0.055, 0.020] | −0.046 [−0.114, 0.023] |
| Full GP + greedy variance | 32 | −0.003 [−0.008, 0.003] | 0.010 [0.001, 0.019] | 0.004 [−0.004, 0.013] | 0.002 [−0.014, 0.018] |
| Full GP + validation greedy | 8 | 0.025 [−0.026, 0.076] | −0.146 [−0.305, 0.014] | −0.039 [−0.139, 0.060] | −0.021 [−0.087, 0.044] |
| Full GP + validation greedy | 16 | −0.036 [−0.099, 0.028] | −0.002 [−0.022, 0.018] | −0.038 [−0.096, 0.020] | −0.060 [−0.120, −0.000] |
| Full GP + validation greedy | 32 | −0.003 [−0.008, 0.003] | 0.010 [0.001, 0.019] | 0.004 [−0.004, 0.013] | 0.002 [−0.014, 0.018] |
| Random-subset + greedy | 8 | 0.054 [−0.035, 0.143] | −0.071 [−0.182, 0.041] | −0.040 [−0.147, 0.066] | −0.007 [−0.042, 0.028] |
| Random-subset + greedy | 16 | −0.010 [−0.026, 0.006] | −0.008 [−0.020, 0.004] | −0.059 [−0.086, −0.031] | −0.049 [−0.106, 0.008] |
| Random-subset + greedy | 32 | −0.004 [−0.011, 0.004] | 0.004 [−0.002, 0.010] | 0.004 [−0.002, 0.011] | −0.001 [−0.010, 0.007] |
| Inclusion-probability order | 8 | −0.130 [−0.296, 0.036] | −0.189 [−0.381, 0.004] | −0.338 [−0.650, −0.026] | −0.107 [−0.204, −0.011] |
| Inclusion-probability order | 16 | −0.450 [−0.699, −0.201] | −0.698 [−1.296, −0.100] | −0.517 [−0.773, −0.261] | −0.216 [−0.309, −0.123] |
| Inclusion-probability order | 32 | −0.575 [−0.999, −0.150] | −0.888 [−1.287, −0.488] | −0.606 [−0.774, −0.439] | −0.367 [−0.467, −0.268] |

## Ordering cost

Seconds to build one order reused at every budget. Prediction itself is not included. Inclusion ranking is an argsort of the learned probabilities.

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Full GP + greedy variance | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 |
| Full GP + validation greedy | 0.094 ± 0.004 | 0.137 ± 0.011 | 0.123 ± 0.007 | 0.110 ± 0.004 |
| Random-subset + greedy | 0.092 ± 0.005 | 0.133 ± 0.009 | 0.122 ± 0.005 | 0.110 ± 0.004 |
| Inclusion-probability order | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## Adaptive-size inclusion model

MAP keeps every point whose inclusion probability is at least 1/2. Sampled sets draw the Bernoulli process eight times. Neither result is placed on the fixed-budget curves.

| Task | MAP k | MAP NLPD ↓ | Sampled k | Sampled NLPD ↓ | Expected k |
|---|:---:|:---:|:---:|:---:|:---:|
| EQ | 2.200 ± 0.837 | 0.626 ± 0.332 | 2.225 ± 0.797 | 0.631 ± 0.335 | 2.199 ± 0.773 |
| Weak periodic | 2.600 ± 0.548 | 0.526 ± 0.179 | 2.575 ± 0.542 | 0.532 ± 0.186 | 2.580 ± 0.517 |
| Matern | 1.200 ± 0.447 | 0.974 ± 0.164 | 1.225 ± 0.379 | 0.987 ± 0.159 | 0.907 ± 0.769 |
| Sawtooth | 1.000 ± 0.000 | 0.892 ± 0.019 | 1.000 ± 0.000 | 0.893 ± 0.019 | 0.239 ± 0.036 |

