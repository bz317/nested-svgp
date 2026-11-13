# ELBO rerun with the noise parameter protected

On a single EQ pilot, equal-weight prefix ELBO training raised the shared noise variance from 0.0095 to 0.059 and moved full-budget test NLPD from −0.86 to −0.29. The prefix term explains a truncated residual by inflating the one noise parameter used at every budget.

This rerun uses full-ELBO weight 10. Prefix and subset ELBO gradients do not update the noise parameter. Lengthscale and output scale remain shared. Data seeds [0, 1, 2, 3, 4], initialisation seeds [0, 1], 1000 steps. Prefix NLPD, the old shared-noise prefix ELBO, and the independent ELBO are the earlier objective-control runs on the same draws.

↓ lower is better. **Bold** is the best mean in that column.

## Test NLPD ↓

### k = 8

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix ELBO, noise protected | −0.194 ± 0.173 | −0.026 ± 0.191 | −0.038 ± 0.176 | 0.656 ± 0.136 |
| Subset ELBO, noise protected | −0.300 ± 0.079 | −0.007 ± 0.174 | **−0.096 ± 0.160** | 0.529 ± 0.157 |
| Prefix NLPD | −0.012 ± 0.105 | −0.002 ± 0.216 | 0.067 ± 0.221 | 0.583 ± 0.166 |
| Independent ELBO | **−0.379 ± 0.088** | **−0.092 ± 0.121** | −0.005 ± 0.165 | **0.458 ± 0.080** |
| Prefix ELBO, shared noise | 0.012 ± 0.139 | 0.061 ± 0.181 | 0.123 ± 0.182 | 0.575 ± 0.115 |

### k = 32

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix ELBO, noise protected | −0.754 ± 0.066 | −0.648 ± 0.018 | −0.672 ± 0.082 | 0.153 ± 0.073 |
| Subset ELBO, noise protected | −0.762 ± 0.047 | −0.604 ± 0.038 | −0.657 ± 0.082 | 0.153 ± 0.077 |
| Prefix NLPD | **−0.770 ± 0.073** | −0.648 ± 0.021 | −0.701 ± 0.084 | **0.129 ± 0.076** |
| Independent ELBO | −0.761 ± 0.057 | **−0.662 ± 0.016** | **−0.710 ± 0.078** | 0.148 ± 0.081 |
| Prefix ELBO, shared noise | −0.303 ± 0.084 | −0.284 ± 0.063 | −0.266 ± 0.043 | 0.207 ± 0.049 |

## Latent RMSE ↓

### k = 8

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix ELBO, noise protected | 0.183 ± 0.087 | 0.229 ± 0.073 | 0.225 ± 0.048 | 0.457 ± 0.057 |
| Subset ELBO, noise protected | 0.162 ± 0.035 | 0.251 ± 0.073 | 0.211 ± 0.071 | 0.427 ± 0.070 |
| Prefix NLPD | 0.242 ± 0.053 | 0.252 ± 0.084 | 0.314 ± 0.103 | 0.446 ± 0.063 |
| Independent ELBO | **0.118 ± 0.021** | **0.190 ± 0.034** | **0.200 ± 0.052** | **0.361 ± 0.048** |
| Prefix ELBO, shared noise | 0.194 ± 0.052 | 0.212 ± 0.082 | 0.220 ± 0.073 | 0.411 ± 0.052 |

### k = 32

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix ELBO, noise protected | 0.048 ± 0.011 | 0.073 ± 0.005 | 0.068 ± 0.006 | **0.265 ± 0.025** |
| Subset ELBO, noise protected | **0.048 ± 0.009** | 0.081 ± 0.005 | 0.072 ± 0.008 | 0.269 ± 0.022 |
| Prefix NLPD | 0.049 ± 0.003 | 0.076 ± 0.005 | 0.064 ± 0.007 | 0.269 ± 0.028 |
| Independent ELBO | 0.052 ± 0.003 | **0.068 ± 0.006** | **0.062 ± 0.006** | 0.269 ± 0.023 |
| Prefix ELBO, shared noise | 0.075 ± 0.004 | 0.112 ± 0.018 | 0.106 ± 0.015 | 0.280 ± 0.022 |

## Learned noise variance ↓

The data-generating noise variance is 0.01. Lower is not always better; the relevant comparison is whether the fit stays near that value.

| Method | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|
| Prefix ELBO, noise protected | 0.013 ± 0.002 | 0.016 ± 0.002 | 0.014 ± 0.002 | 0.054 ± 0.018 |
| Subset ELBO, noise protected | 0.013 ± 0.001 | 0.017 ± 0.003 | 0.014 ± 0.002 | 0.051 ± 0.021 |
| Prefix NLPD | — | — | — | — |
| Independent ELBO | — | — | — | — |
| Prefix ELBO, shared noise | — | — | — | — |

## Paired NLPD ↓ against the independent ELBO

A negative difference favors the first method. Brackets are 95% intervals across draws.

| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|:---:|
| Noise-protected prefix ELBO − independent ELBO | 8 | 0.186 [−0.017, 0.388] | 0.067 [−0.050, 0.183] | −0.033 [−0.273, 0.207] | 0.198 [0.081, 0.315] |
| Noise-protected prefix ELBO − independent ELBO | 32 | 0.007 [−0.032, 0.046] | 0.014 [−0.019, 0.046] | 0.038 [0.016, 0.059] | 0.004 [−0.044, 0.053] |
| Shared-noise prefix ELBO − independent ELBO | 8 | 0.391 [0.276, 0.506] | 0.153 [0.054, 0.252] | 0.128 [−0.102, 0.359] | 0.117 [−0.034, 0.268] |
| Shared-noise prefix ELBO − independent ELBO | 32 | 0.458 [0.324, 0.593] | 0.378 [0.292, 0.463] | 0.443 [0.362, 0.524] | 0.059 [0.001, 0.117] |
| Prefix NLPD − independent ELBO | 8 | 0.368 [0.228, 0.507] | 0.090 [−0.073, 0.252] | 0.073 [−0.207, 0.352] | 0.125 [−0.039, 0.289] |
| Prefix NLPD − independent ELBO | 32 | −0.009 [−0.029, 0.011] | 0.014 [−0.025, 0.053] | 0.008 [−0.005, 0.022] | −0.019 [−0.085, 0.047] |

