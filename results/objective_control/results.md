# Objective control for nested inducing points

Data seeds [0, 1, 2, 3, 4], initialisation seeds [0, 1], 1000 steps, gamma 1.0, inducing-point jitter 0.05. Each reported value averages the initialisations inside a function draw, then reports the mean and standard deviation across draws.

The four joint losses use the same full-set ELBO term and the same uniform average over budgets 4, 8, and 16. Prefix losses read the leading block. Subset losses draw one fresh subset of that size. Independent NLPD+KL replaces the ELBO data term with the marginal predictive log likelihood and keeps the KL.

↓ lower is better. ↑ higher is better. → 0.95 means closer to nominal coverage. **Bold** is the best mean in that column.

## Does the prefix do more than the predictive loss?

Test NLPD ↓. Subset NLPD is the matched control for Prefix NLPD. Subset ELBO is the matched control for Prefix ELBO.

### k = 8

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | −0.012 ± 0.105 | −0.002 ± 0.216 | 0.067 ± 0.221 | 0.583 ± 0.166 |
| Subset NLPD, greedy | 0.018 ± 0.120 | 0.152 ± 0.169 | 0.132 ± 0.192 | 0.544 ± 0.116 |
| Subset NLPD, random mask | 0.318 ± 0.133 | 0.430 ± 0.113 | 0.382 ± 0.159 | 0.671 ± 0.100 |
| Prefix ELBO | 0.012 ± 0.139 | 0.061 ± 0.181 | 0.123 ± 0.182 | 0.575 ± 0.115 |
| Subset ELBO, greedy | 0.104 ± 0.046 | 0.109 ± 0.108 | 0.176 ± 0.141 | 0.491 ± 0.121 |
| Independent NLPD+KL | **−0.459 ± 0.078** | **−0.217 ± 0.103** | **−0.214 ± 0.085** | **0.351 ± 0.077** |
| Independent ELBO | −0.379 ± 0.088 | −0.092 ± 0.121 | −0.005 ± 0.165 | 0.458 ± 0.080 |

### k = 32

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | **−0.770 ± 0.073** | −0.648 ± 0.021 | −0.701 ± 0.084 | 0.129 ± 0.076 |
| Subset NLPD, greedy | −0.758 ± 0.057 | −0.649 ± 0.034 | −0.706 ± 0.080 | 0.099 ± 0.091 |
| Subset NLPD, random mask | −0.758 ± 0.057 | −0.649 ± 0.034 | −0.706 ± 0.080 | 0.099 ± 0.091 |
| Prefix ELBO | −0.303 ± 0.084 | −0.284 ± 0.063 | −0.266 ± 0.043 | 0.207 ± 0.049 |
| Subset ELBO, greedy | −0.066 ± 0.068 | −0.191 ± 0.047 | −0.143 ± 0.029 | 0.235 ± 0.064 |
| Independent NLPD+KL | −0.123 ± 0.235 | −0.485 ± 0.075 | −0.456 ± 0.066 | **0.017 ± 0.079** |
| Independent ELBO | −0.761 ± 0.057 | **−0.662 ± 0.016** | **−0.710 ± 0.078** | 0.148 ± 0.081 |

## Paired differences on the same function draws

Each difference is averaged over initialisations within a draw. The interval is a 95% Student-t interval across the 5 draws. For ↓ metrics, a negative difference favors the first method.

### NLPD ↓

| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|:---:|
| Prefix NLPD − subset NLPD greedy | 4 | 0.173 [−0.081, 0.427] | −0.100 [−0.386, 0.185] | 0.048 [−0.163, 0.258] | 0.006 [−0.129, 0.142] |
| Prefix NLPD − subset NLPD greedy | 8 | −0.030 [−0.175, 0.116] | −0.155 [−0.244, −0.066] | −0.065 [−0.198, 0.068] | 0.039 [−0.119, 0.197] |
| Prefix NLPD − subset NLPD greedy | 12 | −0.073 [−0.176, 0.030] | −0.084 [−0.207, 0.039] | 0.008 [−0.158, 0.174] | 0.060 [−0.121, 0.240] |
| Prefix NLPD − subset NLPD greedy | 16 | −0.051 [−0.152, 0.049] | −0.067 [−0.144, 0.010] | −0.065 [−0.125, −0.004] | 0.023 [−0.180, 0.226] |
| Prefix NLPD − subset NLPD greedy | 32 | −0.012 [−0.038, 0.015] | 0.002 [−0.019, 0.022] | 0.005 [−0.009, 0.019] | 0.030 [−0.001, 0.062] |
| Prefix NLPD − subset NLPD random mask | 4 | −0.183 [−0.536, 0.170] | −0.375 [−0.631, −0.119] | −0.216 [−0.443, 0.010] | −0.049 [−0.112, 0.013] |
| Prefix NLPD − subset NLPD random mask | 8 | −0.329 [−0.455, −0.203] | −0.433 [−0.620, −0.246] | −0.315 [−0.454, −0.175] | −0.088 [−0.186, 0.011] |
| Prefix NLPD − subset NLPD random mask | 12 | −0.284 [−0.388, −0.180] | −0.291 [−0.430, −0.153] | −0.168 [−0.296, −0.040] | −0.094 [−0.208, 0.019] |
| Prefix NLPD − subset NLPD random mask | 16 | −0.251 [−0.336, −0.165] | −0.258 [−0.356, −0.161] | −0.247 [−0.318, −0.176] | −0.125 [−0.230, −0.020] |
| Prefix NLPD − subset NLPD random mask | 32 | −0.012 [−0.038, 0.015] | 0.002 [−0.019, 0.022] | 0.005 [−0.009, 0.019] | 0.030 [−0.001, 0.062] |
| Prefix ELBO − subset ELBO greedy | 4 | 0.186 [−0.304, 0.677] | 0.002 [−0.176, 0.179] | −0.040 [−0.362, 0.281] | 0.064 [−0.042, 0.169] |
| Prefix ELBO − subset ELBO greedy | 8 | −0.093 [−0.246, 0.060] | −0.048 [−0.184, 0.089] | −0.053 [−0.176, 0.070] | 0.084 [−0.020, 0.188] |
| Prefix ELBO − subset ELBO greedy | 12 | −0.119 [−0.327, 0.088] | −0.004 [−0.129, 0.121] | 0.013 [−0.121, 0.147] | 0.130 [0.033, 0.226] |
| Prefix ELBO − subset ELBO greedy | 16 | −0.171 [−0.343, 0.002] | −0.050 [−0.116, 0.016] | 0.020 [−0.085, 0.125] | 0.108 [−0.041, 0.257] |
| Prefix ELBO − subset ELBO greedy | 32 | −0.237 [−0.390, −0.083] | −0.093 [−0.129, −0.058] | −0.123 [−0.153, −0.093] | −0.028 [−0.090, 0.035] |
| Prefix NLPD − prefix ELBO | 4 | 0.109 [−0.396, 0.614] | 0.083 [−0.088, 0.255] | 0.109 [−0.106, 0.324] | −0.060 [−0.164, 0.044] |
| Prefix NLPD − prefix ELBO | 8 | −0.023 [−0.268, 0.221] | −0.064 [−0.175, 0.047] | −0.056 [−0.163, 0.051] | 0.008 [−0.103, 0.120] |
| Prefix NLPD − prefix ELBO | 12 | −0.235 [−0.466, −0.004] | −0.180 [−0.260, −0.100] | −0.138 [−0.222, −0.053] | −0.063 [−0.195, 0.070] |
| Prefix NLPD − prefix ELBO | 16 | −0.380 [−0.570, −0.191] | −0.281 [−0.382, −0.179] | −0.354 [−0.454, −0.253] | −0.083 [−0.205, 0.039] |
| Prefix NLPD − prefix ELBO | 32 | −0.467 [−0.617, −0.318] | −0.364 [−0.424, −0.303] | −0.435 [−0.527, −0.342] | −0.078 [−0.115, −0.041] |
| Prefix NLPD − independent NLPD+KL | 4 | 0.631 [0.326, 0.936] | 0.287 [−0.077, 0.650] | 0.345 [0.078, 0.613] | 0.123 [0.019, 0.227] |
| Prefix NLPD − independent NLPD+KL | 8 | 0.448 [0.328, 0.568] | 0.215 [0.059, 0.371] | 0.282 [−0.008, 0.571] | 0.232 [0.038, 0.426] |
| Prefix NLPD − independent NLPD+KL | 12 | 0.232 [−0.006, 0.471] | 0.078 [−0.069, 0.225] | 0.231 [0.119, 0.343] | 0.279 [0.077, 0.480] |
| Prefix NLPD − independent NLPD+KL | 16 | −0.062 [−0.308, 0.185] | 0.003 [−0.082, 0.087] | −0.002 [−0.157, 0.153] | 0.307 [0.144, 0.469] |
| Prefix NLPD − independent NLPD+KL | 32 | −0.647 [−1.025, −0.270] | −0.163 [−0.256, −0.069] | −0.245 [−0.374, −0.115] | 0.113 [0.093, 0.132] |
| Prefix NLPD − independent ELBO | 4 | 0.431 [0.157, 0.705] | 0.168 [−0.142, 0.479] | 0.238 [0.080, 0.395] | 0.017 [−0.100, 0.134] |
| Prefix NLPD − independent ELBO | 8 | 0.368 [0.228, 0.507] | 0.090 [−0.073, 0.252] | 0.073 [−0.207, 0.352] | 0.125 [−0.039, 0.289] |
| Prefix NLPD − independent ELBO | 12 | 0.367 [0.177, 0.557] | 0.048 [−0.124, 0.219] | 0.196 [0.106, 0.286] | 0.076 [−0.152, 0.304] |
| Prefix NLPD − independent ELBO | 16 | 0.196 [0.117, 0.276] | 0.050 [−0.002, 0.103] | 0.103 [−0.033, 0.240] | 0.050 [−0.130, 0.230] |
| Prefix NLPD − independent ELBO | 32 | −0.009 [−0.029, 0.011] | 0.014 [−0.025, 0.053] | 0.008 [−0.005, 0.022] | −0.019 [−0.085, 0.047] |

### RMSE ↓

| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |
|---|:---:|:---:|:---:|:---:|:---:|
| Prefix NLPD − subset NLPD greedy | 4 | 0.161 [0.045, 0.277] | −0.019 [−0.169, 0.131] | 0.065 [−0.053, 0.184] | −0.005 [−0.050, 0.040] |
| Prefix NLPD − subset NLPD greedy | 8 | 0.044 [−0.059, 0.146] | −0.065 [−0.138, 0.008] | −0.017 [−0.148, 0.115] | −0.006 [−0.044, 0.033] |
| Prefix NLPD − subset NLPD greedy | 12 | 0.038 [−0.036, 0.111] | −0.025 [−0.069, 0.020] | 0.021 [−0.071, 0.112] | −0.011 [−0.062, 0.041] |
| Prefix NLPD − subset NLPD greedy | 16 | −0.004 [−0.017, 0.009] | −0.010 [−0.034, 0.014] | −0.034 [−0.050, −0.018] | −0.044 [−0.102, 0.014] |
| Prefix NLPD − subset NLPD greedy | 32 | −0.002 [−0.006, 0.001] | 0.002 [−0.002, 0.007] | 0.002 [−0.001, 0.005] | −0.002 [−0.010, 0.006] |
| Prefix NLPD − subset NLPD random mask | 4 | −0.117 [−0.345, 0.111] | −0.190 [−0.303, −0.078] | −0.134 [−0.307, 0.040] | −0.025 [−0.045, −0.004] |
| Prefix NLPD − subset NLPD random mask | 8 | −0.226 [−0.314, −0.139] | −0.211 [−0.287, −0.135] | −0.197 [−0.254, −0.139] | −0.055 [−0.101, −0.008] |
| Prefix NLPD − subset NLPD random mask | 12 | −0.138 [−0.220, −0.056] | −0.129 [−0.208, −0.051] | −0.066 [−0.149, 0.017] | −0.058 [−0.100, −0.017] |
| Prefix NLPD − subset NLPD random mask | 16 | −0.110 [−0.127, −0.093] | −0.093 [−0.143, −0.042] | −0.120 [−0.150, −0.091] | −0.073 [−0.132, −0.013] |
| Prefix NLPD − subset NLPD random mask | 32 | −0.002 [−0.006, 0.001] | 0.002 [−0.002, 0.007] | 0.002 [−0.001, 0.005] | −0.002 [−0.010, 0.006] |
| Prefix ELBO − subset ELBO greedy | 4 | 0.096 [−0.092, 0.283] | 0.017 [−0.070, 0.104] | −0.033 [−0.228, 0.162] | 0.019 [−0.014, 0.053] |
| Prefix ELBO − subset ELBO greedy | 8 | −0.005 [−0.050, 0.041] | −0.032 [−0.094, 0.031] | −0.052 [−0.114, 0.010] | 0.007 [−0.022, 0.036] |
| Prefix ELBO − subset ELBO greedy | 12 | −0.002 [−0.046, 0.042] | −0.012 [−0.073, 0.048] | −0.016 [−0.089, 0.056] | 0.006 [−0.020, 0.031] |
| Prefix ELBO − subset ELBO greedy | 16 | −0.029 [−0.059, 0.002] | −0.042 [−0.082, −0.002] | −0.013 [−0.040, 0.015] | −0.013 [−0.054, 0.028] |
| Prefix ELBO − subset ELBO greedy | 32 | −0.035 [−0.073, 0.003] | −0.033 [−0.050, −0.016] | −0.031 [−0.057, −0.005] | −0.005 [−0.014, 0.003] |
| Prefix NLPD − prefix ELBO | 4 | 0.154 [−0.099, 0.406] | 0.089 [−0.001, 0.179] | 0.147 [0.038, 0.257] | 0.002 [−0.012, 0.015] |
| Prefix NLPD − prefix ELBO | 8 | 0.048 [−0.026, 0.122] | 0.040 [0.000, 0.080] | 0.095 [0.014, 0.176] | 0.036 [0.006, 0.065] |
| Prefix NLPD − prefix ELBO | 12 | 0.021 [−0.064, 0.106] | 0.009 [−0.039, 0.057] | 0.082 [−0.021, 0.185] | 0.021 [−0.016, 0.058] |
| Prefix NLPD − prefix ELBO | 16 | −0.020 [−0.042, 0.002] | −0.011 [−0.066, 0.045] | −0.014 [−0.030, 0.003] | 0.016 [−0.010, 0.042] |
| Prefix NLPD − prefix ELBO | 32 | −0.026 [−0.034, −0.018] | −0.035 [−0.061, −0.010] | −0.042 [−0.057, −0.026] | −0.011 [−0.021, −0.001] |
| Prefix NLPD − independent NLPD+KL | 4 | 0.283 [0.110, 0.456] | 0.126 [−0.011, 0.263] | 0.188 [0.022, 0.355] | 0.007 [−0.015, 0.029] |
| Prefix NLPD − independent NLPD+KL | 8 | 0.107 [0.037, 0.178] | 0.071 [−0.003, 0.146] | 0.134 [0.004, 0.263] | −0.006 [−0.077, 0.066] |
| Prefix NLPD − independent NLPD+KL | 12 | 0.102 [0.037, 0.168] | 0.018 [−0.021, 0.057] | 0.131 [0.015, 0.247] | −0.030 [−0.104, 0.043] |
| Prefix NLPD − independent NLPD+KL | 16 | 0.007 [−0.045, 0.059] | 0.015 [−0.014, 0.043] | 0.027 [−0.014, 0.068] | −0.034 [−0.078, 0.010] |
| Prefix NLPD − independent NLPD+KL | 32 | −0.141 [−0.334, 0.051] | −0.019 [−0.051, 0.013] | −0.013 [−0.030, 0.003] | −0.053 [−0.075, −0.032] |
| Prefix NLPD − independent ELBO | 4 | 0.303 [0.129, 0.477] | 0.143 [0.029, 0.257] | 0.215 [0.091, 0.340] | 0.033 [−0.001, 0.067] |
| Prefix NLPD − independent ELBO | 8 | 0.124 [0.074, 0.174] | 0.063 [−0.004, 0.129] | 0.115 [−0.031, 0.260] | 0.085 [0.018, 0.152] |
| Prefix NLPD − independent ELBO | 12 | 0.111 [0.045, 0.178] | 0.049 [−0.045, 0.142] | 0.142 [0.024, 0.260] | 0.071 [−0.013, 0.154] |
| Prefix NLPD − independent ELBO | 16 | 0.029 [0.009, 0.050] | 0.031 [0.001, 0.060] | 0.041 [0.000, 0.081] | 0.052 [−0.020, 0.123] |
| Prefix NLPD − independent ELBO | 32 | −0.003 [−0.006, −0.000] | 0.008 [0.001, 0.016] | 0.002 [−0.003, 0.007] | 0.000 [−0.016, 0.016] |

## Latent-function uncertainty

Intervals below use the latent predictive variance, not the noisy-observation variance. Coverage → 0.95. Width ↓ is the mean 95% interval length.

### k = 2

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 0.873 ± 0.077 | 0.864 ± 0.052 | 0.895 ± 0.051 | **0.959 ± 0.040** |
| Subset NLPD, greedy | 0.996 ± 0.008 | 0.989 ± 0.008 | 0.996 ± 0.007 | 0.979 ± 0.019 |
| Independent NLPD+KL | **0.972 ± 0.035** | **0.974 ± 0.017** | **0.961 ± 0.026** | 0.993 ± 0.008 |
| Independent ELBO | 0.829 ± 0.184 | 0.830 ± 0.143 | 0.798 ± 0.258 | 0.829 ± 0.232 |

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 2.030 ± 0.369 | 1.975 ± 0.432 | 2.207 ± 0.582 | 2.067 ± 0.469 |
| Subset NLPD, greedy | 2.509 ± 0.518 | 2.589 ± 0.269 | 2.830 ± 0.606 | 2.201 ± 0.276 |
| Independent NLPD+KL | 1.996 ± 0.260 | 1.997 ± 0.383 | 2.227 ± 0.510 | 2.156 ± 0.106 |
| Independent ELBO | **1.552 ± 0.460** | **1.474 ± 0.583** | **1.575 ± 0.581** | **1.707 ± 0.576** |

### k = 8

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 0.998 ± 0.003 | **0.976 ± 0.015** | **0.975 ± 0.023** | **0.941 ± 0.021** |
| Subset NLPD, greedy | **0.997 ± 0.004** | 0.988 ± 0.011 | 0.988 ± 0.015 | 0.966 ± 0.019 |
| Independent NLPD+KL | 1.000 ± 0.000 | 0.991 ± 0.011 | 0.996 ± 0.010 | 0.972 ± 0.016 |
| Independent ELBO | 0.836 ± 0.038 | 0.785 ± 0.167 | 0.859 ± 0.092 | 0.832 ± 0.063 |

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 1.240 ± 0.197 | 1.127 ± 0.217 | 1.159 ± 0.211 | 1.714 ± 0.433 |
| Subset NLPD, greedy | 1.511 ± 0.262 | 1.482 ± 0.260 | 1.433 ± 0.351 | 1.749 ± 0.219 |
| Independent NLPD+KL | 0.709 ± 0.064 | 0.881 ± 0.109 | 0.928 ± 0.133 | 1.460 ± 0.217 |
| Independent ELBO | **0.378 ± 0.035** | **0.578 ± 0.275** | **0.685 ± 0.224** | **1.018 ± 0.252** |

### k = 32

| Method | EQ → 0.95 | Weak periodic → 0.95 | Matern → 0.95 | Sawtooth → 0.95 |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | 0.956 ± 0.032 | 0.911 ± 0.016 | **0.949 ± 0.022** | 0.885 ± 0.018 |
| Subset NLPD, greedy | **0.951 ± 0.026** | 0.939 ± 0.022 | 0.943 ± 0.022 | 0.902 ± 0.011 |
| Independent NLPD+KL | 0.994 ± 0.005 | 0.996 ± 0.005 | 0.996 ± 0.003 | **0.981 ± 0.011** |
| Independent ELBO | 0.952 ± 0.033 | **0.952 ± 0.017** | 0.943 ± 0.020 | 0.910 ± 0.028 |

| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |
|---|:---:|:---:|:---:|:---:|
| Prefix NLPD | **0.194 ± 0.015** | **0.257 ± 0.015** | 0.252 ± 0.026 | **0.694 ± 0.191** |
| Subset NLPD, greedy | 0.202 ± 0.015 | 0.276 ± 0.031 | 0.240 ± 0.024 | 0.764 ± 0.144 |
| Independent NLPD+KL | 1.021 ± 0.613 | 0.612 ± 0.158 | 0.634 ± 0.112 | 1.180 ± 0.209 |
| Independent ELBO | 0.203 ± 0.009 | 0.268 ± 0.017 | **0.234 ± 0.019** | 0.835 ± 0.227 |

## Independent EQ models by budget

Validation NLPD is the quantity used to pick the checkpoint. A smaller model with better validation NLPD than a larger one is an optimization or bias result, not a test-set accident.

| Objective | k | Test NLPD ↓ | Val NLPD ↓ | RMSE ↓ | Steps |
|---|:---:|:---:|:---:|:---:|:---:|
| ELBO | 2 | 0.738 ± 0.150 | 0.759 ± 0.234 | 0.509 ± 0.077 | 1000.000 ± 0.000 |
| ELBO | 4 | 0.206 ± 0.211 | 0.243 ± 0.239 | 0.266 ± 0.102 | 1000.000 ± 0.000 |
| ELBO | 6 | −0.179 ± 0.111 | −0.161 ± 0.133 | 0.135 ± 0.025 | 1000.000 ± 0.000 |
| ELBO | 8 | −0.379 ± 0.088 | −0.329 ± 0.130 | 0.118 ± 0.021 | 1000.000 ± 0.000 |
| ELBO | 12 | −0.729 ± 0.151 | −0.704 ± 0.109 | 0.055 ± 0.027 | 1000.000 ± 0.000 |
| ELBO | 16 | −0.815 ± 0.075 | −0.804 ± 0.058 | 0.038 ± 0.008 | 1000.000 ± 0.000 |
| ELBO | 24 | −0.748 ± 0.125 | −0.737 ± 0.098 | 0.046 ± 0.007 | 1000.000 ± 0.000 |
| ELBO | 32 | −0.761 ± 0.057 | −0.765 ± 0.042 | 0.052 ± 0.003 | 1000.000 ± 0.000 |
| NLPD+KL | 2 | 0.626 ± 0.117 | 0.613 ± 0.198 | 0.522 ± 0.088 | 1000.000 ± 0.000 |
| NLPD+KL | 4 | 0.006 ± 0.198 | 0.048 ± 0.219 | 0.286 ± 0.097 | 1000.000 ± 0.000 |
| NLPD+KL | 6 | −0.355 ± 0.111 | −0.354 ± 0.139 | 0.156 ± 0.064 | 1000.000 ± 0.000 |
| NLPD+KL | 8 | −0.459 ± 0.078 | −0.433 ± 0.151 | 0.135 ± 0.031 | 1000.000 ± 0.000 |
| NLPD+KL | 12 | −0.595 ± 0.203 | −0.663 ± 0.111 | 0.064 ± 0.029 | 1000.000 ± 0.000 |
| NLPD+KL | 16 | −0.556 ± 0.135 | −0.621 ± 0.184 | 0.060 ± 0.036 | 1000.000 ± 0.000 |
| NLPD+KL | 24 | −0.218 ± 0.187 | −0.333 ± 0.304 | 0.140 ± 0.156 | 1000.000 ± 0.000 |
| NLPD+KL | 32 | −0.123 ± 0.235 | −0.185 ± 0.313 | 0.190 ± 0.155 | 1000.000 ± 0.000 |

## Initialisation spread

Mean, across function draws, of the standard deviation across the two initialisations. This is the scale of initialisation noise, not a confidence interval.

| Method | k | EQ NLPD | EQ RMSE |
|---|:---:|:---:|:---:|
| Prefix NLPD | 8 | 0.111 | 0.054 |
| Prefix NLPD | 32 | 0.015 | 0.004 |
| Subset NLPD, greedy | 8 | 0.019 | 0.020 |
| Subset NLPD, greedy | 32 | 0.010 | 0.003 |
| Independent ELBO | 8 | 0.027 | 0.006 |
| Independent ELBO | 32 | 0.006 | 0.001 |
| Independent NLPD+KL | 8 | 0.043 | 0.021 |
| Independent NLPD+KL | 32 | 0.097 | 0.060 |

