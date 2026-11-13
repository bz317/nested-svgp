# Nested sparse Gaussian processes

One unwhitened sparse GP is trained for every budget at once. The variational family is $`q(u)=N(m,S)`$ with $`S=LL^\top`$. A budget $`k`$ is a prefix of the inducing set: the first $`k`$ locations, the corresponding mean, and the matching principal submatrix of $`S`$. The full set is unchanged by a reordering of those points.

The primary order is conditional KL. Start from the empty prefix and add the inducing point that most increases $`\mathrm{KL}(q(u_A)\,\|\,p(u_A))`$. That order is recomputed after every training step. The training loss is the mean predictive NLPD of the prefixes at $`k\in\{4,8,16\}`$, plus the full-set ELBO.

Installation is in [INSTALL.md](INSTALL.md). The 1D comparison is below. A short account of the methods is in [summary_so_far.txt](summary_so_far.txt).

## Synthetic results

Four tasks on $`[-2, 2]`$, three data seeds, 32 inducing points, budgets $`k=1,\ldots,32`$. Each entry is the mean across seeds, with the seed standard deviation in parentheses.

**Full** is the raw score at $`k=32`$. Lower is better.

**BP-AUC** is the budget-performance AUC. For one task and one metric, take each seed’s curve $`y_s(k)`$. Let $`L`$ and $`H`$ be the minimum and maximum, over methods and budgets, of the seed-averaged curves. The same $`L`$ and $`H`$ are used for every method. The scaled curve is $`(H-y_s(k))/(H-L)`$, which is higher when the raw score is better. BP-AUC is the trapezoidal average of that curve from $`k=1`$ to $`k=32`$, divided by $`31`$. The number shown is the mean of the three seed AUCs. Higher is better. Bold marks the best mean in that column. A tie at the displayed precision is bolded on both.

GVS reads prefixes of the condKL posterior, so its full-budget score matches condKL. The naive sparse GP is one full ELBO, sorted once by conditional KL.

### Test NLPD

#### EQ

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | **−0.792 (0.089)** | **0.880 (0.012)** |
| Naive sparse GP | −0.778 (0.066) | 0.840 (0.009) |
| RST–Greedy | **−0.792 (0.087)** | 0.844 (0.018) |
| PSIP | −0.276 (0.120) | 0.779 (0.075) |
| SOGP | −0.452 (0.085) | 0.786 (0.011) |
| GVS | **−0.792 (0.089)** | 0.768 (0.048) |

#### Weak periodic

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | −0.644 (0.042) | **0.811 (0.022)** |
| Naive sparse GP | −0.652 (0.047) | 0.771 (0.014) |
| RST–Greedy | **−0.685 (0.011)** | 0.784 (0.064) |
| PSIP | 0.687 (0.534) | 0.665 (0.035) |
| SOGP | −0.629 (0.039) | 0.742 (0.024) |
| GVS | −0.644 (0.042) | 0.643 (0.111) |

#### Matern

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | −0.733 (0.096) | **0.818 (0.025)** |
| Naive sparse GP | **−0.738 (0.091)** | 0.787 (0.041) |
| RST–Greedy | −0.730 (0.098) | 0.742 (0.065) |
| PSIP | −0.068 (0.315) | 0.685 (0.039) |
| SOGP | −0.691 (0.085) | 0.751 (0.037) |
| GVS | −0.733 (0.096) | 0.688 (0.071) |

#### Sawtooth

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | **0.148 (0.039)** | **0.721 (0.130)** |
| Naive sparse GP | 0.178 (0.041) | 0.658 (0.076) |
| RST–Greedy | 0.166 (0.025) | 0.716 (0.054) |
| PSIP | 0.457 (0.128) | 0.683 (0.027) |
| SOGP | 0.317 (0.134) | 0.441 (0.248) |
| GVS | **0.148 (0.039)** | 0.678 (0.132) |

### Latent RMSE

#### EQ

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | 0.050 (0.003) | **0.901 (0.011)** |
| Naive sparse GP | 0.052 (0.002) | 0.891 (0.008) |
| RST–Greedy | **0.049 (0.003)** | 0.869 (0.023) |
| PSIP | 0.220 (0.041) | 0.781 (0.013) |
| SOGP | 0.107 (0.010) | 0.859 (0.006) |
| GVS | 0.050 (0.003) | 0.770 (0.039) |

#### Weak periodic

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | 0.066 (0.007) | **0.874 (0.012)** |
| Naive sparse GP | **0.065 (0.008)** | 0.836 (0.025) |
| RST–Greedy | 0.066 (0.011) | 0.817 (0.052) |
| PSIP | 0.473 (0.261) | 0.663 (0.043) |
| SOGP | 0.070 (0.006) | 0.825 (0.032) |
| GVS | 0.066 (0.007) | 0.730 (0.050) |

#### Matern

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | 0.060 (0.006) | **0.875 (0.017)** |
| Naive sparse GP | 0.059 (0.005) | 0.858 (0.024) |
| RST–Greedy | **0.058 (0.005)** | 0.775 (0.066) |
| PSIP | 0.333 (0.120) | 0.674 (0.042) |
| SOGP | 0.068 (0.004) | 0.848 (0.021) |
| GVS | 0.060 (0.006) | 0.750 (0.007) |

#### Sawtooth

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | 0.274 (0.026) | 0.718 (0.204) |
| Naive sparse GP | **0.270 (0.031)** | 0.706 (0.203) |
| RST–Greedy | 0.274 (0.026) | 0.561 (0.098) |
| PSIP | 0.406 (0.123) | **0.739 (0.053)** |
| SOGP | 0.274 (0.020) | 0.702 (0.133) |
| GVS | 0.274 (0.026) | 0.590 (0.116) |

The same numbers are drawn in `figures/fig_order_scores.pdf`. The legend column is BP-AUC.

## Layout

| Path | Role |
|---|---|
| `svgp.py` | Kernels, ELBO, predictions, conditional-KL order, training step |
| `data.py` | EQ, Matérn-5/2, weak periodic, and sawtooth tasks on $`[-2,2]`$ |
| `sogp.py` | Csató–Opper sparse online GP |
| `self_test.py` | Finite-difference checks for the ELBO |
| `run_synthetic1d.py` | First nested-prefix comparison |
| `run_control.py` | Objective controls |
| `run_ordering.py` | Order-update experiments |
| `run_score_order.py` | Every-step scores, including conditional KL |
| `run_literature.py` | PSIP, SOGP, and GVS |
| `run_ablation.py` | Prefix training against a full model with the same order |
| `run_comparators.py`, `run_elbo_rerun.py` | Earlier comparator and ELBO runs |
| `results/` | Saved metrics and markdown tables |
| `figures/` | Plot scripts and the figures they produced |

Saved figures:

- `figures/fig_ordering.pdf`
- `figures/fig_order_scores.pdf`
- `figures/fig_ablation.pdf`
- `figures/fig_budget_summary.pdf`
- `figures/fig_budget_nlpd_by_task.pdf`

## Check the install

From this directory:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python self_test.py
```

## Rerun a synthetic comparison

Tasks are `eq`, `matern`, `weak_periodic`, and `sawtooth`. A full run uses 1000 steps. The commands below keep the defaults.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

python run_ordering.py
python run_score_order.py --score ckl
python run_score_order.py --full-budget ckl
python run_score_order.py --full-budget rst
python run_score_order.py --merge-full
python run_literature.py --method psip
python run_literature.py --method sogp
python run_literature.py --method gvs
python run_literature.py --merge
python run_ablation.py
python run_ablation.py --merge

python figures/gen_fig_ordering.py
python figures/gen_fig_order_scores.py
python figures/gen_fig_ablation.py
python figures/gen_fig_budget.py
```

Tables are written next to the records, for example `results/ordering/results.md`, `results/order_scores/results.md`, `results/order_scores/literature.md`, and `results/order_scores/ablation.md`.
