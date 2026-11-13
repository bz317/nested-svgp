# Nested sparse Gaussian processes

One unwhitened sparse GP is trained for every budget at once. The variational family is $`q(u)=N(m,S)`$ with $`S=LL^\top`$. A budget $`k`$ is a prefix of the inducing set: the first $`k`$ locations, the corresponding mean, and the matching principal submatrix of $`S`$. The full set is unchanged by a reordering of those points.

The primary order is conditional KL. Start from the empty prefix and add the inducing point that most increases $`\mathrm{KL}(q(u_A)\,\|\,p(u_A))`$. That order is recomputed after every training step. Each training step adds the full-set ELBO to the predictive NLPD of one prefix, whose size $`k`$ is drawn uniformly from $`\{1,\ldots,32\}`$.

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
| condKL | −0.783 (0.058) | **0.851 (0.010)** |
| Naive sparse GP | −0.778 (0.066) | 0.817 (0.011) |
| RST–Greedy | **−0.792 (0.087)** | 0.822 (0.021) |
| PSIP | −0.276 (0.120) | 0.747 (0.086) |
| SOGP | −0.452 (0.085) | 0.755 (0.012) |
| GVS | −0.783 (0.058) | 0.739 (0.017) |

#### Weak periodic

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | **−0.715 (0.030)** | **0.814 (0.072)** |
| Naive sparse GP | −0.652 (0.047) | 0.744 (0.015) |
| RST–Greedy | −0.685 (0.011) | 0.758 (0.068) |
| PSIP | 0.687 (0.534) | 0.631 (0.037) |
| SOGP | −0.629 (0.039) | 0.714 (0.026) |
| GVS | **−0.715 (0.030)** | 0.675 (0.093) |

#### Matern

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | −0.734 (0.099) | **0.826 (0.050)** |
| Naive sparse GP | **−0.738 (0.091)** | 0.791 (0.040) |
| RST–Greedy | −0.730 (0.098) | 0.746 (0.064) |
| PSIP | −0.068 (0.315) | 0.690 (0.038) |
| SOGP | −0.691 (0.085) | 0.755 (0.037) |
| GVS | −0.734 (0.099) | 0.705 (0.013) |

#### Sawtooth

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | **0.101 (0.061)** | **0.788 (0.023)** |
| Naive sparse GP | 0.178 (0.041) | 0.630 (0.072) |
| RST–Greedy | 0.166 (0.025) | 0.686 (0.052) |
| PSIP | 0.457 (0.128) | 0.653 (0.025) |
| SOGP | 0.317 (0.134) | 0.423 (0.238) |
| GVS | **0.101 (0.061)** | 0.744 (0.036) |

### Latent RMSE

#### EQ

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | 0.050 (0.003) | **0.900 (0.007)** |
| Naive sparse GP | 0.052 (0.002) | 0.891 (0.008) |
| RST–Greedy | **0.049 (0.003)** | 0.869 (0.023) |
| PSIP | 0.220 (0.041) | 0.782 (0.013) |
| SOGP | 0.107 (0.010) | 0.860 (0.006) |
| GVS | 0.050 (0.003) | 0.760 (0.020) |

#### Weak periodic

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | **0.063 (0.005)** | **0.874 (0.051)** |
| Naive sparse GP | 0.065 (0.008) | 0.836 (0.025) |
| RST–Greedy | 0.066 (0.011) | 0.817 (0.052) |
| PSIP | 0.473 (0.261) | 0.662 (0.043) |
| SOGP | 0.070 (0.006) | 0.825 (0.032) |
| GVS | **0.063 (0.005)** | 0.751 (0.054) |

#### Matern

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | 0.061 (0.006) | **0.879 (0.036)** |
| Naive sparse GP | 0.059 (0.005) | 0.857 (0.025) |
| RST–Greedy | **0.058 (0.005)** | 0.773 (0.066) |
| PSIP | 0.333 (0.120) | 0.671 (0.042) |
| SOGP | 0.068 (0.004) | 0.847 (0.021) |
| GVS | 0.061 (0.006) | 0.749 (0.021) |

#### Sawtooth

| Method | Full | BP-AUC |
|---|---:|---:|
| condKL | **0.255 (0.015)** | **0.817 (0.044)** |
| Naive sparse GP | 0.270 (0.031) | 0.676 (0.194) |
| RST–Greedy | 0.274 (0.026) | 0.538 (0.094) |
| PSIP | 0.406 (0.123) | 0.708 (0.051) |
| SOGP | 0.274 (0.020) | 0.673 (0.128) |
| GVS | **0.255 (0.015)** | 0.656 (0.018) |

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
