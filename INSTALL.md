# Install

The code is NumPy and SciPy. Matplotlib is used only to draw the figures. Python 3.10 or later is enough. The runs that produced the saved results used Python 3.10, NumPy 1.26.4, SciPy 1.15.3, and Matplotlib 3.10.9.

No GPU and no extra packages are required.

## Virtual environment

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy>=1.26" "scipy>=1.11" "matplotlib>=3.8"
```

Leave the virtual environment with `deactivate`.

## Conda

```bash
conda create -n svgp python=3.10
conda activate svgp
conda install numpy scipy matplotlib
```

## Check

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python self_test.py
```

Set one thread for the BLAS libraries before a training run. Several processes on the same machine otherwise oversubscribe the cores. Figure scripts should also set `MPLBACKEND=Agg` when no display is available.

Training commands are in [README.md](README.md).
