# Python scripts

These scripts read the output files of `diagramc`.

They need `numpy` and `matplotlib`, and `plot_green.py` also needs `scipy`. Those packages are installed in the conda
environment `env00`:

```
conda_enable
conda activate env00
python python_scripts/<script>.py ...
conda_disable            # removes conda from the shell completely
```

Without activating the environment you can also call its interpreter directly:
`~/miniconda3/envs/env00/bin/python python_scripts/<script>.py ...`.

Both scripts save the figure next to the (first) input as `<stem>.pdf` unless `-o` is given. `--show` also opens
an interactive window. The colors come from the same fixed palette in both scripts, and each series also gets its
own marker shape, so series can be told apart without color.

| Script | Input | Shows |
|---|---|---|
| `plot_rhat.py` | `<prefix>_rhat.dat` | the thermalization history: R̂ and ESS per monitored scalar |
| `plot_green.py` | `<prefix>_green_trace.dat`, `_green_func.dat`, `_green_laguerre.dat` (+ `_cov.dat`) | G(p, τ), the Laguerre coefficients, and a tail fit for E₀ and Z |

---

## `plot_rhat.py`: thermalization history

```
python plot_rhat.py RUN_rhat.dat [MORE_rhat.dat ...] [-o out.pdf] [--scalars sign energy]
                    [--x steps|checkpoint] [--rmax 1.05] [--show]
```

It reads `<prefix>_rhat.dat`, which `diagramc` appends to at every R̂ checkpoint during thermalization. So the file
can be plotted while the run is still going. The file has one numeric row per checkpoint:
`checkpoint steps batches <s>_rbulk <s>_rfold <s>_essb <s>_esst ... pass consecutive`. The header gives `rhat_max`,
the number of consecutive passes required, and the required ESS.

**Figure**

- **Top panel:** R̂ against steps per chain, for each monitored scalar. Bulk R̂ is solid with filled markers, folded
  R̂ dashed with open markers, and the `rhat_max` threshold is drawn as a line.
- **Bottom panel:** bulk ESS (solid) and tail ESS (dashed) on a log scale, with the required ESS drawn as a line.
- **Shading** marks the checkpoints that passed.
- **Top axis:** the number of batch means per chain in each checkpoint's window. The ESS scales with it.
- **Constant scalars** are drawn in gray and labelled "constant". An example is the sign in a run without negative
  diagrams: it has R̂ = 1 and ESS = number of draws by construction, which says nothing about convergence.

**Options**

| Option | Meaning |
|---|---|
| `--scalars` | plot only these scalars, e.g. `sign energy` |
| `--x` | x axis: `steps` (default) or `checkpoint` |
| `--rmax` | cap of the R̂ axis. The default is 1.020, raised when `rmax` is larger; points above the cap are drawn on the edge with their value. `0` removes the cap |
| `-o` | output file. Only allowed with a single input; several inputs give one PDF each |

---

## `plot_green.py`: Green function

```
python plot_green.py FILE [FILE ...] [-o out.pdf] [--comp trace G_00 G_xx] [--L 10 20]
                     [--fit TMIN TMAX] [--tmax T] [--linear] [--show]
```

Several files are overlaid on the same axes, for example the histogram and the Laguerre estimate from the same run.
The script recognises the file type from the first header line:

| File | Content |
|---|---|
| `_green_trace.dat` | histogram of tr G(p, τ): `tau_centre hits trace err` |
| `_green_func.dat` | histogram of the matrix G: `G_00 … G_22` (band basis), `trace`, `G_xx … G_zz` (orbital basis), each followed by its error |
| `_green_laguerre.dat` | Laguerre coefficients g_l (block 0) and tr G rebuilt at the bin centres (block 1) |
| `_green_laguerre_cov.dat` | jackknife covariance of the g_l. It is found automatically next to the Laguerre file and is optional (older runs don't have it) |

All files hold tr G (or G_nm) itself, with the e^{μτ} sampling factor already removed. A single pole therefore
decays as Z e^{−E₀τ}.

**Laguerre curves.** These are rebuilt on a 400-point τ grid from the coefficients:

    G(τ) = e^{−μτ} Σ_{l≤L} g_l φ_l(τ),   φ_l(τ) = √s e^{−sτ/2} L_l(sτ)

- Errors: e^{−μτ} √diag(B C Bᵀ), with B_jl = φ_l(τ_j) and C the covariance of the coefficients.
- `--L` draws one curve per truncation L ≤ l_max. The default is l_max.
- Without the `_cov` file the curve has no error band.

**Figure**

- **Top panel:** G(τ) on a log axis (`--linear` for a linear one).
  - Histograms are drawn as error-bar markers; Laguerre curves as a line with a ±1σ band.
  - The axes cover only the significant points, those with err < |G|. Beyond the last one the run has almost no
    samples, and e^{−μτ} only amplifies the noise.
  - `--tmax` sets the right end of the τ axis by hand.
- **Second panel,** when a Laguerre file is given: |g_l| on a log axis.
  - Open markers show g_l < 0.
  - The dotted line is 2σ_l. Coefficients below it are compatible with zero, which guides the choice of truncation L.
  - The `--L` values are marked as vertical lines.

**Fit (`--fit TMIN TMAX`)**

The script fits G = Z e^{−E₀τ} on the window for every series, and prints E₀, Z and χ²/dof:

- **Histogram bins** are treated as independent (`scipy.optimize.curve_fit` with absolute errors).
- **Laguerre curves** use the full covariance of the grid points. All the points share the same g_l, so they are
  strongly correlated, and their covariance has low rank. The fit is a generalized least squares with an
  eigen-truncated pseudo-inverse: it keeps the eigenvalues above 1e-10 of the largest, and dof = kept rank − 2.
- **Laguerre files without `_cov`** are fitted on the file's own rebuilt points, with their diagonal errors. The
  script then warns that the errors and χ² are too optimistic.

Choose the window inside the sampled region: the part where the points still have small relative errors. The
sampling density falls like e^{−(E₀−μ)τ}.

**Options**

| Option | Meaning |
|---|---|
| `--comp` | `green_func` components to plot, e.g. `trace G_00 G_xx`; default `trace`. Ignored for the other two file types |
| `--L` | Laguerre truncations, one curve each; default l_max |
| `--fit TMIN TMAX` | fit window for Z e^{−E₀τ} |
| `--tmax` | right end of the τ axis |
| `--linear` | linear G axis |

At most 8 series fit in one plot, one per palette color. Beyond that the script stops; split the plot.

**Examples**

```
# histogram vs Laguerre at several truncations, with the tail fit
python plot_green.py out/run_green_trace.dat out/run_green_laguerre.dat --L 10 20 40 --fit 3 15

# matrix elements in the band and orbital bases
python plot_green.py out/run_green_func.dat --comp G_00 G_11 G_xx G_zz
```
