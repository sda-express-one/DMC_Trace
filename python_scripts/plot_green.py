#!/usr/bin/env python3
"""Plot the imaginary-time Green function G(p, tau) written by DMC_Trace.

Three file types, recognised from their first header line (all values are tr G or G_nm itself: the e^{mu tau}
sampling factor is already removed, so a single pole decays as Z e^{-E0 tau}):

    <prefix>_green_trace.dat     "# Green function trace estimator"
        columns: tau_centre hits trace err                          (histogram of tr G)
    <prefix>_green_func.dat      "# matrix Green function estimator"
        columns: tau_centre hits G_00 err ... G_22 err trace err G_xx err ... G_zz err
                                                                    (band basis n, m and orbital basis x, y, z)
    <prefix>_green_laguerre.dat  "# Green function trace estimator, Laguerre coefficients ..."
        header: tau_max, mu, s, l_max; block 0 "l g_l err", block 1 "tau trace err" (rebuilt at the bin centres)
        optional <prefix>_green_laguerre_cov.dat: jackknife covariance of the g_l, (l_max+1)^2 dense matrix

Laguerre curves are rebuilt here on a dense tau grid,
    tr G(tau) = e^{-mu tau} sum_{l <= L} g_l phi_l(tau),   phi_l(tau) = sqrt(s) e^{-s tau/2} L_l(s tau),
with errors e^{-mu tau} sqrt(diag(B_L C_L B_L^T)) from the covariance (no band without the _cov file). --L draws
one curve per truncation L <= l_max (default: l_max).

Figure: top panel G(tau) (log scale unless --linear), histograms as error-bar markers, Laguerre curves as a line
with a 1-sigma band. If a Laguerre file is given, a second panel shows |g_l| +- err on a log scale (open markers:
g_l < 0) with 2 err dotted: coefficients below it are compatible with zero, a guide for the truncation.
The tau axis ends at the last point with err < |G| (beyond it the run has essentially no samples and e^{-mu tau}
only amplifies noise), the G axis spans those points; --tmax overrides both. Bands are drawn only where err < |G|.

--fit TMIN TMAX fits G = Z e^{-E0 tau} on that window for every series and prints E0, Z, chi^2/dof:
  - histogram bins: independent errors (scipy curve_fit, absolute_sigma);
  - Laguerre curves: the grid points are strongly correlated (they share the g_l) and their covariance
    E B C B^T E is of low rank, so the fit is a generalized least squares with an eigen-truncated pseudo-inverse
    (eigenvalues > 1e-10 of the largest); dof = that rank - 2.

Usage:
    python plot_green.py FILE [FILE ...] [-o out.pdf] [--comp trace G_00 G_xx] [--L 10 20]
                                         [--fit TMIN TMAX] [--tmax T] [--linear] [--show]
Without -o the figure is saved next to the first input as <name>.pdf.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, least_squares

# validated categorical palette (as plot_rhat.py), assigned in fixed order (never cycled)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#3a3a38"
MUTED = "#8a897f"
FIT_SHADE = "#ecebe4"

N_DENSE = 400            # tau grid of the rebuilt Laguerre curves
PINV_RTOL = 1e-10        # eigenvalue cut of the pseudo-inverse in the correlated fit


# ---------------------------------------------------------------- reading

def header_lines(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.startswith("#")]


def file_kind(path):
    first = header_lines(path)[:1]
    if not first:
        sys.exit(f"{path}: no header")
    h = first[0]
    if "Laguerre coefficients" in h:
        return "laguerre"
    if "matrix Green function estimator" in h:
        return "func"
    if "Green function trace estimator" in h:
        return "trace"
    sys.exit(f"{path}: not a green_trace / green_func / green_laguerre file ({h!r})")


def header_value(lines, key):
    for line in lines:
        m = re.search(rf"\b{re.escape(key)} = (\S+)", line)
        if m:
            return float(m.group(1))
    sys.exit(f"header key {key!r} not found")


def read_blocks(path):
    """Numeric blocks separated by blank lines (gnuplot index), '#' lines dropped."""
    blocks, cur = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                if cur:
                    blocks.append(np.array(cur))
                    cur = []
                continue
            if s.startswith("#"):
                continue
            cur.append([float(x) for x in s.split()])
    if cur:
        blocks.append(np.array(cur))
    return blocks


def read_histogram(path, kind, comps):
    """Series of a green_trace or green_func file: (name, tau, value, err) for each requested component."""
    data = read_blocks(path)[0]
    if kind == "trace":
        names = ["trace"]
    else:
        col_line = next((l for l in header_lines(path) if l.startswith("# columns:")), None)
        if col_line is None:
            sys.exit(f"{path}: no '# columns:' line")
        tokens = col_line.split(":", 1)[1].split()
        names = [t for t in tokens[2:] if t != "err"]      # after tau_centre hits: value/err pairs
    out = []
    for c in comps:
        if c not in names:
            if kind == "trace" and c != "trace":
                continue                                   # a trace file carries only the trace
            sys.exit(f"{path}: no component {c!r} (available: {' '.join(names)})")
        i = 2 + 2 * names.index(c)
        out.append((c, data[:, 0], data[:, i], data[:, i + 1]))
    return out


def laguerre_psi(x, lmax):
    """psi_l(x) = e^{-x/2} L_l(x), l = 0..lmax, for an array x: shape (len(x), lmax+1). Three-term recurrence."""
    x = np.asarray(x, dtype=float)
    psi = np.empty((x.size, lmax + 1))
    psi[:, 0] = np.exp(-0.5 * x)
    if lmax >= 1:
        psi[:, 1] = (1.0 - x) * psi[:, 0]
    for l in range(1, lmax):
        psi[:, l + 1] = ((2 * l + 1 - x) * psi[:, l] - l * psi[:, l - 1]) / (l + 1)
    return psi


class Laguerre:
    def __init__(self, path):
        lines = header_lines(path)
        self.path = Path(path)
        self.tau_max = header_value(lines, "tau_max")
        self.mu = header_value(lines, "mu")
        self.s = header_value(lines, "s")
        self.l_max = int(header_value(lines, "l_max"))
        blocks = read_blocks(path)
        self.g, self.g_err = blocks[0][:, 1], blocks[0][:, 2]
        self.grid = blocks[1]                                       # tau trace err at the bin centres
        cov_path = self.path.with_name(self.path.stem + "_cov" + self.path.suffix)
        self.cov = np.loadtxt(cov_path) if cov_path.exists() else None
        if self.cov is None:
            print(f"{path}: no {cov_path.name}: Laguerre curves without error band, fits uncorrelated", file=sys.stderr)

    def basis(self, tau, L):
        return np.sqrt(self.s) * laguerre_psi(self.s * np.asarray(tau), L)

    def rebuild(self, tau, L):
        """tr G at tau from l <= L: value, error (None without covariance), full covariance (None without)."""
        B = self.basis(tau, L)
        scale = np.exp(-self.mu * np.asarray(tau))
        value = scale * (B @ self.g[: L + 1])
        if self.cov is None:
            return value, None, None
        C = self.cov[: L + 1, : L + 1]
        BC = B @ C
        err = scale * np.sqrt(np.maximum(np.einsum("ij,ij->i", BC, B), 0.0))
        cov = scale[:, None] * (BC @ B.T) * scale[None, :]
        return value, err, cov


# ---------------------------------------------------------------- fitting

def model(tau, z, e0):
    return z * np.exp(-e0 * tau)


def initial_guess(tau, y):
    ok = y > 0
    if ok.sum() < 2:
        return 1.0, 0.0
    slope, intercept = np.polyfit(tau[ok], np.log(y[ok]), 1)
    return np.exp(intercept), -slope


def fit_independent(tau, y, err):
    ok = np.isfinite(y) & np.isfinite(err) & (err > 0)
    tau, y, err = tau[ok], y[ok], err[ok]
    if tau.size < 3:
        return None
    p, pcov = curve_fit(model, tau, y, p0=initial_guess(tau, y), sigma=err, absolute_sigma=True, maxfev=20000)
    chi2 = float(np.sum(((y - model(tau, *p)) / err) ** 2))
    return p, np.sqrt(np.diag(pcov)), chi2, tau.size - 2


def fit_correlated(tau, y, cov):
    """GLS with the eigen-truncated pseudo-inverse of cov: residuals whitened in the kept eigen-directions."""
    w, V = np.linalg.eigh(cov)
    keep = w > PINV_RTOL * w.max()
    W = (V[:, keep] / np.sqrt(w[keep])).T                  # rows: whitening directions
    rank = int(keep.sum())
    if rank < 3:
        return None
    res = least_squares(lambda p: W @ (y - model(tau, *p)), x0=initial_guess(tau, y), method="lm")
    J = res.jac
    pcov = np.linalg.pinv(J.T @ J)          # pinv: a degenerate direction gives a 0 error, not a crash
    chi2 = float(res.fun @ res.fun)
    return res.x, np.sqrt(np.diag(pcov)), chi2, rank - 2


# ---------------------------------------------------------------- plotting

def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "Plot the Green function G(p, tau).").split("\n")[0])
    ap.add_argument("files", nargs="+", help="green_trace / green_func / green_laguerre .dat file(s)")
    ap.add_argument("-o", "--output", help="output figure; default <first input stem>.pdf")
    ap.add_argument("--comp", nargs="+", default=["trace"],
                    help="green_func components (e.g. trace G_00 G_xx; default trace)")
    ap.add_argument("--L", nargs="+", type=int, help="Laguerre truncations L <= l_max (default l_max)")
    ap.add_argument("--fit", nargs=2, type=float, metavar=("TMIN", "TMAX"), help="fit Z e^{-E0 tau} on [TMIN, TMAX]")
    ap.add_argument("--tmax", type=float, help="right end of the tau axis (default: last point with err < |G|)")
    ap.add_argument("--linear", action="store_true", help="linear G axis (default log)")
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    args = ap.parse_args()

    # series: dict(label, tau, value, err, cov, kind)
    series, laguerres = [], []
    multi = len(args.files) > 1
    for path in args.files:
        kind = file_kind(path)
        stem = Path(path).stem
        if kind == "laguerre":
            lag = Laguerre(path)
            laguerres.append((stem, lag))
            tau = np.linspace(lag.tau_max / N_DENSE, lag.tau_max, N_DENSE)
            for L in (args.L or [lag.l_max]):
                if not 0 <= L <= lag.l_max:
                    sys.exit(f"{path}: L = {L} outside 0..{lag.l_max}")
                value, err, cov = lag.rebuild(tau, L)
                sr = dict(label=f"{stem}  Laguerre L = {L}", tau=tau, value=value, err=err, cov=cov, kind="curve")
                if cov is None and L == lag.l_max:
                    # no covariance: fit the file's own rebuilt points (bin centres) with their diagonal errors
                    sr["fit_points"] = (lag.grid[:, 0], lag.grid[:, 1], lag.grid[:, 2])
                series.append(sr)
        else:
            for name, tau, value, err in read_histogram(path, kind, args.comp):
                label = f"{stem}  {name}" if (multi or len(args.comp) > 1) else name
                series.append(dict(label=label, tau=tau, value=value, err=err, cov=None, kind="points"))
    if not series:
        sys.exit("nothing to plot")
    if len(series) > len(SERIES_COLORS):
        sys.exit(f"{len(series)} series: at most {len(SERIES_COLORS)} (split the plot)")

    two = bool(laguerres)
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(7.5, 7.5 if two else 4.8), squeeze=False,
                             gridspec_kw=dict(height_ratios=[3, 2] if two else [1]))
    ax = axes[0, 0]

    if args.fit:
        ax.axvspan(*args.fit, color=FIT_SHADE, zorder=0, lw=0)

    sig_tau, sig_lo, sig_hi = [], [], []      # significant points (err < |G|): they set the axis ranges
    for i, sr in enumerate(series):
        color, marker = SERIES_COLORS[i], SERIES_MARKERS[i]
        tau, y, err = sr["tau"], sr["value"], sr["err"]
        sig = np.isfinite(y) & (y > 0 if not args.linear else np.ones_like(y, bool))
        if err is not None:
            sig &= (err > 0) & (err < np.abs(y))
        if sig.any():
            e = err[sig] if err is not None else 0.0
            sig_tau.append(tau[sig].max())
            sig_lo.append(np.min(y[sig] - e) if args.linear else np.min(np.maximum(y[sig] - e, 0.5 * y[sig])))
            sig_hi.append(np.max(y[sig] + e))
        if not args.linear:
            bad = ~(y > 0)
            if bad.any():
                print(f"{sr['label']}: {bad.sum()} non-positive points not shown on the log axis", file=sys.stderr)
        if sr["kind"] == "points":
            ax.errorbar(tau, y, yerr=err, fmt=marker, ms=4, color=color, ecolor=color, elinewidth=1, capsize=0,
                        mec="white", mew=0.6, label=sr["label"], zorder=3)
        else:
            ax.plot(tau, y, color=color, lw=2, label=sr["label"], zorder=2)
            if err is not None:      # band only where it is significant: a band through 0 is meaningless on a log axis
                ax.fill_between(tau, y - err, y + err, where=sig, color=color, alpha=0.25, lw=0, zorder=1)

        if args.fit:
            t0, t1 = args.fit
            w = (tau >= t0) & (tau <= t1)
            if sr["cov"] is not None:
                r = fit_correlated(tau[w], y[w], sr["cov"][np.ix_(w, w)])
                how = "correlated"
            elif err is not None:
                r = fit_independent(tau[w], y[w], err[w])
                how = "independent"
            elif "fit_points" in sr:
                ft, fy, fe = sr["fit_points"]
                fw = (ft >= t0) & (ft <= t1)
                r = fit_independent(ft[fw], fy[fw], fe[fw])
                how = "file's grid points, correlation ignored: errors and chi2 too optimistic"
            else:
                r, how = None, "no errors"
            if r is None:
                print(f"{sr['label']}: fit on [{t0}, {t1}]: not enough points ({how})")
                continue
            (z, e0), (dz, de0), chi2, dof = r
            print(f"{sr['label']}: fit on [{t0:g}, {t1:g}] ({how}): E0 = {e0:.6f} +- {de0:.6f}   "
                  f"Z = {z:.6f} +- {dz:.6f}   chi2/dof = {chi2:.1f}/{dof} = {chi2 / max(dof, 1):.2f}")
            tt = np.linspace(t0, t1, 100)
            ax.plot(tt, model(tt, z, e0), color=INK, lw=1, ls="--", zorder=4)

    if not args.linear:
        ax.set_yscale("log")
    t_right = args.tmax if args.tmax else (1.05 * max(sig_tau) if sig_tau else None)
    if t_right:
        ax.set_xlim(0, t_right)
    if sig_lo and not args.tmax:
        lo, hi = min(sig_lo), max(sig_hi)
        ax.set_ylim((lo / 2, hi * 2) if not args.linear else (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)))
    ax.set_xlabel(r"$\tau$", color=INK)
    ax.set_ylabel(r"$G(p, \tau)$", color=INK)
    ax.grid(True, color="#e6e5df", lw=0.6)
    ax.tick_params(colors=INK)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    handles, labels = ax.get_legend_handles_labels()
    if args.fit:
        handles.append(Line2D([], [], color=INK, lw=1, ls="--"))
        labels.append(r"fit $Z e^{-E_0\tau}$ (shaded window)")
    if len(series) > 1 or args.fit:
        ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK)
    else:
        ax.set_title(series[0]["label"], color=INK, fontsize=10)

    if two:
        bx = axes[1, 0]
        for i, (stem, lag) in enumerate(laguerres):
            # colour of the file's first curve, so a file keeps its identity across the panels
            j = next(k for k, sr in enumerate(series) if sr["label"].startswith(stem + "  Laguerre"))
            color, marker = SERIES_COLORS[j], SERIES_MARKERS[j]
            l = np.arange(lag.g.size)
            neg = lag.g < 0
            # no error bars (they reach 0 on a log axis): the 2 sigma line is the noise level
            bx.plot(l[~neg], np.abs(lag.g[~neg]), marker, ms=5, color=color, mec="white", mew=0.6, ls="none",
                    label=f"{stem}  $g_l > 0$")
            bx.plot(l[neg], np.abs(lag.g[neg]), marker, ms=5, color=color, mfc="white", mec=color, ls="none",
                    label=f"{stem}  $g_l < 0$")
            bx.plot(l, 2 * lag.g_err, color=color, lw=1, ls=":", label=f"{stem}  $2\\,\\sigma_l$")
            for L in (args.L or []):
                bx.axvline(L, color=MUTED, lw=0.8, ls="--", zorder=0)
        bx.set_yscale("log")
        bx.set_xlabel(r"$l$", color=INK)
        bx.set_ylabel(r"$|g_l|$", color=INK)
        bx.grid(True, color="#e6e5df", lw=0.6)
        bx.tick_params(colors=INK)
        for sp in bx.spines.values():
            sp.set_color(MUTED)
        bx.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.tight_layout()
    out = Path(args.output) if args.output else Path(args.files[0]).with_suffix(".pdf")
    fig.savefig(out)
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
