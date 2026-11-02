#!/usr/bin/env python3
"""Plot the thermalization history written live by DMC_Trace to <prefix>_rhat.dat.

File format (thermalization_control::openLog / appendLive):
    # checkpoint steps batches <s>_rbulk <s>_rfold <s>_essb <s>_esst ... pass consecutive
    # R-hat thermalization history, ...
    # rhat_max = 1.0100   consecutive = 5   10 chains
    # required ESS bulk/tail = 1000.0 1000.0 (20 split chains)     <- written once the first window exists
    1 1000000 781 1.007199 1.003246 1081.6 2245.4 ... 0 0
    ...
`steps` is per chain; entries are nan before the window exists.

Figure: top panel R-hat (bulk solid/filled, folded dashed/open) per scalar against steps, with the rhat_max
threshold; bottom panel bulk-ESS (solid) and tail-ESS (dashed) on a log scale, with the required ESS. Each scalar
has its own color and marker shape. Checkpoints that passed are shaded. The top axis gives the batch means per
chain in each checkpoint's window (the ESS scales with it, so a smaller window lowers the ESS).

The R-hat axis is capped at 1.020 by default (raised to 1 + 1.5 (rhat_max - 1) if rhat_max is so large that the
threshold would not fit); points above the cap are drawn on the top edge with their value. --rmax sets another cap, --rmax 0 removes it.

A scalar that is constant in every window (e.g. the sign of a run without negative signs) has R-hat = 1 and
ESS = number of draws by construction, which says nothing about convergence; it is drawn in gray and labelled
"constant".

Usage:
    python plot_rhat.py run_rhat.dat [more_rhat.dat ...] [-o out.pdf] [--scalars sign energy] [--x checkpoint]
                                     [--rmax 1.05] [--show]
Without -o each input is saved next to itself as <name>.pdf.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# validated categorical palette, assigned in fixed order (never cycled)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#3a3a38"
MUTED = "#8a897f"
PASS_SHADE = "#e3eedf"

# one marker shape per scalar, same fixed order as the colors: identity is never color-alone
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
CONSTANT_COLOR = "#b4b3aa"

DEFAULT_RMAX = 1.020     # default cap of the R-hat axis

SUFFIXES = ("_rbulk", "_rfold", "_essb", "_esst")


def is_constant(data, scalar, meta):
    """True if the scalar was constant in every window: both R-hat exactly 1 and both ESS equal to the number of
    draws (convergence.cpp returns S = split chains x draws for zero variance; split chains drop an odd batch)."""
    rb, rf, eb, et = (data[scalar + suf] for suf in SUFFIXES)
    w = np.isfinite(rb)
    if not w.any():
        return False
    if not (np.all(rb[w] == 1.0) and np.all(rf[w] == 1.0) and np.all(eb[w] == et[w])):
        return False
    if "chains" in meta:
        draws = meta["chains"] * 2 * np.floor(data["batches"][w] / 2)
        return bool(np.all(np.abs(eb[w] - draws) < 0.5))
    return True


def read_rhat(path):
    """Return (data dict column -> array, meta dict, list of scalar names)."""
    names = None
    meta = {}
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                body = s[1:].strip()
                if names is None:
                    names = body.split()
                    continue
                m = re.search(r"rhat_max\s*=\s*([0-9.eE+-]+)", body)
                if m:
                    meta["rhat_max"] = float(m.group(1))
                m = re.search(r"consecutive\s*=\s*(\d+)", body)
                if m:
                    meta["consecutive"] = int(m.group(1))
                m = re.search(r"(\d+)\s+chains\s*$", body)
                if m and "required" not in body:
                    meta["chains"] = int(m.group(1))
                m = re.search(r"required ESS bulk/tail\s*=\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)(?:\s*\((\d+) split chains\))?", body)
                if m:
                    meta["req_bulk"] = float(m.group(1))
                    meta["req_tail"] = float(m.group(2))
                    if m.group(3):
                        meta["split_chains"] = int(m.group(3))
                continue
            rows.append([float(v) for v in s.split()])   # float('nan') parses 'nan'

    if names is None:
        raise ValueError(f"{path}: no header line")
    if not rows:
        raise ValueError(f"{path}: no checkpoint rows yet")
    table = np.array(rows, dtype=float)
    if table.shape[1] != len(names):
        raise ValueError(f"{path}: {table.shape[1]} columns in the rows but {len(names)} names in the header")

    data = {n: table[:, i] for i, n in enumerate(names)}
    scalars = [n[: -len("_rbulk")] for n in names if n.endswith("_rbulk")]
    for sc in scalars:
        missing = [sc + suf for suf in SUFFIXES if sc + suf not in data]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
    return data, meta, scalars


def shade_passes(ax, x, passed):
    """Shade a cell around every passing checkpoint (bounded by the midpoints to its neighbours)."""
    if x.size == 1:
        edges = np.array([x[0] - 0.5, x[0] + 0.5])
    else:
        mid = 0.5 * (x[1:] + x[:-1])
        edges = np.concatenate(([x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]))
    for lo, hi, p in zip(edges[:-1], edges[1:], passed):
        if p > 0.5:
            ax.axvspan(lo, hi, color=PASS_SHADE, lw=0, zorder=0)
    ax.set_xlim(edges[0], edges[-1])


def plot_file(path, out, scalars_wanted=None, xkey="steps", rmax=None, show=False):
    data, meta, scalars = read_rhat(path)
    if scalars_wanted:
        unknown = [s for s in scalars_wanted if s not in scalars]
        if unknown:
            raise ValueError(f"{path}: unknown scalars {unknown}; available: {scalars}")
        # keep each scalar's color tied to its position in the file, so filtering never repaints the survivors
        selected = [s for s in scalars if s in scalars_wanted]
    else:
        selected = scalars
    n_slots = len(SERIES_COLORS)
    if len(scalars) > n_slots:
        print(f"warning: {len(scalars)} scalars but {n_slots} colors/markers; use --scalars to split them",
              file=sys.stderr)
    # color and marker tied to the position in the file, so filtering never repaints the survivors
    color = {s: SERIES_COLORS[i % n_slots] for i, s in enumerate(scalars)}
    marker = {s: SERIES_MARKERS[i % n_slots] for i, s in enumerate(scalars)}
    constant = {s: is_constant(data, s, meta) for s in selected}
    for s in selected:
        if constant[s]:
            color[s] = CONSTANT_COLOR

    x = data[xkey]
    passed = data["pass"]
    consec = data["consecutive"]

    fig, (ax_r, ax_e) = plt.subplots(2, 1, sharex=True, figsize=(9.0, 7.0), constrained_layout=True)
    for ax in (ax_r, ax_e):
        shade_passes(ax, x, passed)
        ax.grid(True, color="#d8d7d0", lw=0.6, zorder=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=9)

    def draw(ax, s, y_main, y_alt):
        z = 2.5 if constant[s] else 3      # constant scalars behind the informative ones
        lw = 1.2 if constant[s] else 2
        ax.plot(x, y_main, color=color[s], lw=lw, marker=marker[s], ms=5, zorder=z)
        if not constant[s]:                # for a constant scalar both variants coincide
            ax.plot(x, y_alt, color=color[s], lw=1.3, ls="--", marker=marker[s], ms=4, mfc="white", zorder=z)

    # --- R-hat panel
    for s in selected:
        draw(ax_r, s, data[s + "_rbulk"], data[s + "_rfold"])
    rhat_max = meta.get("rhat_max", 1.01)
    ax_r.axhline(rhat_max, color=INK, lw=1.2, zorder=2)
    ax_r.axhline(1.0, color=MUTED, lw=0.8, zorder=2)
    ax_r.set_ylabel(r"split $\hat R$", color=INK)

    # axis cap: by default DEFAULT_RMAX (never below 1 + 1.5 (rhat_max - 1), so the threshold stays visible); below the cap the axis still reaches past rhat_max so the
    # threshold and its label stay inside
    cap = max(DEFAULT_RMAX, 1.0 + 1.5 * (rhat_max - 1.0)) if rmax is None else (rmax if rmax > 0 else None)
    r_all = np.concatenate([data[s + suf] for s in selected for suf in ("_rbulk", "_rfold")])
    r_all = r_all[np.isfinite(r_all)]
    if r_all.size:
        lo, hi = r_all.min(), max(r_all.max(), rhat_max)
        pad = 0.08 * (hi - min(lo, 1.0))
        bottom = min(lo, 1.0) - pad
        top = hi + 2 * pad
        if cap is not None and r_all.max() > cap:
            top = cap
            ax_r.set_ylim(bottom, top)
            # clipped points on the top edge, with their value; stacked when several share a checkpoint
            stack = {}
            for s in selected:
                for suf, filled in (("_rbulk", True), ("_rfold", False)):
                    y = data[s + suf]
                    for xi, yi in zip(x, y):
                        if np.isfinite(yi) and yi > cap:
                            ax_r.plot([xi], [cap], marker=marker[s], ms=5 if filled else 4, color=color[s],
                                      mfc=color[s] if filled else "white", clip_on=False, zorder=4)
                            k = stack.get(xi, 0)
                            stack[xi] = k + 1
                            ax_r.annotate(f"{yi:.4f}", xy=(xi, cap), xytext=(6, -4 - 9 * k), textcoords="offset points",
                                          ha="left", va="top", fontsize=7, color=INK)
        else:
            ax_r.set_ylim(bottom, top)
    ax_r.annotate(f"rhat_max = {rhat_max:g}", xy=(0, rhat_max), xycoords=("axes fraction", "data"),
                  xytext=(4, 3), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=INK)

    # batch means per chain in each window, on the top axis (thinned to about 15 labels)
    top_ax = ax_r.secondary_xaxis("top")
    step = max(1, int(np.ceil(x.size / 15)))
    idx = np.arange(0, x.size, step)
    top_ax.set_xticks(x[idx], labels=[f"{int(b)}" for b in data["batches"][idx]])
    top_ax.tick_params(colors=INK, labelsize=8)
    top_ax.spines["top"].set_color(MUTED)
    top_ax.set_xlabel("batches per chain in the window", color=INK, fontsize=9)

    # --- ESS panel
    for s in selected:
        draw(ax_e, s, data[s + "_essb"], data[s + "_esst"])
    if "req_bulk" in meta:
        ax_e.axhline(meta["req_bulk"], color=INK, lw=1.2, zorder=2)
        if meta["req_tail"] != meta["req_bulk"]:
            ax_e.axhline(meta["req_tail"], color=INK, lw=1.2, ls="--", zorder=2)
            label = f"required bulk / tail = {meta['req_bulk']:g} / {meta['req_tail']:g}"
        else:
            label = f"required ESS = {meta['req_bulk']:g}"
        ax_e.annotate(label, xy=(0, max(meta["req_bulk"], meta["req_tail"])), xycoords=("axes fraction", "data"),
                      xytext=(4, 3), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=INK)
    ax_e.set_yscale("log")
    ax_e.set_ylabel("ESS (all chains)", color=INK)
    ax_e.set_xlabel("steps per chain" if xkey == "steps" else xkey, color=INK)
    if xkey == "steps":
        ax_e.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # --- legends, outside on the right: scalars (color + marker), then the line styles
    scalar_handles = [Line2D([], [], color=color[s], lw=2, marker=marker[s], ms=5,
                             label=f"{s} (constant)" if constant[s] else s) for s in selected]
    style = [Line2D([], [], color=INK, lw=2, marker="o", ms=5, label=r"bulk $\hat R$ / bulk-ESS"),
             Line2D([], [], color=INK, lw=1.3, ls="--", marker="o", ms=4, mfc="white", label=r"folded $\hat R$ / tail-ESS"),
             Patch(facecolor=PASS_SHADE, edgecolor="none", label="checkpoint passed")]
    fig.legend(handles=scalar_handles, loc="outside right upper", frameon=False, fontsize=9, title="scalar",
               title_fontsize=9, labelcolor=INK)
    fig.legend(handles=style, loc="outside right lower", frameon=False, fontsize=8, labelcolor=INK)

    # thermalised: the first checkpoint whose consecutive count reached the requirement (title only)
    n_req = meta.get("consecutive")
    therm_x = None
    if n_req is not None:
        hit = np.nonzero(consec >= n_req)[0]
        if hit.size:
            therm_x = x[hit[0]]
    info = [f"{meta['chains']} chains" if "chains" in meta else None,
            f"thermalised at {xkey} = {therm_x:.4g}" if therm_x is not None else "not thermalised yet"]
    fig.suptitle(Path(path).name + "   (" + ", ".join(i for i in info if i) + ")", fontsize=10, color=INK)

    fig.savefig(out)
    print(f"{path} -> {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="<prefix>_rhat.dat file(s)")
    ap.add_argument("-o", "--output", help="output figure (only with a single input); default <input stem>.pdf")
    ap.add_argument("--scalars", nargs="+", help="plot only these scalars (e.g. sign energy)")
    ap.add_argument("--x", dest="xkey", choices=("steps", "checkpoint"), default="steps", help="x axis (default steps)")
    ap.add_argument("--rmax", type=float, help=f"cap of the R-hat axis (default {DEFAULT_RMAX:.3f}; 0 = no cap)")
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    args = ap.parse_args()

    if args.output and len(args.files) > 1:
        ap.error("-o needs a single input file")
    status = 0
    for f in args.files:
        out = args.output or str(Path(f).with_suffix(".pdf"))
        try:
            plot_file(f, out, args.scalars, args.xkey, args.rmax, args.show)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()
