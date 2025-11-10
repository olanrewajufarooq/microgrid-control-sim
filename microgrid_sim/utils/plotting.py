"""
microgrid_sim/utils/plotting.py

Standard plotting utilities with:
- Fixed, high-contrast palette (first 5 = black, green, red, yellow, blue)
- Consistent colors per series across ALL subplots in a run
- Separate Generation vs Consumption panels
- "Net vs Grid" subplot (net before grid, grid required (slack), grid component actual)
- Clear subplot titles
- Optional saving of conference-quality figures (PNG/PDF, 300 dpi)
  to <base_dir>/<sim_name>/ (combined + individuals)

Conventions
-----------
Power (kW): generation > 0, load/charging < 0
Cash flow per step: NEGATIVE = expense (you paid), POSITIVE = revenue (you received)

Public API
----------
plot_simulation(
    df,
    actions=None,
    xlim_auto=True,
    sim_name=None,
    base_dir="plots",
    save=False,
    formats=("png","pdf"),
    style="conference"
)
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# Fixed, high-contrast palette
# --------------------------
_BASE_PALETTE = [
    "#000000",  # black
    "#2CA02C",  # green
    "#D62728",  # red
    "#FFBF00",  # yellow (gold)
    "#1F77B4",  # blue
    "#E377C2",  # magenta
    "#17BECF",  # cyan/teal
    "#8C564B",  # brown
    "#9467BD",  # purple
    "#7F7F7F",  # gray
]

_COLOR_OVERRIDES = {
    "grid":                 "#000000",  # black (component actual)
    "grid_exp":             "#000000",  # black (negative in consumption plot)
    "grid_required":        "#000000",  # black (negative in consumption plot)
    "grid_slack_kw":        "#7F7F7F",  # gray (slack requirement)
    "pv":                   "#FFBF00",  # yellow
    "wind":                 "#1F77B4",  # blue
    "hydro":                "#17BECF",  # cyan/teal
    "diesel":               "#D62728",  # red
    "house":                "#2CA02C",  # green
    "house1":               "#17BECF",  # cyan
    "house2":               "#FFBF00",  # yellow
    "house3":               "#1F77B4",  # blue
    "factory":              "#D62728",  # red
    "bat":                  "#17BECF",  # cyan
    "total_cashflow":       "#9467BD",  # purple
    "unmet_load_kw":        "#D62728",  # red
    "curtailed_gen_kw":     "#7F7F7F",  # gray
    "downtime":             "#000000",  # black
    "net_power_unbalanced": "#2CA02C",  # green (net before grid)
}

def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


# --------------------------
# Utilities
# --------------------------
def _infer_components(df_cols) -> Tuple[List[str], bool]:
    comps = []
    for c in df_cols:
        if c.endswith("_power") and c != "grid_power":
            comps.append(c.replace("_power", ""))
    has_grid = "grid_power" in df_cols
    seen, ordered = set(), []
    for c in comps:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered, has_grid

def _set_xaxis(axs, n_steps: int):
    for ax in axs:
        ax.set_xlim(0, max(0, n_steps - 1))
    tick_step = max(1, max(1, n_steps) // 12)
    axs[-1].set_xticks(list(range(0, n_steps, tick_step)))

def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _apply_conference_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "lines.linewidth": 2.0,
        "figure.autolayout": False,
    })

def _collect_series_names(df, actions: Optional[List[Dict[str, float]]], has_grid: bool) -> List[str]:
    names = set()
    comps, _ = _infer_components(df.columns)
    for n in comps:
        names.add(n)
    if has_grid:
        names.add("grid")
        names.add("grid_import")
    for special in ["total_cashflow", "unmet_load_kw", "curtailed_gen_kw", "downtime",
                    "net_power_unbalanced", "grid_slack_kw"]:
        if special in df.columns:
            names.add(special)
    if actions is not None:
        for a in actions:
            for k in a.keys():
                names.add(k)
    ordered = []
    for key in _COLOR_OVERRIDES.keys():
        if key in { _normalize_name(n) for n in names }:
            for n in names:
                if _normalize_name(n) == key and n not in ordered:
                    ordered.append(n)
    for n in sorted(names, key=lambda s: s.lower()):
        if n not in ordered:
            ordered.append(n)
    return ordered

def _build_color_map(df, actions: Optional[List[Dict[str, float]]]) -> Dict[str, str]:
    has_grid = "grid_power" in df.columns
    names = _collect_series_names(df, actions, has_grid)
    palette = list(_BASE_PALETTE)
    used_colors = set()
    color_map: Dict[str, str] = {}

    # overrides first
    for n in names:
        key = _normalize_name(n)
        if key in _COLOR_OVERRIDES:
            color = _COLOR_OVERRIDES[key]
            color_map[n] = color
            used_colors.add(color)

    # remaining from palette
    palette_cycle = [c for c in palette if c not in used_colors] or palette
    idx = 0
    for n in names:
        if n in color_map:
            continue
        color_map[n] = palette_cycle[idx % len(palette_cycle)]
        idx += 1

    return color_map


# --------------------------
# Plotters (accept color_map)
# --------------------------
def _plot_generation(ax, df, comps: List[str], has_grid: bool, color_map: Dict[str, str]):
    x = np.arange(len(df))
    any_plotted = False
    for cname in comps:
        col = f"{cname}_power"
        if col in df.columns:
            y = np.clip(df[col].values.astype(float), 0, None)
            if np.any(np.nan_to_num(y) > 0):
                ax.plot(x, y, label=cname, color=color_map.get(cname, "#000000"))
                any_plotted = True
    ax.set_title("Generation Power (kW) — positive portions only")
    ax.set_ylabel("kW")
    if any_plotted:
        ax.legend(ncol=4)

def _plot_consumption(ax, df, comps: List[str], has_grid: bool, color_map: Dict[str, str]):
    x = np.arange(len(df))
    any_plotted = False
    for cname in comps:
        col = f"{cname}_power"
        if col in df.columns:
            y = df[col].values.astype(float)
            y_neg = np.where(y < 0, y, 0.0)
            if np.any(np.nan_to_num(y_neg) < 0):
                ax.plot(x, y_neg, label=cname, color=color_map.get(cname, "#000000"))
                any_plotted = True
    if has_grid and "grid_power" in df.columns:
        g = df["grid_power"].values.astype(float)
        grid_import_as_neg = -np.clip(g, 0, None)
        if np.any(np.nan_to_num(grid_import_as_neg) < 0):
            ax.plot(x, grid_import_as_neg, label="grid_import",
                    color=color_map.get("grid_import", "#000000"))
            any_plotted = True
    ax.set_title("Consumption Power (kW) — negative portions & grid imports")
    ax.set_ylabel("kW")
    if any_plotted:
        ax.legend(ncol=4)

def _plot_net_vs_grid(ax, df, color_map: Dict[str, str]):
    """
    Net vs Grid:
    - net_power_unbalanced: sum of all non-grid components (gen > 0, load/charge < 0)
    - grid_slack_kw: what the system asked the grid to do (aggregate slack)  [new, dashed]
    - grid_power: grid component actual (import > 0, export < 0)

    Visual check: ideally net + grid_actual ~ 0 (modulo unmet/curtail).
    """
    x = np.arange(len(df))
    if "net_power_unbalanced" in df.columns:
        ax.plot(x, df["net_power_unbalanced"], label="net_power_unbalanced",
                color=color_map.get("net_power_unbalanced", "#2CA02C"))
    if "grid_slack_kw" in df.columns:
        ax.plot(x, df["grid_slack_kw"], label="grid_required (slack)",
                linestyle="--", color=color_map.get("grid_slack_kw", "#7F7F7F"))
    if "grid_power" in df.columns:
        ax.plot(x, df["grid_power"], label="grid_actual",
                color=color_map.get("grid", "#000000"))
    ax.set_title("Net vs Grid (kW)")
    ax.set_ylabel("kW")
    ax.legend(ncol=4)

def _plot_socs(ax, df, color_map: Dict[str, str]):
    x = np.arange(len(df))
    plotted = False
    for c in df.columns:
        if c.endswith("_soc"):
            name = c.replace("_soc", "")
            ax.plot(x, df[c], label=name, color=color_map.get(name, "#000000"))
            plotted = True
    ax.set_title("State of Charge (SOC)")
    ax.set_ylabel("SOC (0–1)")
    if plotted:
        ax.legend(ncol=4)

def _plot_costs(ax, df, color_map: Dict[str, str]):
    """
    Per-step cash flow by component (from *_cost columns).
    Negative = expense, Positive = revenue.
    """
    x = np.arange(len(df))
    # Prefer common interesting components if present, else plot any *_cost columns
    interesting = [c for c in ["grid", "diesel", "bat", "pv", "wind", "factory", "house"]
                   if f"{c}_cost" in df.columns]
    cols = [f"{c}_cost" for c in interesting] or [c for c in df.columns if c.endswith("_cost")]
    plotted = False
    for col in cols:
        name = col.replace("_cost", "")
        ax.plot(x, df[col], label=name, color=color_map.get(name, "#000000"))
        plotted = True
    ax.set_title("Per-Step Cash Flow (NEG=expense, POS=revenue)")
    ax.set_ylabel("$ / step")
    if plotted:
        ax.legend(ncol=4)

def _plot_cumulative_cash(ax, df, color_map: Dict[str, str]):
    x = np.arange(len(df))
    if "total_cashflow" in df.columns:
        ax.plot(x, df["total_cashflow"].cumsum(),
                color=color_map.get("total_cashflow", "#9467BD"))
    ax.set_title("Cumulative Total Cash Flow")
    ax.set_ylabel("$")

def _plot_unmet_curtailed(ax, df, color_map: Dict[str, str]):
    x = np.arange(len(df))
    plotted = False
    if "unmet_load_kw" in df.columns:
        ax.plot(x, df["unmet_load_kw"], label="Unmet Load (kW)",
                color=color_map.get("unmet_load_kw", "#D62728"))
        plotted = True
    if "curtailed_gen_kw" in df.columns:
        ax.plot(x, df["curtailed_gen_kw"], label="Curtailed Gen (kW)",
                color=color_map.get("curtailed_gen_kw", "#7F7F7F"))
        plotted = True
    if "downtime" in df.columns:
        ax.step(x, df["downtime"], where="post", label="Downtime (0/1)",
                color=color_map.get("downtime", "#000000"), alpha=0.7)
        plotted = True
    ax.set_title("Security of Supply")
    ax.set_ylabel("kW (and 0/1)")
    if plotted:
        ax.legend(ncol=3)

def _plot_actions(ax, actions: List[Dict[str, float]], color_map: Dict[str, str]):
    x = np.arange(len(actions))
    keys = set()
    for a in actions:
        for k in a.keys():
            keys.add(k)
    keys = sorted(keys, key=lambda s: s.lower())
    plotted = False
    for k in keys:
        series = []
        for a in actions:
            v = a.get(k, 0.0)
            if isinstance(v, dict):
                v = v.get("power_setpoint", 0.0)
            series.append(float(v))
        ax.plot(x, series, label=k, color=color_map.get(k, "#000000"))
        plotted = True
    ax.set_title("Actions by Component")
    ax.set_ylabel("Setpoint")
    if plotted:
        ax.legend(ncol=4)


# --------------------------
# Main entry: plot + save
# --------------------------
def plot_simulation(
    df,
    actions: Optional[List[Dict[str, float]]] = None,
    xlim_auto: bool = True,
    sim_name: Optional[str] = None,
    base_dir: str = "plots",
    save: bool = False,
    formats: Iterable[str] = ("png", "pdf"),
    style: Optional[str] = "conference",
):
    """
    Create a multi-panel dashboard and (optionally) save combined + individual figures.

    Returns dict: {"fig": fig, "axes": axes_list, "out_dir": path or None}
    """
    try:
        import pandas as pd
        if not hasattr(df, "columns"):
            df = pd.DataFrame(df)
    except Exception:
        pass

    if style == "conference":
        _apply_conference_style()

    steps = len(df)
    comps, has_grid = _infer_components(df.columns)
    color_map = _build_color_map(df, actions)

    # Combined dashboard:
    # 1) Generation  2) Consumption  3) Net vs Grid  4) SOC  5) per-step cash
    # 6) cumulative cash  7) security  8) actions (opt)
    nrows = 7 + (1 if actions is not None else 0)
    fig, axs = plt.subplots(nrows, 1, figsize=(14, 3*nrows), sharex=True)
    ax_idx = 0

    _plot_generation(axs[ax_idx], df, comps, has_grid, color_map); ax_idx += 1
    _plot_consumption(axs[ax_idx], df, comps, has_grid, color_map); ax_idx += 1
    _plot_net_vs_grid(axs[ax_idx], df, color_map); ax_idx += 1
    _plot_socs(axs[ax_idx], df, color_map); ax_idx += 1
    _plot_costs(axs[ax_idx], df, color_map); ax_idx += 1
    _plot_cumulative_cash(axs[ax_idx], df, color_map); ax_idx += 1
    _plot_unmet_curtailed(axs[ax_idx], df, color_map); ax_idx += 1
    if actions is not None:
        _plot_actions(axs[ax_idx], actions, color_map); ax_idx += 1

    axs[-1].set_xlabel("Step")
    if xlim_auto:
        _set_xaxis(axs, steps)

    fig.suptitle("Microgrid Simulation — Overview", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_dir = None
    if save and sim_name:
        out_dir = os.path.join(base_dir, sim_name)
        indiv_dir = os.path.join(out_dir, "individuals")
        _ensure_dir(indiv_dir)

        # Save combined
        for fmt in formats:
            fig.savefig(os.path.join(out_dir, f"overview.{fmt}"))

        # Individual figures
        def _save_single(plot_fn, filename_root: str):
            f, ax = plt.subplots(1, 1, figsize=(10, 4))
            plot_fn(ax)
            ax.set_xlabel("Step")
            if xlim_auto:
                _set_xaxis([ax], steps)
            f.tight_layout()
            for fmt in formats:
                f.savefig(os.path.join(indiv_dir, f"{filename_root}.{fmt}"))
            plt.close(f)

        _save_single(lambda ax: _plot_generation(ax, df, comps, has_grid, color_map), "generation")
        _save_single(lambda ax: _plot_consumption(ax, df, comps, has_grid, color_map), "consumption")
        _save_single(lambda ax: _plot_net_vs_grid(ax, df, color_map), "net_vs_grid")
        _save_single(lambda ax: _plot_socs(ax, df, color_map), "soc")
        _save_single(lambda ax: _plot_costs(ax, df, color_map), "costs")
        _save_single(lambda ax: _plot_cumulative_cash(ax, df, color_map), "cumulative_cash")
        _save_single(lambda ax: _plot_unmet_curtailed(ax, df, color_map), "security")
        if actions is not None:
            _save_single(lambda ax: _plot_actions(ax, actions, color_map), "actions")

    return {"fig": fig, "axes": list(axs), "out_dir": out_dir}
