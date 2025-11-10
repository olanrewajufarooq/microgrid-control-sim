"""
microgrid_sim/utils/plotting.py

Standard plotting utilities with:
- Time-based x-axis (hours or days with sub-hour ticks)
- Fixed, high-contrast palette
- Consistent colors per series across ALL subplots
- Standard subplots for Generation, Consumption, Grid, SOC, etc.
- Optional saving of conference-quality figures (PNG/PDF, 300 dpi)

Public API
----------
plot_simulation(
    df,
    actions=None,
    sim_dt_minutes=60,  # <-- Used to calculate time axis
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
import matplotlib.ticker as ticker  # <-- Import the ticker module


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
    # --- Grid ---
    "grid":                 "#000000",  # black (component actual)
    "grid_exp":             "#000000",  # (alias for grid)
    "grid_import":          "#000000",  # (alias for grid)
    "grid_slack_kw":        "#7F7F7F",  # gray (slack requirement)

    # --- Generation ---
    "pv":                   "#FFBF00",  # yellow
    "wind":                 "#1F77B4",  # blue
    "diesel":               "#D62728",  # red
    "bat":                  "#E377C2",  # magenta

    # --- Loads ---
    "factory":              "#8C564B",  # brown
    "house1":               "#2CA02C",  # green
    "house2":               "#17BECF",  # cyan
    "house3":               "#9467BD",  # purple

    # --- KPIs ---
    "total_cashflow":       "#9467BD",  # purple
    "unmet_load_kw":        "#D62728",  # red
    "curtailed_gen_kw":     "#7F7F7F",  # gray
    "downtime":             "#000000",  # black
    "net_power_unbalanced": "#2CA02C",  # green (net before grid)
}

def _normalize_name(name: str) -> str:
    """Helper to standardize component names for color mapping."""
    return str(name).strip().lower()


# --------------------------
# Utilities
# --------------------------
def _infer_components(df_cols) -> Tuple[List[str], bool]:
    """Infers component names from DataFrame columns."""
    comps = []
    has_grid = False
    for c in df_cols:
        if c.endswith("_power"):
            if c.startswith("grid"):
                has_grid = True
            # Add all non-grid components
            if not c.startswith("grid"):
                comps.append(c.replace("_power", ""))

    seen, ordered = set(), []
    for c in comps:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered, has_grid

def _set_xaxis_time(axs: List[plt.Axes], x_axis_data: np.ndarray, x_label: str, total_hours: float):
    """
    Helper to set the new time-based x-axis.
    Uses Major/Minor ticks for multi-day plots.
    """
    if not x_axis_data.any():
        return

    # Set x-limits based on the data (which is in hours)
    for ax in axs:
        ax.set_xlim(x_axis_data[0], x_axis_data[-1])

    # Get the bottom axis (the only one that needs labels)
    bottom_ax = axs[-1]

    if total_hours <= 48: # 0-2 days, show simple hours
        tick_step = max(1, int(total_hours / 12)) * 2 # e.g., 2, 4, 6, 8
        ticks = np.arange(0, total_hours + 1, tick_step)

        bottom_ax.set_xticks(ticks)
        bottom_ax.set_xticklabels([f"{t:.0f}h" for t in ticks]) # e.g., "12h"
        bottom_ax.set_xlabel("Time (Hours)")

    else: # > 2 days, show days with hour sub-ticks

        # --- Set Major Ticks (Days) ---
        # Set a major tick every 24 hours (at 0, 24, 48, ...)
        bottom_ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

        # Format the major tick labels as "1", "2", "3"
        def day_formatter(hour, pos):
            # The tick at 0h is Day 1, 24h is Day 2, etc.
            return f"{int(hour / 24) + 1}"

        bottom_ax.xaxis.set_major_formatter(ticker.FuncFormatter(day_formatter))

        # --- Set Minor Ticks (Hours) ---
        # Set a minor tick every 6 hours (at 6, 12, 18, 30, 36, 42, ...)
        bottom_ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))

        # Format the minor tick labels as "6h", "12h", "18h"
        def hour_formatter(hour, pos):
            hour_of_day = hour % 24
            if hour_of_day == 0:
                return "" # Don't label the major ticks (0h, 24h, etc.)
            return f"{int(hour_of_day)}h" # Label as "6h", "12h", "18h"

        bottom_ax.xaxis.set_minor_formatter(ticker.FuncFormatter(hour_formatter))

        # Adjust label appearance
        bottom_ax.tick_params(axis='x', which='major', labelsize=10, length=6, width=1.2)
        bottom_ax.tick_params(axis='x', which='minor', labelsize=8, labelcolor='#555', length=3)

        bottom_ax.set_xlabel(x_label) # "Time (Days)"

def _ensure_dir(path: str):
    """Ensures a directory exists."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _apply_conference_style():
    """Applies a clean style for plots."""
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
    """Collects all unique series names for consistent color mapping."""
    names = set()
    comps, _ = _infer_components(df.columns)
    for n in comps:
        names.add(n)

    # Add all grid-like components
    for c in df.columns:
        if c.startswith("grid") and c.endswith("_power"):
            names.add(c.replace("_power", ""))

    if has_grid:
        names.add("grid_import") # Special name for consumption plot

    for special in ["total_cashflow", "unmet_load_kw", "curtailed_gen_kw", "downtime",
                    "net_power_unbalanced", "grid_slack_kw"]:
        if special in df.columns:
            names.add(special)

    if actions is not None:
        for a_dict in actions:
            if a_dict:
                for k in a_dict.keys():
                    names.add(k)

    # Order names: Overrides first, then alphabetical
    ordered = []
    seen = set()
    for key in _COLOR_OVERRIDES.keys():
        for n in names:
            if _normalize_name(n) == key and n not in seen:
                ordered.append(n)
                seen.add(n)

    for n in sorted(names, key=lambda s: s.lower()):
        if n not in seen:
            ordered.append(n)
            seen.add(n)

    return ordered

def _build_color_map(df, actions: Optional[List[Dict[str, float]]]) -> Dict[str, str]:
    """Builds the name -> hex color mapping."""
    _, has_grid = _infer_components(df.columns)
    names = _collect_series_names(df, actions, has_grid)
    palette = list(_BASE_PALETTE)
    used_colors = set()
    color_map: Dict[str, str] = {}

    # Apply overrides first
    for n in names:
        key = _normalize_name(n)
        if key in _COLOR_OVERRIDES:
            color = _COLOR_OVERRIDES[key]
            color_map[n] = color
            used_colors.add(color)

    # Apply remaining from palette
    palette_cycle = [c for c in palette if c not in used_colors] or palette
    idx = 0
    for n in names:
        if n in color_map:
            continue
        color_map[n] = palette_cycle[idx % len(palette_cycle)]
        idx += 1

    return color_map


# --------------------------
# Plotters (accept x_axis_data)
# --------------------------

def _plot_generation(ax, df, comps: List[str], has_grid: bool, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots all positive power (generation, battery discharge)."""
    any_plotted = False

    # Plot all components inferred (PV, Wind, Diesel, Bat)
    for cname in comps:
        col = f"{cname}_power"
        if col in df.columns:
            y = np.clip(df[col].values.astype(float), 0, None)
            if np.any(np.nan_to_num(y) > 0):
                ax.plot(x_axis_data, y, label=cname, color=color_map.get(cname, "#000000"))
                any_plotted = True

    ax.set_title("Generation Power (kW) — positive portions only")
    ax.set_ylabel("kW")
    if any_plotted:
        ax.legend(ncol=4, loc='upper left')

def _plot_consumption(ax, df, comps: List[str], has_grid: bool, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots all negative power (loads, battery charge, grid import)."""
    any_plotted = False

    # Plot loads and battery charging
    for cname in comps:
        col = f"{cname}_power"
        if col in df.columns:
            y = df[col].values.astype(float)
            y_neg = np.where(y < 0, y, 0.0) # Only negative parts
            if np.any(np.nan_to_num(y_neg) < 0):
                ax.plot(x_axis_data, y_neg, label=cname, color=color_map.get(cname, "#000000"))
                any_plotted = True

    # Plot grid import (as a negative value)
    if has_grid:
        for c in df.columns:
             if c.startswith("grid") and c.endswith("_power"):
                g = df[c].values.astype(float)
                grid_import_as_neg = -np.clip(g, 0, None) # Clip positive (import) -> make negative
                if np.any(np.nan_to_num(grid_import_as_neg) < 0):
                    name = c.replace("_power", "")
                    ax.plot(x_axis_data, grid_import_as_neg, label=f"{name}_import",
                            color=color_map.get(f"{name}_import", "#000000"))
                    any_plotted = True

    ax.set_title("Consumption Power (kW) — negative portions & grid imports")
    ax.set_ylabel("kW")
    if any_plotted:
        ax.legend(ncol=4, loc='upper left')

def _plot_net_vs_grid(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots net power vs. grid actual power."""
    if "net_power_unbalanced" in df.columns:
        ax.plot(x_axis_data, df["net_power_unbalanced"], label="net_power_unbalanced",
                color=color_map.get("net_power_unbalanced", "#2CA02C"))
    if "grid_slack_kw" in df.columns:
        ax.plot(x_axis_data, df["grid_slack_kw"], label="grid_required (slack)",
                linestyle="--", color=color_map.get("grid_slack_kw", "#7F7F7F"))

    # Plot all grid components' actual power
    for c in df.columns:
        if c.startswith("grid") and c.endswith("_power"):
            name = c.replace("_power", "")
            ax.plot(x_axis_data, df[c], label=f"{name}_actual",
                    color=color_map.get(name, "#000000"))

    ax.set_title("Net vs Grid (kW)")
    ax.set_ylabel("kW")
    ax.legend(ncol=4, loc='upper left')

def _plot_socs(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots SoC for all storage components."""
    plotted = False
    for c in df.columns:
        if c.endswith("_soc"):
            name = c.replace("_soc", "")
            ax.plot(x_axis_data, df[c], label=name, color=color_map.get(name, "#000000"))
            plotted = True
    ax.set_title("State of Charge (SOC)")
    ax.set_ylabel("SOC (0–1)")
    if plotted:
        ax.legend(ncol=4, loc='upper left')

def _plot_costs(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots per-step cash flow by component."""
    plotted = False
    for col in df.columns:
        if col.endswith("_cost") and col != "total_cashflow":
            name = col.replace("_cost", "")
            if np.any(np.nan_to_num(df[col]) != 0):
                ax.plot(x_axis_data, df[col], label=name, color=color_map.get(name, "#000000"))
                plotted = True

    ax.set_title("Per-Step Cash Flow (NEG=expense, POS=revenue)")
    ax.set_ylabel("$ / step")
    if plotted:
        ax.legend(ncol=4, loc='upper left')

def _plot_cumulative_cash(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots the cumulative sum of total cash flow."""
    if "total_cashflow" in df.columns:
        ax.plot(x_axis_data, df["total_cashflow"].cumsum(),
                color=color_map.get("total_cashflow", "#9467BD"))
    ax.set_title("Cumulative Total Cash Flow")
    ax.set_ylabel("$")

def _plot_unmet_curtailed(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots unmet load, curtailed generation, and downtime."""
    plotted = False
    if "unmet_load_kw" in df.columns and np.any(np.nan_to_num(df["unmet_load_kw"]) > 0):
        ax.plot(x_axis_data, df["unmet_load_kw"], label="Unmet Load (kW)",
                color=color_map.get("unmet_load_kw", "#D62728"))
        plotted = True
    if "curtailed_gen_kw" in df.columns and np.any(np.nan_to_num(df["curtailed_gen_kw"]) > 0):
        ax.plot(x_axis_data, df["curtailed_gen_kw"], label="Curtailed Gen (kW)",
                color=color_map.get("curtailed_gen_kw", "#7F7F7F"))
        plotted = True
    if "downtime" in df.columns and np.any(np.nan_to_num(df["downtime"]) > 0):
        ax.step(x_axis_data, df["downtime"], where="post", label="Downtime (0/1)",
                color=color_map.get("downtime", "#000000"), alpha=0.7)
        plotted = True
    ax.set_title("Security of Supply")
    ax.set_ylabel("kW (and 0/1)")
    if plotted:
        ax.legend(ncol=3, loc='upper left')

def _plot_actions(ax, actions: List[Dict[str, float]], color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots the actions taken by the controller."""
    keys = set()
    for a in actions:
        if a:
            for k in a.keys():
                keys.add(k)
    keys = sorted(keys, key=lambda s: s.lower())

    plotted = False
    for k in keys:
        series = []
        for a in actions:
            v = a.get(k, 0.0) if a else 0.0
            if isinstance(v, dict):
                v = v.get("power_setpoint", 0.0)
            try:
                series.append(float(v))
            except (ValueError, TypeError):
                 series.append(np.nan) # Handle 'connect'/'disconnect'

        if not all(np.isnan(series)):
            # Use step plot for setpoints
            ax.step(x_axis_data, series, where='post', label=k, color=color_map.get(k, "#000000"))
            plotted = True

    ax.set_title("Actions by Component (Numeric Setpoints)")
    ax.set_ylabel("Setpoint")
    if plotted:
        ax.legend(ncol=4, loc='upper left')


# --------------------------
# Main entry: plot + save
# --------------------------
def plot_simulation(
    df,
    actions: Optional[List[Dict[str, float]]] = None,
    sim_dt_minutes: int = 60,
    sim_name: Optional[str] = None,
    base_dir: str = "plots",
    save: bool = False,
    formats: Iterable[str] = ("png", "pdf"),
    style: Optional[str] = "conference",
):
    """
    Create a multi-panel dashboard and (optionally) save combined + individual figures.

    Args:
        df (pd.DataFrame): The results dataframe from MicrogridEnv.get_results().
        actions (list, optional): The list of action dicts used.
        sim_dt_minutes (int): The duration of a single step (e.g., 1 or 60).
        sim_name (str, optional): Name for saving the plot, e.g., "scen_1".
        base_dir (str): Root directory to save plots.
        save (bool): Whether to save the plots to disk.
        formats (iterable): Formats to save (e.g., ["png", "pdf"]).
        style (str, optional): Matplotlib style to apply ("conference").

    Returns:
        dict: {"fig": fig, "axes": axes_list, "out_dir": path or None}
    """
    try:
        import pandas as pd
        if not hasattr(df, "columns"):
            df = pd.DataFrame(df)
    except Exception:
        pass # Allow non-pandas dicts

    if style == "conference":
        _apply_conference_style()

    # --- Time Axis Calculation ---
    n_steps = len(df)
    total_hours = (n_steps * sim_dt_minutes) / 60.0

    # ALWAYS calculate x_axis_data in HOURS
    x_axis_data = np.arange(n_steps) * (sim_dt_minutes / 60.0)

    if total_hours <= 48:
        x_label = f"Time (Hours)"
    else:
        x_label = f"Time (Days)"
    # ---

    comps, has_grid = _infer_components(df.columns)

    # We need the action list to be the same length as the dataframe
    if actions is not None:
        if len(actions) > n_steps:
            actions = actions[:n_steps]
        elif len(actions) < n_steps:
            actions = actions + [{}] * (n_steps - len(actions))

    color_map = _build_color_map(df, actions)

    nrows = 7 + (1 if actions is not None else 0)
    fig, axs = plt.subplots(nrows, 1, figsize=(14, 3*nrows), sharex=True)
    ax_idx = 0

    # --- Call plotters with new x_axis_data ---
    _plot_generation(axs[ax_idx], df, comps, has_grid, color_map, x_axis_data); ax_idx += 1
    _plot_consumption(axs[ax_idx], df, comps, has_grid, color_map, x_axis_data); ax_idx += 1
    _plot_net_vs_grid(axs[ax_idx], df, color_map, x_axis_data); ax_idx += 1
    _plot_socs(axs[ax_idx], df, color_map, x_axis_data); ax_idx += 1
    _plot_costs(axs[ax_idx], df, color_map, x_axis_data); ax_idx += 1
    _plot_cumulative_cash(axs[ax_idx], df, color_map, x_axis_data); ax_idx += 1
    _plot_unmet_curtailed(axs[ax_idx], df, color_map, x_axis_data); ax_idx += 1
    if actions is not None:
        _plot_actions(axs[ax_idx], actions, color_map, x_axis_data); ax_idx += 1

    # --- Apply new axis formatting ---
    _set_xaxis_time(axs, x_axis_data, x_label, total_hours)

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

        # --- Fix _save_single function ---
        def _save_single(plot_fn_lambda, filename_root: str):
            f, ax = plt.subplots(1, 1, figsize=(10, 4))
            plot_fn_lambda(ax) # Lambda now bakes in all args
            _set_xaxis_time([ax], x_axis_data, x_label, total_hours) # Use new axis
            f.tight_layout()
            for fmt in formats:
                f.savefig(os.path.join(indiv_dir, f"{filename_root}.{fmt}"))
            plt.close(f)

        # Update lambdas to pass x_axis_data
        _save_single(lambda ax: _plot_generation(ax, df, comps, has_grid, color_map, x_axis_data), "generation")
        _save_single(lambda ax: _plot_consumption(ax, df, comps, has_grid, color_map, x_axis_data), "consumption")
        _save_single(lambda ax: _plot_net_vs_grid(ax, df, color_map, x_axis_data), "net_vs_grid")
        _save_single(lambda ax: _plot_socs(ax, df, color_map, x_axis_data), "soc")
        _save_single(lambda ax: _plot_costs(ax, df, color_map, x_axis_data), "costs")
        _save_single(lambda ax: _plot_cumulative_cash(ax, df, color_map, x_axis_data), "cumulative_cash")
        _save_single(lambda ax: _plot_unmet_curtailed(ax, df, color_map, x_axis_data), "security")
        if actions is not None:
            _save_single(lambda ax: _plot_actions(ax, actions, color_map, x_axis_data), "actions")

    return {"fig": fig, "axes": list(axs), "out_dir": out_dir}
