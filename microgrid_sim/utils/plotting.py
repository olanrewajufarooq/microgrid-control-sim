"""
microgrid_sim/utils/plotting.py

Standard plotting utilities with dynamic support for multiple components of the same type.
FIXED: Restores missing helper functions and implements instance-specific coloring/styles.
"""
from __future__ import annotations
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --------------------------
# Fixed, high-contrast palette and overrides
# --------------------------
_BASE_PALETTE = [
    "#000000", "#2CA02C", "#D62728", "#FFBF00", "#1F77B4", "#E377C2",
    "#17BECF", "#8C564B", "#9467BD", "#7F7F7F",
]

# Overrides define the base color for component groups and KPIs
_COLOR_OVERRIDES = {
    "grid": "#000000", "grid_import": "#000000", "grid_slack_kw": "#7F7F7F",
    "pv": "#FFBF00", "wind": "#1F77B4", "diesel": "#D62728",
    "bat": "#E377C2", "factory": "#8C564B", "house": "#2CA02C",
    "site_load": "#8C564B", "total_cashflow": "#9467BD",
    "unmet_load_kw": "#D62728", "curtailed_gen_kw": "#7F7F7F",
    "downtime": "#000000", "net_power_unbalanced": "#1F77B4",
}

# --- Internal Helper Functions (Restored and Defined) ---
def _ensure_dir(path: str):
    """Ensures a directory exists."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _apply_conference_style():
    """Applies a clean style for plots."""
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.grid": True,
        "grid.linestyle": ":", "grid.alpha": 0.6, "lines.linewidth": 2.0,
        "figure.autolayout": False,
    })

def _set_xaxis_time(axs: List[plt.Axes], x_axis_data: np.ndarray, x_label: str, total_hours: float):
    """Helper to set the new time-based x-axis."""
    if not x_axis_data.any(): return
    for ax in axs: ax.set_xlim(x_axis_data[0], x_axis_data[-1])
    bottom_ax = axs[-1]
    if total_hours <= 48:
        tick_step = max(1, int(total_hours / 12)) * 2
        ticks = np.arange(0, total_hours + 1, tick_step)
        bottom_ax.set_xticks(ticks)
        bottom_ax.set_xticklabels([f"{t:.0f}h" for t in ticks])
        bottom_ax.set_xlabel("Time (Hours)")
    else:
        bottom_ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        def day_formatter(hour, pos): return f"{int(hour / 24) + 1}"
        bottom_ax.xaxis.set_major_formatter(ticker.FuncFormatter(day_formatter))
        bottom_ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
        def hour_formatter(hour, pos):
            hour_of_day = hour % 24
            if hour_of_day == 0: return ""
            return f"{int(hour_of_day)}h"
        bottom_ax.xaxis.set_minor_formatter(ticker.FuncFormatter(hour_formatter))
        bottom_ax.tick_params(axis='x', which='major', labelsize=10, length=6, width=1.2)
        bottom_ax.tick_params(axis='x', which='minor', labelsize=8, labelcolor='#555', length=3)
        bottom_ax.set_xlabel(x_label)
# --------------------------------------------------------------------------

def _get_base_name(name: str) -> str:
    """Extracts the base component type name (e.g., 'diesel_1' -> 'diesel')."""
    name = name.lower()
    if 'utility_grid' in name.lower(): return 'grid'
    for base in ["pv", "wind", "diesel", "bat", "house", "factory", "grid", "site_load"]:
        if name.startswith(base): return base
    return name.split('_')[0]

def _find_components(df_cols: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Identifies all unique component names from the DataFrame columns, separating them
    into Gens, Storage, Loads, and Grids.
    """
    unique_names = set()
    for col in df_cols:
        if col.endswith(("_power", "_cost", "_soc", "_downtime")):
            name = col.rsplit('_', 1)[0]
            if name.endswith(("power", "cost", "downtime")): continue
            unique_names.add(name)

    gens, storage, loads, grids = [], [], [], []
    for name in sorted(list(unique_names)):
        base_type = _get_base_name(name)

        if base_type in ("pv", "wind", "diesel"): gens.append(name)
        elif base_type == "bat": storage.append(name)
        elif base_type in ("house", "factory", "site_load"): loads.append(name)
        elif base_type == "grid": grids.append(name)

    gens = [g for g in gens if _get_base_name(g) != "grid"]
    return gens, storage, loads, grids

def _collect_series_names(df, actions: Optional[List[Dict[str, float]]]) -> List[str]:
    """Collects all unique series names for consistent color mapping."""
    names = set()
    gens, storage, loads, grids = _find_components(df.columns)

    for group in [gens, storage, loads, grids]:
        for name in group: names.add(name)

    for special in _COLOR_OVERRIDES.keys():
        if special in df.columns: names.add(special)

    if grids and any(f"{g}_power" in df.columns for g in grids):
        names.add(f"{grids[0]}_import")

    if actions is not None:
        for a_dict in actions:
            if a_dict:
                for k in a_dict.keys(): names.add(k)

    # Sort names by base type then instance name
    ordered = []
    base_priority = {base: i for i, base in enumerate(_COLOR_OVERRIDES.keys())}
    grouped_names: Dict[str, List[str]] = {}
    for n in names:
        base = _get_base_name(n)
        if base not in grouped_names: grouped_names[base] = []
        grouped_names[base].append(n)

    sorted_bases = sorted(grouped_names.keys(), key=lambda b: base_priority.get(b, 99))

    for base in sorted_bases:
        for name in sorted(grouped_names[base]):
            if name not in ordered: ordered.append(name)

    return ordered

def _build_color_map(df, actions: Optional[List[Dict[str, float]]]) -> Dict[str, str]:
    """
    Builds the name -> hex color mapping, ensuring individual instances get the base color.
    The differentiation is handled by line styles in the plotters.
    """
    names = _collect_series_names(df, actions)
    color_map: Dict[str, str] = {}

    for n in names:
        base_type = _get_base_name(n)

        if f"{_get_base_name(n)}_import" in n: color = _COLOR_OVERRIDES.get("grid_import")
        elif n in _COLOR_OVERRIDES: color = _COLOR_OVERRIDES.get(n)
        else: color = _COLOR_OVERRIDES.get(base_type)

        if color: color_map[n] = color

    return color_map

# --- Plotters (Fixed to use cname for color lookup, relying on fixed _build_color_map) ---
def _plot_generation(ax, df, gen_comps: List[str], stor_comps: List[str],
                     color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots all positive power (generation, battery discharge)."""
    any_plotted = False
    style_cycle = ['-', '--', ':', '-.']

    for i, cname in enumerate(sorted(gen_comps + stor_comps)):
        col = f"{cname}_power"
        if col in df.columns:
            y = np.clip(df[col].values.astype(float), 0, None)
            if np.any(np.nan_to_num(y) > 0):
                ax.plot(x_axis_data, y, label=cname,
                        color=color_map.get(cname, "#000000"),
                        linestyle=style_cycle[i % len(style_cycle)], linewidth=2)
                any_plotted = True
    ax.set_title("Generation Power (kW) - positive portions only")
    ax.set_ylabel("kW")
    if any_plotted: ax.legend(ncol=4, loc='best')

def _plot_consumption(ax, df, load_comps: List[str], stor_comps: List[str],
                      color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots all negative power (loads, battery charge)."""
    any_plotted = False
    style_cycle = ['-', '--', ':', '-.']

    # Plot Loads and Storage Consumption
    for i, cname in enumerate(sorted(load_comps + stor_comps)):
        col = f"{cname}_power"
        if col in df.columns:
            y = df[col].values.astype(float)
            y_neg = np.where(y < 0, y, 0.0)
            if np.any(np.nan_to_num(y_neg) < 0):
                ax.plot(x_axis_data, y_neg, label=cname,
                        color=color_map.get(cname, "#000000"),
                        linestyle=style_cycle[i % len(style_cycle)], linewidth=2)
                any_plotted = True

    ax.set_title("Consumption Power (kW) - negative portions")
    ax.set_ylabel("kW")
    if any_plotted: ax.legend(ncol=4, loc='best')

def _plot_net_vs_grid(ax, df, grid_comps: List[str], color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots net power vs. grid actual power."""
    if "net_power_unbalanced" in df.columns:
        ax.plot(x_axis_data, df["net_power_unbalanced"], label="net_power_unbalanced",
                 color=color_map.get("net_power_unbalanced", "#1F77B4"), linewidth=2)
    if "grid_slack_kw" in df.columns:
        ax.plot(x_axis_data, df["grid_slack_kw"], label="grid_required (slack)",
                 linestyle="--", color=color_map.get("grid_slack_kw", "#7F7F7F"), linewidth=2)

    for cname in grid_comps:
        col = f"{cname}_power"
        if col in df.columns:
            ax.plot(x_axis_data, df[col], label=f"{cname}_actual",
                     color=color_map.get(cname, "#000000"), linestyle='-', linewidth=2)

    ax.set_title("Net vs Grid (kW)")
    ax.set_ylabel("kW")
    ax.legend(ncol=4, loc='best')

def _plot_socs(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots State of Charge for all storage units."""
    plotted = False
    style_cycle = ['-', '--', ':', '-.']

    for i, c in enumerate(sorted(df.columns)):
        if c.endswith("_soc"):
            name = c.replace("_soc", "")
            ax.plot(x_axis_data, df[c], label=name,
                    color=color_map.get(name, "#000000"),
                    linestyle=style_cycle[i % len(style_cycle)], linewidth=2)
            plotted = True
    ax.set_title("State of Charge (SOC)")
    ax.set_ylabel("SOC (0-1)")
    if plotted: ax.legend(ncol=4, loc='best')

def _plot_costs(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots per-step cash flow for components (non-accumulated)."""
    plotted = False
    style_cycle = ['-', '--', ':', '-.']

    for i, col in enumerate(sorted(df.columns)):
        if col.endswith("_cost") and col != "total_cashflow":
            name = col.replace("_cost", "")
            if np.any(np.nan_to_num(df[col]) != 0):
                ax.plot(x_axis_data, df[col], label=name,
                        color=color_map.get(name, "#000000"),
                        linestyle=style_cycle[i % len(style_cycle)], linewidth=1.5)
                plotted = True
    ax.set_title("Per-Step Cash Flow (NEG=expense, POS=revenue)")
    ax.set_ylabel("$ / step")
    if plotted:
        ax.legend(ncol=4, loc='best')

def _plot_cumulative_cash(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots the cumulative total cash flow."""
    if "total_cashflow" in df.columns:
        ax.plot(x_axis_data, df["total_cashflow"].cumsum(),
                 color=color_map.get("total_cashflow", "#9467BD"))
    ax.set_title("Cumulative Total Cash Flow")
    ax.set_ylabel("$")

def _plot_unmet_curtailed(ax, df, color_map: Dict[str, str], x_axis_data: np.ndarray):
    """
    Plots system reliability metrics with a secondary y-axis for downtime (0/1).
    """
    plotted_left = False
    plotted_right = False
    y_series_left = []

    # --- Left axis: unmet & curtailed energy (kW) ---
    if "unmet_load_kw" in df.columns:
        y = np.nan_to_num(df["unmet_load_kw"].values, nan=0.0)
        ax.plot(
            x_axis_data, y,
            label="Unmet Load (kW)",
            color=color_map.get("unmet_load_kw", "#D62728"),
            linewidth=2,
        )
        y_series_left.append(y)
        plotted_left = True

    if "curtailed_gen_kw" in df.columns:
        y = np.nan_to_num(df["curtailed_gen_kw"].values, nan=0.0)
        ax.plot(
            x_axis_data, y,
            label="Curtailed Gen (kW)",
            color=color_map.get("curtailed_gen_kw", "#7F7F7F"),
            linewidth=2,
        )
        y_series_left.append(y)
        plotted_left = True

    ax.set_ylabel("kW")
    ax.set_title("Security of Supply")

    # --- Right axis: downtime (0/1) ---
    if "downtime" in df.columns:
        ax2 = ax.twinx()
        y = np.nan_to_num(df["downtime"].values, nan=0.0)
        ax2.step(
            x_axis_data, y, where="post",
            label="Downtime (0/1)",
            color=color_map.get("downtime", "#000000"),
            alpha=0.8, linewidth=1.8,
        )
        ax2.set_ylabel("Downtime (0/1)")
        ax2.set_ylim(-0.1, 1.1)
        plotted_right = True
    else:
        ax2 = None

    # --- Legends ---
    if plotted_left and plotted_right:
        # combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, ncol=3, loc="best")
    elif plotted_left:
        ax.legend(ncol=3, loc="best")
    elif plotted_right:
        ax2.legend(ncol=3, loc="best")

    # --- Safety y-limits if everything is zero ---
    if y_series_left:
        max_left = float(np.max([np.max(np.abs(y)) for y in y_series_left]))
        if max_left == 0.0:
            ax.set_ylim(-0.1, 1.0)


def _plot_component_downtime(ax, df, components: List[str], color_map: Dict[str, str],
                             x_axis_data: np.ndarray):
    """
    Plots per-component downtime (0/1) using downtime columns produced by the simulation.
    Expects columns named "<component>_downtime". If absent, the component is skipped.
    """
    plotted = False
    style_cycle = ['-', '--', ':', '-.']
    for i, cname in enumerate(sorted(components)):
        col = f"{cname}_downtime"
        if col in df.columns:
            vals = df[col].values.astype(float)
            ax.step(x_axis_data, vals, where="post", label=f"{cname} downtime",
                    color=color_map.get(cname, "#000000"), linestyle=style_cycle[i % len(style_cycle)],
                    linewidth=1.8)
            plotted = True
    ax.set_title("Component Downtime (0=on, 1=down)")
    ax.set_ylabel("Downtime (0/1)")
    ax.set_ylim(-0.1, 1.1)
    if plotted:
        ax.legend(ncol=4, loc="best")

def _plot_actions(ax, actions: List[Dict[str, Union[float, Dict[str, float], str]]], color_map: Dict[str, str], x_axis_data: np.ndarray):
    """Plots the numeric setpoints and binary states from the action dictionary."""
    keys = set()
    for a in actions:
        if a:
            for k, v in a.items():
                if isinstance(v, (float, int, np.floating, np.integer)) or \
                   (isinstance(v, dict) and "power_setpoint" in v) or \
                   isinstance(v, str):
                    keys.add(k)

    keys = sorted(keys, key=lambda s: _get_base_name(s).lower() + s.lower())
    plotted = False
    style_cycle = ['-', '--', ':', '-.']

    for i, k in enumerate(keys):
        series = []
        is_discrete_state = False

        for a in actions:
            v = a.get(k, 0.0)

            if isinstance(v, dict): # Battery/Diesel dictionary action
                v = v.get("power_setpoint", 0.0)

            elif isinstance(v, str): # Grid/Renewable string action ("connect", "disconnect")
                is_discrete_state = True
                if v.lower() in ("disconnect", "off"):
                    series.append(0.0)
                elif v.lower() in ("connect", "on"):
                    series.append(1.0)
                else:
                    series.append(np.nan)
                continue

            try:
                series.append(float(v))
            except (ValueError, TypeError):
                series.append(np.nan)

        if not all(np.isnan(series)):
            if is_discrete_state:
                 ax.step(x_axis_data, series, where='post', label=k, color=color_map.get(k, "#000000"), linestyle=style_cycle[i % len(style_cycle)], linewidth=1.5)
            else:
                 ax.step(x_axis_data, series, where='post', label=k, color=color_map.get(k, "#000000"), linestyle=style_cycle[i % len(style_cycle)])

            plotted = True

    ax.set_title("Actions by Component (Setpoints and Binary States)")
    ax.set_ylabel("Power / State")
    if plotted: ax.legend(ncol=4, loc='best')


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
    """
    try:
        import pandas as pd
        if not hasattr(df, "columns"): df = pd.DataFrame(df)
    except Exception:
        return {"fig": None, "axes": [], "out_dir": None}

    if style == "conference": _apply_conference_style()

    n_steps = len(df)
    total_hours = (n_steps * sim_dt_minutes) / 60.0
    x_axis_data = np.arange(n_steps) * (sim_dt_minutes / 60.0)
    x_label = "Time (Days)" if total_hours > 48 else "Time (Hours)"

    gens, storage, loads, grids = _find_components(df.columns)

    if actions is not None:
        if len(actions) > n_steps: actions = actions[:n_steps]
        elif len(actions) < n_steps: actions = actions + [{}] * (n_steps - len(actions))

    color_map = _build_color_map(df, actions)

    nrows = 8 + (1 if actions is not None else 0)
    fig, axs = plt.subplots(nrows, 1, figsize=(14, 3*nrows), sharex=True)
    ax_idx = 0

    _plot_generation(axs[ax_idx], df, gens, storage, color_map, x_axis_data)
    ax_idx += 1

    _plot_consumption(axs[ax_idx], df, loads, storage, color_map, x_axis_data)
    ax_idx += 1

    _plot_net_vs_grid(axs[ax_idx], df, grids, color_map, x_axis_data)
    ax_idx += 1

    _plot_socs(axs[ax_idx], df, color_map, x_axis_data)
    ax_idx += 1

    _plot_costs(axs[ax_idx], df, color_map, x_axis_data)
    ax_idx += 1

    _plot_cumulative_cash(axs[ax_idx], df, color_map, x_axis_data)
    ax_idx += 1

    _plot_unmet_curtailed(axs[ax_idx], df, color_map, x_axis_data)
    ax_idx += 1

    downtime_targets = gens + storage + grids
    _plot_component_downtime(axs[ax_idx], df, downtime_targets, color_map, x_axis_data)
    ax_idx += 1

    if actions is not None:
        _plot_actions(axs[ax_idx], actions, color_map, x_axis_data)
        ax_idx += 1

    _set_xaxis_time(axs, x_axis_data, x_label, total_hours)

    fig.suptitle("Microgrid Simulation - Overview", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_dir = None
    if save and sim_name:
        out_dir = os.path.join(base_dir, sim_name)
        indiv_dir = os.path.join(out_dir, "individuals")
        _ensure_dir(indiv_dir)

        for fmt in formats: fig.savefig(os.path.join(out_dir, f"overview.{fmt}"))

        def _save_single(plot_fn_lambda, filename_root: str):
            f, ax = plt.subplots(1, 1, figsize=(10, 4))
            plot_fn_lambda(ax)
            _set_xaxis_time([ax], x_axis_data, x_label, total_hours)
            f.tight_layout()
            for fmt in formats: f.savefig(os.path.join(indiv_dir, f"{filename_root}.{fmt}"))
            plt.close(f)

        _save_single(lambda ax: _plot_generation(ax, df, gens, storage, color_map, x_axis_data), "generation")
        _save_single(lambda ax: _plot_consumption(ax, df, loads, storage, color_map, x_axis_data), "consumption")
        _save_single(lambda ax: _plot_net_vs_grid(ax, df, grids, color_map, x_axis_data), "net_vs_grid")
        _save_single(lambda ax: _plot_socs(ax, df, color_map, x_axis_data), "soc")
        _save_single(lambda ax: _plot_costs(ax, df, color_map, x_axis_data), "costs")
        _save_single(lambda ax: _plot_cumulative_cash(ax, df, color_map, x_axis_data), "cumulative_cash")
        _save_single(lambda ax: _plot_unmet_curtailed(ax, df, color_map, x_axis_data), "security")
        _save_single(lambda ax: _plot_component_downtime(ax, df, downtime_targets, color_map, x_axis_data), "component_downtime")
        if actions is not None:
            _save_single(lambda ax: _plot_actions(ax, actions, color_map, x_axis_data), "actions")

    return {"fig": fig, "axes": list(axs), "out_dir": out_dir}

# === Reward progression helper (SB3 Monitor CSV) ===
def plot_reward_progress(
    monitor_csv_path: str,
    title: str = "Training Reward",
    out_path: Optional[str] = None,
    rolling: int = 10,
):
    """
    Plot episodic rewards from a Stable-Baselines3 Monitor CSV.
    - monitor_csv_path: path given to Monitor(..., filename=...)
    - rolling: window for rolling mean overlay
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if not os.path.isfile(monitor_csv_path):
        print(f"[plot_reward_progress] File not found: {monitor_csv_path}")
        return None

    # SB3 Monitor CSV starts with comment lines beginning '#'
    with open(monitor_csv_path, "r") as f:
        lines = [ln for ln in f.readlines() if not ln.startswith("#")]
    if not lines:
        print("[plot_reward_progress] No data rows in monitor CSV yet.")
        return None

    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines)))
    # expected columns: r (episode reward), l (length), t (time seconds)
    if "r" not in df.columns:
        print("[plot_reward_progress] Column 'r' not found.")
        return None

    rewards = df["r"].values
    episodes = np.arange(1, len(rewards) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, rewards, label="Episode reward")
    if rolling and len(rewards) >= rolling:
        roll = pd.Series(rewards).rolling(rolling, min_periods=1).mean().values
        ax.plot(episodes, roll, linestyle="--", label=f"Rolling mean ({rolling})")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
    return fig, ax

