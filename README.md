# Microgrid Simulation and Control Framework

A Python-based simulation framework for modeling and controlling microgrids, built with an OOP design:

- Flexible configurations of generators, storage, loads
- Clear sign conventions (power and cash flow)
- Ready for EMS, MPC, MDP/POMDP, and RL (Gym/Gymnasium)

**Experiments live outside the codebase** in notebooks, per your requirement.

## Structure

- `microgrid_sim/`: core Python package
- `notebooks/`: Jupyter notebooks for testing and experiments
- `data/`: placeholder for user-provided time-series data (CSV/Parquet)

## Conventions

- **Power**: generation > 0 kW, consumption < 0 kW  
- **Cash flow** (per time step): **negative = expense**, **positive = revenue**

## References

We follow control-oriented modeling and EMS structure from:  
Bordons, C.; García-Torres, F.; Ridao, M. (2020). *Model Predictive Control of Microgrids*. Springer.  
Concrete components will include specific chapter/equation references in their docstrings.
