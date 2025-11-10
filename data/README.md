# Data Folder

Place your time-series CSVs or parquet files here, or implement a loader function that reads from your own paths.

## Example Streams

- `residential_load` (kW)
- `factory_load` (kW)
- `pv_irradiance` (W/m^2) or `pv_power` (kW)
- `wind_speed` (m/s) or `wind_power` (kW)
- `hydro_flow` (m^3/s) or `hydro_power` (kW)
- `price_import` ($/kWh), optional `price_export` ($/kWh)

## Loader Protocol

Create a callable with signature:

```python
def loader(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> dict[str, pd.DataFrame]:
    ...
```
