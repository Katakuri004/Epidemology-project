import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path


JHU_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data"
CONFIRMED_URL = f"{JHU_BASE}/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
DEATHS_URL = f"{JHU_BASE}/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"


def _load_timeseries(url: str, admin_level: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # JHU columns: Province/State, Country/Region, Lat, Long, <date columns...>
    if admin_level == "country":
        geo_cols = ["Country/Region"]
    elif admin_level == "admin1":
        # Keep Province/State; fall back to country if missing
        df["Province/State"] = df["Province/State"].fillna(df["Country/Region"])
        geo_cols = ["Country/Region", "Province/State", "Lat", "Long"]
    else:
        raise ValueError("admin_level must be 'country' or 'admin1'")
    date_cols = df.columns[4:]
    grouped = df.groupby(geo_cols)[date_cols].sum().reset_index()
    # Ensure date parsing and sorting
    melted = grouped.melt(id_vars=geo_cols, var_name="date", value_name="value")
    melted["date"] = pd.to_datetime(melted["date"], format="%m/%d/%y")
    sort_cols = [c for c in geo_cols if c in melted.columns]
    sort_cols.append("date")
    melted = melted.sort_values(sort_cols)  # ascending
    return melted


def _pivot_to_matrix(cases: pd.DataFrame, deaths: pd.DataFrame, admin_level: str) -> tuple[pd.DataFrame, np.ndarray]:
    # Align entities and dates
    if admin_level == "country":
        key_cols = ["Country/Region"]
    else:
        key_cols = ["Country/Region", "Province/State", "Lat", "Long"]
    entities = cases[key_cols].drop_duplicates()
    entities = entities.merge(deaths[key_cols].drop_duplicates(), on=key_cols, how="inner")
    common_dates = sorted(set(cases["date"]) & set(deaths["date"]))

    # Build matrices cumulative
    def to_matrix(df: pd.DataFrame) -> np.ndarray:
        sub = df[df["date"].isin(common_dates)]
        piv = sub.pivot_table(index="date", columns=key_cols, values="value", aggfunc="sum").reindex(index=common_dates)
        # Ensure columns align with entities order
        piv = piv.reindex(columns=pd.MultiIndex.from_frame(entities[key_cols]))
        return piv.to_numpy(dtype=np.float32)  # (T, N)

    cum_cases = to_matrix(cases)
    cum_deaths = to_matrix(deaths)

    # Compute daily new as diff with clip to non-negative
    new_cases = np.diff(cum_cases, axis=0, prepend=cum_cases[:1])
    new_deaths = np.diff(cum_deaths, axis=0, prepend=cum_deaths[:1])
    # Replace impossible negatives and NaNs
    new_cases = np.nan_to_num(new_cases, nan=0.0)
    new_deaths = np.nan_to_num(new_deaths, nan=0.0)
    new_cases = np.clip(new_cases, a_min=0.0, a_max=None)
    new_deaths = np.clip(new_deaths, a_min=0.0, a_max=None)

    # Features: [new_cases, new_deaths, cum_cases]
    T, N = new_cases.shape
    series = np.stack([new_cases, new_deaths, cum_cases], axis=-1)  # (T, N, F=3)

    if admin_level == "country":
        nodes = pd.DataFrame({"node_id": range(entities.shape[0]), "name": entities["Country/Region"].to_list()})
    else:
        nodes = entities.reset_index(drop=True).reset_index(names="node_id")
        nodes.rename(columns={"index": "node_id", "Country/Region": "country", "Province/State": "name", "Lat": "lat", "Long": "lon"}, inplace=True)
        nodes = nodes[["node_id", "country", "name", "lat", "lon"]]
    return nodes, series.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JHU CSSE COVID-19 time series and build processed dataset")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--admin", type=str, default="country", choices=["country", "admin1"], help="Geographic level")
    args = parser.parse_args()

    cfg = load_config(args.config)
    proc_dir = resolve_path(cfg.dataset.get("processed_dir", "data/processed"))
    os.makedirs(proc_dir, exist_ok=True)

    # Load JHU time series
    cases_df = _load_timeseries(CONFIRMED_URL, args.admin)
    deaths_df = _load_timeseries(DEATHS_URL, args.admin)

    nodes_df, series = _pivot_to_matrix(cases_df, deaths_df, args.admin)

    # Persist nodes and series
    nodes_path = Path(proc_dir) / "nodes.csv"
    series_path = Path(proc_dir) / "series.npy"
    nodes_df.to_csv(nodes_path, index=False)
    np.save(series_path, series)
    print({"nodes": int(series.shape[1]), "timesteps": int(series.shape[0]), "features": int(series.shape[2]), "path": str(series_path), "admin": args.admin})


if __name__ == "__main__":
    main()


