"""
Downloads real NOAA OISST (Optimum Interpolation Sea Surface Temperature)
data from the public ERDDAP server — no API key or account required.

This replaces the synthetic dataset with real satellite-derived SST data,
so the physics-informed loss has genuine turbulent/discontinuous patterns
to be tested against (synthetic Gaussian blobs are too smooth to be a
real test of the physics constraint).

Source: NOAA Coral Reef Watch / OISST v2.1 daily, via ERDDAP.
"""

import argparse
import os
import xarray as xr

# Public ERDDAP endpoint for NOAA OISST v2.1 daily SST (no auth required)
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"


def build_query_url(lat_min, lat_max, lon_min, lon_max, date_start, date_end):
    """
    Builds an ERDDAP OPeNDAP query URL for a bounding box + date range.
    Variable: 'sst' (sea surface temperature, degrees C)
    """
    url = (
        f"{ERDDAP_BASE}.nc?"
        f"sst[({date_start}):1:({date_end})][(0.0):1:(0.0)]"
        f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    )
    return url


def download(lat_min, lat_max, lon_min, lon_max, date_start, date_end, out_path):
    url = build_query_url(lat_min, lat_max, lon_min, lon_max, date_start, date_end)
    print(f"Requesting data from ERDDAP:\n{url}\n")

    try:
        ds = xr.open_dataset(url)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch data from NOAA ERDDAP. This can happen if the "
            f"server is temporarily unavailable, the date range has no data yet, "
            f"or there's no internet access from this environment.\n"
            f"Original error: {e}"
        )

    # Drop the singleton 'zlev' depth dimension if present
    if "zlev" in ds.dims:
        ds = ds.squeeze("zlev", drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path)

    print(f"Saved real NOAA SST data to {out_path}")
    print(f"Shape: {ds['sst'].shape}  (time, lat, lon)")
    print(f"Date range: {ds.time.values[0]} to {ds.time.values[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Download real NOAA OISST SST data")
    parser.add_argument("--lat_min", type=float, default=10.0)
    parser.add_argument("--lat_max", type=float, default=30.0)
    parser.add_argument("--lon_min", type=float, default=-90.0)
    parser.add_argument("--lon_max", type=float, default=-70.0)
    parser.add_argument("--date_start", type=str, default="2023-01-01")
    parser.add_argument("--date_end", type=str, default="2023-06-30")
    parser.add_argument("--out", type=str, default="./data/raw/noaa_sst_real.nc")
    args = parser.parse_args()

    download(
        args.lat_min, args.lat_max, args.lon_min, args.lon_max,
        args.date_start, args.date_end, args.out
    )


if __name__ == "__main__":
    main()