"""
Downloads real NOAA OISST (Optimum Interpolation Sea Surface Temperature)
data from the public ERDDAP server — no API key or account required.
"""

import argparse
import os
import requests
import xarray as xr

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"


def build_query_url(lat_min, lat_max, lon_min, lon_max, date_start, date_end):
    url = (
        f"{ERDDAP_BASE}.nc?"
        f"sst[({date_start}):1:({date_end})][(0.0):1:(0.0)]"
        f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    )
    return url


def download(lat_min, lat_max, lon_min, lon_max, date_start, date_end, out_path):
    url = build_query_url(lat_min, lat_max, lon_min, lon_max, date_start, date_end)
    print(f"Requesting data from ERDDAP:\n{url}\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Download the actual .nc bytes via plain HTTP GET first — this avoids
    # netCDF4's OPeNDAP client, which mishandles ERDDAP's query-string syntax
    # and throws "Malformed or unexpected Constraint" when opened directly
    # as a URL with xr.open_dataset().
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Failed to download data from NOAA ERDDAP.\n"
            f"This can happen if the server is temporarily unavailable, "
            f"the date range/bounding box has no data, or there's no "
            f"internet access from this environment.\n"
            f"Original error: {e}"
        )

    with open(out_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded {len(response.content) / 1e6:.2f} MB to {out_path}")

    # Now open the local file to verify it's valid and print info
    try:
        ds = xr.open_dataset(out_path)
    except Exception as e:
        # If this fails, the downloaded content was likely an error page,
        # not real NetCDF data
        with open(out_path, "r", errors="ignore") as f:
            preview = f.read(500)
        raise RuntimeError(
            f"Downloaded file is not valid NetCDF. ERDDAP likely returned "
            f"an error page instead of data. First 500 chars of response:\n"
            f"{preview}\n\nOriginal error: {e}"
        )

    if "zlev" in ds.dims:
        ds = ds.squeeze("zlev", drop=True)
        ds.to_netcdf(out_path)  # re-save without the singleton dim

    print(f"Verified valid NetCDF file.")
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