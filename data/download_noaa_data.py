"""
Downloads real NOAA OISST (Optimum Interpolation Sea Surface Temperature)
data from a public ERDDAP server — no API key or account required.

Includes retry logic and a fallback mirror, since ERDDAP servers
occasionally return transient 502/503 errors under load.
"""

import argparse
import os
import time
import requests
import xarray as xr

# Primary + fallback ERDDAP mirrors hosting the same OISST v2.1 dataset
ERDDAP_MIRRORS = [
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180",
    "https://polarwatch.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180",
]


def build_query_url(base, lat_min, lat_max, lon_min, lon_max, date_start, date_end):
    return (
        f"{base}.nc?"
        f"sst[({date_start}):1:({date_end})][(0.0):1:(0.0)]"
        f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    )


def try_download(url, out_path, max_retries=3, timeout=180):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}...")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            with open(out_path, "wb") as f:
                f.write(response.content)

            print(f"  Downloaded {len(response.content) / 1e6:.2f} MB")
            return True

        except requests.exceptions.RequestException as e:
            print(f"  Failed: {e}")
            if attempt < max_retries:
                wait = attempt * 5
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return False


def download(lat_min, lat_max, lon_min, lon_max, date_start, date_end, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    success = False
    for base in ERDDAP_MIRRORS:
        url = build_query_url(base, lat_min, lat_max, lon_min, lon_max, date_start, date_end)
        print(f"\nTrying mirror: {base}\n{url}\n")
        if try_download(url, out_path):
            success = True
            break

    if not success:
        raise RuntimeError(
            f"Failed to download from all ERDDAP mirrors after retries.\n"
            f"This is likely a temporary NOAA server issue. Try again in a few "
            f"minutes, or try a smaller date range / bounding box first to test "
            f"connectivity (e.g. --date_start 2023-01-01 --date_end 2023-01-31)."
        )

    # Verify it's valid NetCDF
    try:
        ds = xr.open_dataset(out_path)
    except Exception as e:
        with open(out_path, "r", errors="ignore") as f:
            preview = f.read(500)
        raise RuntimeError(
            f"Downloaded file is not valid NetCDF (ERDDAP likely returned an "
            f"error page). First 500 chars:\n{preview}\n\nOriginal error: {e}"
        )

    if "zlev" in ds.dims:
        ds = ds.squeeze("zlev", drop=True)
        ds.to_netcdf(out_path)

    print(f"\nVerified valid NetCDF file.")
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