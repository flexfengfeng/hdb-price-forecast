"""Download the latest HDB resale flat prices (Jan 2017 onwards) from data.gov.sg.

Run:  python download_data.py
Saves: data/hdb_resale.csv
"""
import time
from pathlib import Path

import pandas as pd
import requests

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
API_BASE = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}"
OUT_PATH = Path(__file__).parent / "data" / "hdb_resale.csv"


def download(timeout_s=120):
    # data.gov.sg prepares the file first, then gives us a download link
    requests.get(f"{API_BASE}/initiate-download", timeout=15).raise_for_status()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = requests.get(f"{API_BASE}/poll-download", timeout=15).json()
        if body["data"].get("status") == "DOWNLOAD_SUCCESS":
            return pd.read_csv(body["data"]["url"])
        time.sleep(3)
    raise TimeoutError("data.gov.sg download was not ready in time")


if __name__ == "__main__":
    df = download()
    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {OUT_PATH}  ({df['month'].min()} to {df['month'].max()})")
