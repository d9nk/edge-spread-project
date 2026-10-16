from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

AUTHORS_URL = "https://raw.githubusercontent.com/eguidotti/bidask/main/pseudocode/{name}.csv"
AUTHORS_DIR = os.path.join(ROOT, "data", "authors")


def _authors_file(name: str):
    import pandas as pd

    path = os.path.join(AUTHORS_DIR, f"bidask_{name}.csv")
    if not os.path.exists(path):
        try:
            import requests

            r = requests.get(AUTHORS_URL.format(name=name), timeout=60)
            r.raise_for_status()
            os.makedirs(AUTHORS_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as exc:
            pytest.skip(f"the authors' test data could not be downloaded: {exc!r}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def authors_ohlc():
    return _authors_file("ohlc")


@pytest.fixture(scope="session")
def authors_ohlc_miss():
    return _authors_file("ohlc-miss")
