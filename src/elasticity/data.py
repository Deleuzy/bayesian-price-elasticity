"""Dominick's Finer Foods scanner data: download, checksum and tidy panel.

Terms (Kilts Center, Chicago Booth): "These data are for academic research
purposes only. Users must acknowledge in their working papers and/or
publications the Kilts Center for Marketing at the University of Chicago Booth
School of Business." The raw files are therefore not committed; this module
downloads them from the Kilts Center and checks their SHA-256.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_time_features

BASE = "https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset"
FILES = {
    "upcche.csv": f"{BASE}/upc_csv-files/upcche.csv",
    "wche.zip": f"{BASE}/movement_csv-files/wche.zip",
}
SHA256 = {
    "upcche.csv": "318363a838f773f99c0397f1789e67f681e21c13b0929e34a3c0be982d6523ff",
    "wche.zip": "9eed85e4af7425aaa94799659646999730ea1ba4339856ff6a4c3b38e09041d2",
}
RAW = Path("data/raw")
CACHE = Path("data/cache")
WEEK1_START = pd.Timestamp("1989-09-14")  # manual, Part 7: week 1 = 09/14/89 to 09/20/89
SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*OZ\.?\s*$", re.IGNORECASE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(raw: Path = RAW) -> None:
    """Download the cheese UPC and movement files if missing and verify hashes."""
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = raw / name
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
        got = sha256(path)
        if got != SHA256[name]:
            raise ValueError(f"{name}: sha256 {got} does not match {SHA256[name]}")


def read_movement(raw: Path = RAW) -> pd.DataFrame:
    """Stream the 657 MB movement CSV, keeping valid rows with positive sales."""
    cols = ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "PROFIT", "OK"]
    parts = []
    with zipfile.ZipFile(raw / "wche.zip") as z, z.open("wche.csv") as fh:
        for ch in pd.read_csv(fh, usecols=cols, chunksize=2_000_000, dtype={"SALE": "string"}):
            ch = ch[(ch.OK == 1) & (ch.PRICE > 0) & (ch.MOVE > 0) & (ch.QTY > 0)]
            parts.append(ch.drop(columns="OK"))
    return pd.concat(parts, ignore_index=True)


def label(descrip: str, size: str) -> str:
    """Readable name from the truncated UPC description (KR = Kraft, DOM = Dominick's own brand)."""
    d = re.sub(r"^KR\.?\s*", "Kraft ", descrip.strip())
    d = re.sub(r"^DOM\s+", "DFF own ", d)
    words = [w if w in ("Kraft", "DFF", "own") else w.capitalize() for w in d.split()]
    return f"{' '.join(words)} {size.strip().lower()}"


def choose_products(mv: pd.DataFrame, upc: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top n UPCs by revenue among fixed-weight packs (size given in ounces).

    Random-weight chunks ("8 TO 1") are excluded because their unit is unclear.
    """
    rev = (mv.PRICE * mv.MOVE / mv.QTY).groupby(mv.UPC).sum().rename("revenue")
    u = upc.drop_duplicates("UPC").set_index("UPC")
    u = u[u.SIZE.astype(str).str.match(SIZE_RE)]
    u = u[~u.DESCRIP.astype(str).str.strip().str.lower().isin(["end", ""])]
    top = rev[rev.index.isin(u.index)].sort_values(ascending=False).head(n)
    out = u.loc[top.index, ["DESCRIP", "SIZE"]].copy()
    out["revenue"] = top
    out["label"] = [label(d, s) for d, s in zip(out.DESCRIP, out.SIZE, strict=True)]
    return out.reset_index()


def tidy(mv: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """One row per product x store x week with logs, cost, promo flag and instruments."""
    d = mv[mv.UPC.isin(products.UPC)].copy()
    d["price"] = d.PRICE / d.QTY
    # PROFIT is DFF's gross margin in percent, so unit cost = price * (1 - margin)
    # (average acquisition cost, manual remark 3).
    d["cost"] = d.price * (1 - d.PROFIT / 100)
    d = d[(d.cost > 0) & (d.PROFIT > -50) & (d.PROFIT < 90)]
    d["units"] = d.MOVE.astype(float)
    d["deal"] = d.SALE.notna().astype(float)
    lab = dict(zip(products.UPC, products.label, strict=True))
    order = list(products.UPC)
    d["product"] = pd.Categorical(d.UPC.map(lab), categories=[lab[u] for u in order])
    d = d.rename(columns={"STORE": "store", "WEEK": "week"})
    d["log_q"] = np.log(d.units)
    d["log_p"] = np.log(d.price)
    d["log_cost"] = np.log(d.cost)
    d = add_hausman(d)
    d["date"] = WEEK1_START + pd.to_timedelta((d.week - 1) * 7, unit="D")
    d = add_time_features(d)
    keep = [
        "product", "store", "week", "date", "units", "price", "cost", "deal",
        "log_q", "log_p", "log_cost", "hausman", "t", "quarter", "s1", "c1", "s2", "c2",
    ]  # fmt: skip
    return d[keep].sort_values(["product", "store", "week"]).reset_index(drop=True)


def add_hausman(d: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out mean log price of the same product in all other stores that week."""
    g = d.groupby(["product", "week"], observed=True).log_p
    s, n = g.transform("sum"), g.transform("count")
    d = d.copy()
    d["hausman"] = np.where(n > 1, (s - d.log_p) / (n - 1).clip(lower=1), np.nan)
    return d.dropna(subset=["hausman"])


def load_panel(n_products: int = 12, raw: Path = RAW, cache: Path = CACHE) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"cheese_top{n_products}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    upc = pd.read_csv(raw / "upcche.csv")
    mv = read_movement(raw)
    products = choose_products(mv, upc, n_products)
    panel = tidy(mv, products)
    panel.to_parquet(path)
    return panel


def main() -> None:
    """Download the two Kilts files if missing, verify SHA-256, and build the panel."""
    download()
    panel = load_panel(12)
    print(f"panel: {len(panel):,} store-week rows, {panel['product'].nunique()} products")


if __name__ == "__main__":
    main()
