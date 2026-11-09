"""OMIE day-ahead prices (marginalpdbc daily files) -> one tidy CSV.

Each daily file holds the prices for the Spanish zone, published after
the ~12:00 gate closure the day before delivery. Since the Oct 2025 MTU
change, files carry 96 quarter-hour periods (older days had 24 hours). Format:

    MARGINALPDBC;
    2026;11;06;1;275.3;275.3;   (year;month;day;period;price_es;price_pt)

Files exist for ~6 years back. No key needed. Downloads are incremental:
only days missing from the CSV are fetched.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

URL = ("https://www.omie.es/es/file-download?parents%5B%5D=marginalpdbc"
       "&filename=marginalpdbc_{ymd}.{ver}")
UA = {"User-Agent": "Mozilla/5.0 (luz-forecast data pull)"}
OUT = Path(__file__).resolve().parent / "raw" / "omie_prices.csv"
FIRST_DAY = date(2023, 1, 1)  # oldest files on the current OMIE site (~4y rolling)


def fetch_day(d: date) -> list[tuple[date, int, float]] | None:
    """One day's (date, period 1-96, EUR/MWh) rows, or None if unavailable."""
    ymd = d.strftime("%Y%m%d")
    for ver in (1, 2):
        try:
            req = urllib.request.Request(URL.format(ymd=ymd, ver=ver), headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("latin-1")
            rows = []
            for line in text.splitlines()[1:]:
                parts = line.split(";")
                if len(parts) < 6 or not parts[0]:
                    continue
                rows.append((d, int(parts[3]), float(parts[4])))
            return rows if rows else None
        except urllib.error.HTTPError as e:
            if e.code == 404 and ver == 1:
                continue
            return None
        except Exception:
            time.sleep(2)  # transient: one quiet retry happens on next run
            return None
    return None


def update(out: Path = OUT, start: date = FIRST_DAY,
           end: date | None = None) -> int:
    """Fetch missing days up to `end` (default: yesterday), one month at a
    time so a killed run keeps finished months. Returns rows added."""
    out.parent.mkdir(parents=True, exist_ok=True)
    end = end or (date.today() - timedelta(days=1))
    have = pd.DataFrame()
    if out.exists():
        have = pd.read_csv(out, parse_dates=["date"])
    last = have["date"].max().date() if len(have) else start - timedelta(days=1)
    total = 0
    while last < end:
        nxt = last + timedelta(days=1)
        month_end = (nxt.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        chunk_end = min(end, month_end)
        days = [nxt + timedelta(days=i)
                for i in range((chunk_end - nxt).days + 1)]
        got = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for res in ex.map(fetch_day, days):
                if res:
                    got.extend(res)
        if got:
            new = pd.DataFrame(got, columns=["date", "period", "price_eur_mwh"])
            new["date"] = pd.to_datetime(new["date"])
            have = (pd.concat([have, new])
                    .drop_duplicates(["date", "period"], keep="last")
                    .sort_values(["date", "period"]))
            have.to_csv(out, index=False)
            total += len(new)
        last = chunk_end
        print(f"  {chunk_end}: {total} rows so far", file=sys.stderr)
    return total


if __name__ == "__main__":
    n = update()
    print(f"omie: {n} rows added -> {OUT}", file=sys.stderr)
