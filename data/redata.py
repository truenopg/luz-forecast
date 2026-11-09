"""REE apidatos: national electricity demand (real and forecast) -> tidy CSV.

One request per day to demanda/demanda-tiempo-real, which returns 5-minute
series; we aggregate to hourly means. The API throws transient 400s under
load, so each day is retried with backoff and skipped if it keeps failing
(the next incremental run picks it up).

Series kept: Real (measured) and Prevista (REE's forecast). Prevista is the
leakage-safe candidate feature, but its values are revised intraday - see
README: at feature time we must only use the forecast as known at the
day-ahead gate closure (12:00 D-1), or lag it.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

URL = ("https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real"
       "?start_date={d}T00:00&end_date={d}T23:59&time_trunc=hour")
UA = {"User-Agent": "Mozilla/5.0 (luz-forecast data pull)",
      "Accept": "application/json"}
OUT = Path(__file__).resolve().parent / "raw" / "ree_demand.csv"
FIRST_DAY = date(2023, 1, 1)


def fetch_day(d: date) -> list[tuple] | None:
    import json
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(d=d.isoformat()), headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.load(r)
            rows = []
            for inc in payload["included"]:
                title = inc["attributes"]["title"]
                if title not in ("Real", "Prevista"):
                    continue
                for v in inc["attributes"]["values"]:
                    ts = pd.Timestamp(v["datetime"]).floor("h")
                    rows.append((d, ts.hour, title.lower(), float(v["value"])))
            return rows or None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def update(out: Path = OUT, start: date = FIRST_DAY,
           end: date | None = None) -> int:
    """Hourly MW per day, one row per (date, hour, series). Month chunks so
    a killed run keeps finished months."""
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
        days = [nxt + timedelta(days=i) for i in range((chunk_end - nxt).days + 1)]
        got = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for res in ex.map(fetch_day, days):
                if res:
                    got.extend(res)
        if got:
            new = pd.DataFrame(got, columns=["date", "hour", "series", "mw"])
            new["date"] = pd.to_datetime(new["date"])
            new = (new.groupby(["date", "hour", "series"], as_index=False)["mw"]
                   .mean())  # 5-min -> hourly mean
            have = (pd.concat([have, new])
                    .drop_duplicates(["date", "hour", "series"], keep="last")
                    .sort_values(["date", "hour", "series"]))
            have.to_csv(out, index=False)
            total += len(new)
        last = chunk_end
        print(f"  {chunk_end}: {total} rows so far", file=sys.stderr)
    return total


if __name__ == "__main__":
    n = update()
    print(f"ree demand: {n} rows added -> {OUT}", file=sys.stderr)
