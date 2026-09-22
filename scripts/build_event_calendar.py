#!/usr/bin/env python3
"""Build the historical high-impact event table for the breakout news replay.

Pre-registered in research_notes.md ("PRE-REGISTRATION — Breakout news-proximity
replay", 2026-09-22). Official sources only, one row per scheduled release, times
in UTC via zoneinfo (so every DST changeover is right — the live calendar's fixed
+5h was not):

  US NFP, CPI   FRED release/dates (ids 50, 10)             08:30 America/New_York
  FOMC          federalreserve.gov historical + calendar pages, scheduled meetings only
  BoE           mpcvoting.xlsx "Bank Rate Decisions"          12:00 Europe/London
  ECB           ecb.europa.eu monetary-policy-statement lists, decision + press conf
  UK CPI        ONS CPI bulletin archive (from 2015 — ONS's own boundary)

    FRED_API_KEY=... python3 scripts/build_event_calendar.py
    # or put the key in data/news_events/.fred_key (gitignored)

Writes data/news_events/events.csv: utc, currency, event, tier, source.
Network only; touches no bot state.
"""
from __future__ import annotations

import csv
import io
import os
import re
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "news_events"
START_YEAR = 2004
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "Chrome/124 Safari/537.36"}
NY, LDN, FRA, UTC = (ZoneInfo(z) for z in ("America/New_York", "Europe/London",
                                             "Europe/Berlin", "UTC"))
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}
MON3 = {k[:3]: v for k, v in MONTHS.items()}

# Unscheduled decisions a calendar could not have known about in advance.
BOE_UNSCHEDULED = {date(2008, 10, 8), date(2020, 3, 11), date(2020, 3, 19)}


def utc(d: date, hh: int, mm: int, tz: ZoneInfo) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz).astimezone(UTC)


def get(url: str, attempts: int = 3, **kw) -> requests.Response:
    for attempt in range(attempts):
        try:
            r = requests.get(url, headers=UA, timeout=45, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * attempt ** 2)   # ONS rate-limits bursts
    raise RuntimeError(f"fetch failed: {url}")


def fred_key() -> str:
    k = os.environ.get("FRED_API_KEY") or (OUT / ".fred_key").read_text().strip()
    if not k:
        sys.exit("no FRED API key (env FRED_API_KEY or data/news_events/.fred_key)")
    return k


# ----------------------------------------------------------------------------- US data

def us_releases() -> list[dict]:
    key, rows = fred_key(), []
    for rid, name in ((50, "NFP"), (10, "CPI")):
        j = get("https://api.stlouisfed.org/fred/release/dates", params={
            "release_id": rid, "api_key": key, "file_type": "json",
            "realtime_start": f"{START_YEAR}-01-01", "limit": 10000}).json()
        dates = sorted(date.fromisoformat(r["date"]) for r in j["release_dates"])
        if name == "CPI":
            # February carries an extra seasonal-factor revision a few days BEFORE
            # the real CPI; keep the last date in each month.
            by_month: dict = {}
            for d in dates:
                by_month[(d.year, d.month)] = d
            dates = sorted(by_month.values())
        rows += [{"utc": utc(d, 8, 30, NY), "currency": "USD", "event": name,
                  "tier": "core", "source": f"FRED release {rid}"} for d in dates]
    return rows


# ----------------------------------------------------------------------------- FOMC

# The Fed abbreviates the month on meetings that span two months: "Feb 31-1" is
# Jan 31-Feb 1, labelled with the END month. The statement is on the last day.
_MON = (r"(?:January|February|March|April|May|June|July|August|September|October|November|"
        r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)")
HIST_RE = re.compile(
    rf"({_MON})\s+(\d{{1,2}})(?:\s*-\s*(?:({_MON})\s+)?(\d{{1,2}}))?\s+(Meeting|Conference Call)"
    r"\s+-\s+(\d{4})")


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def fomc_statement_time(d: date, press_conf: bool) -> tuple[int, int]:
    """ET release time of the statement (see amendment B2 in research_notes.md)."""
    if d >= date(2013, 1, 1):
        return 14, 0
    if d >= date(2011, 4, 1) and press_conf:
        return 12, 30
    return 14, 15


def fomc_press_time(d: date) -> tuple[int, int]:
    return (14, 30) if d >= date(2013, 1, 1) else (14, 15)


def fomc() -> tuple[list[dict], dict]:
    cal = get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm").text
    cal_years = sorted({int(y) for y in re.findall(r"(\d{4}) FOMC Meetings", cal)})
    meetings: list[tuple[date, bool]] = []
    # historical years
    for y in range(START_YEAR, cal_years[0]):
        text = _strip(get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm").text)
        heads = list(HIST_RE.finditer(text))
        for i, m in enumerate(heads):
            if m.group(5) != "Meeting" or int(m.group(6)) != y:
                continue            # conference calls are unscheduled
            seg = text[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
            month = MON3[(m.group(3) or m.group(1))[:3]]
            day = int(m.group(4) or m.group(2))
            meetings.append((date(y, month, day), "Press Conference" in seg))
    # calendar page years
    blocks = re.split(r"(\d{4}) FOMC Meetings", cal)
    for j in range(1, len(blocks), 2):
        y, body = int(blocks[j]), blocks[j + 1]
        pairs = re.findall(
            r"fomc-meeting__month[^>]*>(?:\s*<[^>]+>)*\s*([^<]+?)\s*<.*?fomc-meeting__date[^>]*>\s*([^<]+?)\s*<",
            body, flags=re.S)
        for mon, dd in pairs:
            if "unscheduled" in dd.lower() or "notation" in dd.lower():
                continue
            mon_last = mon.split("/")[-1].strip()[:3]
            day_last = int(re.findall(r"\d+", dd)[-1])
            meetings.append((date(y, MON3[mon_last], day_last), y >= 2019 or "*" in dd))
    today = date.today()
    rows = []
    for d, pc in sorted(set(meetings)):
        if d.year < START_YEAR or d > today:
            continue
        hh, mm = fomc_statement_time(d, pc)
        rows.append({"utc": utc(d, hh, mm, NY), "currency": "USD", "event": "FOMC statement",
                     "tier": "core", "source": "federalreserve.gov"})
        if pc:
            hh, mm = fomc_press_time(d)
            rows.append({"utc": utc(d, hh, mm, NY), "currency": "USD", "event": "FOMC press conference",
                         "tier": "core", "source": "federalreserve.gov"})
    return rows, {"fomc_meetings": len({d for d, _ in meetings if START_YEAR <= d.year and d <= today}),
                  "fomc_calendar_years": f"{cal_years[0]}-{cal_years[-1]}"}


# ----------------------------------------------------------------------------- BoE

def _xlsx_column_dates(blob: bytes, sheet_name: str, col: str) -> list[date]:
    """Excel-serial dates from one column of one sheet, without openpyxl (the bot's
    locked venv doesn't carry it)."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    sid = next(s.get(f"{{{ns['r']}}}id") for s in wb.find("m:sheets", ns) if s.get("name") == sheet_name)
    path = "xl/" + target[sid].lstrip("/").removeprefix("xl/")
    sheet = ET.fromstring(z.read(path))
    out = []
    for c in sheet.iter(f"{{{ns['m']}}}c"):
        ref = c.get("r", "")
        if re.fullmatch(rf"{col}\d+", ref) and c.get("t") is None:
            v = c.find("m:v", ns)
            if v is not None:
                x = float(v.text)
                if 30000 < x < 60000:
                    out.append(date(1899, 12, 30) + timedelta(days=int(x)))
    return out


def boe() -> tuple[list[dict], dict]:
    blob = get("https://www.bankofengland.co.uk/-/media/boe/files/"
               "monetary-policy-summary-and-minutes/mpcvoting.xlsx").content
    ds = sorted(set(_xlsx_column_dates(blob, "Bank Rate Decisions", "B")))
    kept = [d for d in ds if d.year >= START_YEAR and d not in BOE_UNSCHEDULED]
    rows = [{"utc": utc(d, 12, 0, LDN), "currency": "GBP", "event": "BoE rate decision",
             "tier": "core", "source": "BoE mpcvoting.xlsx"} for d in kept]
    return rows, {"boe_decisions": len(kept),
                  "boe_unscheduled_dropped": sorted(str(d) for d in ds if d in BOE_UNSCHEDULED)}


# ----------------------------------------------------------------------------- ECB

def ecb() -> tuple[list[dict], dict]:
    rows, n = [], 0
    for y in range(START_YEAR, date.today().year + 1):
        html = get(f"https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/"
                   f"{y}/html/index_include.en.html").text
        ids = sorted(set(re.findall(r"is(\d{6})", html)))
        for s in ids:
            d = date(2000 + int(s[:2]), int(s[2:4]), int(s[4:6]))
            if d.year != y:
                continue
            n += 1
            new = d >= date(2022, 7, 21)
            rows.append({"utc": utc(d, 14 if new else 13, 15 if new else 45, FRA), "currency": "EUR",
                         "event": "ECB decision", "tier": "core", "source": "ecb.europa.eu"})
            rows.append({"utc": utc(d, 14, 45 if new else 30, FRA), "currency": "EUR",
                         "event": "ECB press conference", "tier": "core", "source": "ecb.europa.eu"})
        time.sleep(0.5)
    return rows, {"ecb_meetings": n}


# ----------------------------------------------------------------------------- UK CPI

UK_CPI_0700_FROM = date(2020, 4, 1)


def uk_cpi_time(d: date) -> tuple[int, int]:
    """ONS pages carry no release time. The 09:30 -> 07:00 move was dated from
    GBP/USD 1-minute reaction on the release days (amendment B3, MEDIUM): the last
    clear 09:30 prints are 2020-01/02, 07:00 leads from 2020-04 and is unambiguous
    from 2021-07. The replay runs a both-times sensitivity over 2020-04..2021-06."""
    return (7, 0) if d >= UK_CPI_0700_FROM else (9, 30)


def uk_cpi() -> tuple[list[dict], dict]:
    """Release dates from the ONS bulletin archive (from 2015 - ONS's own boundary)."""
    base = "https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/previousreleases"
    dates, page = set(), 1
    while True:
        text = _strip(get(f"{base}?page={page}", attempts=6).text)
        found = re.findall(r"Released on (\d{1,2}) ([A-Z][a-z]+) (20\d{2})", text)
        if not found:
            break
        dates |= {date(int(y), MONTHS[m], int(d)) for d, m, y in found}
        page += 1
        time.sleep(4)
        if page > 40:
            break
    rows = [{"utc": utc(d, *uk_cpi_time(d), LDN), "currency": "GBP", "event": "UK CPI",
             "tier": "core", "source": "ONS bulletin archive"} for d in sorted(dates)]
    return rows, {"uk_cpi_releases": len(dates), "uk_cpi_first": str(min(dates)) if dates else None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, notes = us_releases(), {}
    for fn in (fomc, boe, ecb, uk_cpi):
        r, n = fn()
        rows += r
        notes.update(n)
    rows.sort(key=lambda r: r["utc"])
    with open(OUT / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utc", "currency", "event", "tier", "source"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "utc": r["utc"].strftime("%Y-%m-%dT%H:%M:%SZ")})
    counts: dict = {}
    for r in rows:
        counts[r["event"]] = counts.get(r["event"], 0) + 1
    print(f"{len(rows)} events → {OUT / 'events.csv'}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<24} {v}")
    for k, v in notes.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
