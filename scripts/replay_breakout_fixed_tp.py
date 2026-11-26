"""Fixed take-profit on the breakout engine (21y Dukascopy). Descriptive, grid stated up front:
TP = 1R, 1.5R, 2R, 3R, 5R (R = initial stop), 0.5x and 1.0x prior-day ATR40 (daily), and 50 pts (Gold).
Stop checked before TP inside a bar (conservative). After TP the engine may re-enter on the next break, paying cost."""
import sys
sys.path.insert(0, "/Users/paulturner/IG/scripts"); sys.path.insert(0, "/Users/paulturner/IG")
import numpy as np, pandas as pd
import replay_breakout_news as rbn
import replay_breakout_atr_exit as rx
from config import MARKETS
from src.breakout import BREAKOUT_CONFIGS
from src.indicators import calculate_atr

def run(df, epic, diffs, tp_kind=None, tp_val=0.0, directions=("BUY",)):
    cfg = BREAKOUT_CONFIGS[epic]; mk = next(m for m in MARKETS if m.epic == epic)
    N, M, k = cfg.n, cfg.m, cfg.stop_atr_mult
    h_lo, h_hi = (mk.trading_start + 1) % 24, (mk.trading_end + 1) % 24
    o, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr = calculate_atr(df["high"], df["low"], df["close"], 14).to_numpy(float)
    datr = df["datr40"].to_numpy(float)
    htf = df["htf"].to_numpy(object); dates = df["date"].to_numpy("datetime64[ns]"); hours = df["date"].dt.hour.to_numpy()
    cost = rbn.SPREAD[epic]; out = []; i, n = N + 1, len(df)
    while i < n:
        a = atr[i]
        if not np.isfinite(a) or a <= 0: i += 1; continue
        upper, lower = hi[i - N:i].max(), lo[i - N:i].min()
        d = "BUY" if hi[i] >= upper else "SELL" if lo[i] <= lower else None
        if d is None or d not in directions: i += 1; continue
        hh = hours[i]
        if (hh < h_lo or hh >= h_hi) if h_lo < h_hi else (h_hi <= hh < h_lo): i += 1; continue
        if htf[i] != ("BULLISH" if d == "BUY" else "BEARISH"): i += 1; continue
        s = 1.0 if d == "BUY" else -1.0
        entry = cl[i]; sd = max(a * k, mk.min_stop_distance); stop = entry - s * sd
        dist = None
        if tp_kind == "R": dist = tp_val * sd
        elif tp_kind == "D": dist = tp_val * datr[i] if np.isfinite(datr[i]) else None
        elif tp_kind == "P": dist = tp_val
        tp = entry + s * dist if dist else None
        j, xp, why = i + 1, None, None
        while j < n:
            if s * (lo[j] if d == "BUY" else hi[j]) <= s * stop:
                xp, why = (min(o[j], stop) if d == "BUY" else max(o[j], stop)), "stop"; break
            lvl = lo[j - M:j].min() if d == "BUY" else hi[j - M:j].max()
            if s * (lo[j] if d == "BUY" else hi[j]) <= s * lvl:
                xp, why = (min(o[j], lvl) if d == "BUY" else max(o[j], lvl)), "trail"; break
            if tp is not None and s * (hi[j] if d == "BUY" else lo[j]) >= s * tp:
                xp, why = (max(o[j], tp) if d == "BUY" else min(o[j], tp)), "tp"; break
            j += 1
        if xp is None: xp, why, j = cl[-1], "open-at-end", n - 1
        g = s * (xp - entry)
        nights = int((pd.Timestamp(dates[j]).normalize() - pd.Timestamp(dates[i]).normalize()).days)
        fin = nights * rbn.financing_rate(epic, d, pd.Timestamp(dates[i]), diffs) / 365.0 * entry
        mfe = max(s * ((hi[i+1:j+1].max() if d == "BUY" else lo[i+1:j+1].min()) - entry), 0) if j > i else 0.0
        out.append(dict(epic=epic, market=mk.name, date=pd.Timestamp(dates[i]), d=d, why=why, sd=sd, dist_r=(dist / sd if dist else np.nan),
                        r=(g - cost - fin) / sd, mfe_r=mfe / sd))
        i = j + 1
    return pd.DataFrame(out)

fr = rx.frames(); diffs = rbn.rate_diffs()
for e, df in fr.items():
    dd = df.set_index("date").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    da = calculate_atr(dd["high"], dd["low"], dd["close"], 40).shift(1)   # prior completed day
    df["datr40"] = da.reindex(df["date"].dt.normalize()).to_numpy()
ARMS = [("base", None, 0), ("TP 1R", "R", 1), ("TP 1.5R", "R", 1.5), ("TP 2R", "R", 2), ("TP 3R", "R", 3), ("TP 5R", "R", 5),
        ("TP 0.5xD-ATR40", "D", 0.5), ("TP 1.0xD-ATR40", "D", 1.0), ("TP 50pts", "P", 50.0)]
START = pd.Timestamp("2005-01-01")
def stats(t):
    if not len(t): return "n=0"
    r = t["r"]; top3 = r.nlargest(3).sum()
    h1 = t[t.date < "2016-01-01"]["r"].sum(); h2 = t[t.date >= "2016-01-01"]["r"].sum()
    return f"n={len(t):5d} net={r.sum():+8.1f}R mean={r.mean():+.3f} WR={100*(r>0).mean():4.0f}% maxW={r.max():5.1f} 05-15={h1:+7.1f} 16-26={h2:+7.1f} since2023={t[t.date>='2023-01-01']['r'].sum():+6.1f}"
for dirs in (("BUY",), ("SELL",)):
    print(f"\n===== {dirs[0]} =====")
    for e, df in fr.items():
        name = next(m for m in MARKETS if m.epic == e).name
        print(f"--- {name}")
        for label, kind, val in ARMS:
            if kind == "P" and name != "Gold": continue
            t = run(df, e, diffs, kind, val, dirs); t = t[t.date >= START] if len(t) else t
            extra = f" medTP={t['dist_r'].median():.2f}R hitTP={100*(t.why=='tp').mean():.0f}%" if kind in ("D", "P") and len(t) else ""
            print(f"{label:16s} {stats(t)}{extra}")
        if dirs == ("BUY",):
            b = run(df, e, diffs, None, 0, dirs); b = b[b.date >= START]
            print("   baseline MFE reach: " + " ".join(f"{x}R:{100*(b.mfe_r>=x).mean():.0f}%" for x in (0.5, 1, 1.5, 2, 3, 5)))
