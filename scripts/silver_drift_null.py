# Post-verdict descriptive check for scripts/silver_variants.py (2026-10-02). Run from the repo root.
"""Descriptive, post-verdict: does the selected silver variant beat RANDOM long entries
with identical stop/trail/cost mechanics (the drift null)?"""
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import numpy as np, pandas as pd
import silver_variants as sv, replay_breakout_news as rbn
from src.indicators import calculate_atr
sv.rbm.register(); fr = sv.frames(); df = fr["4h"]

def random_longs(df, n_entries_per_period, k=2, m=27, rng=None, spread=3.0, fin=rbn.GOLD_ALL_IN, htf_on=True):
    o,hi,lo,cl=(df[c].to_numpy(float) for c in ("open","high","low","close"))
    atr=calculate_atr(df["high"],df["low"],df["close"],14).to_numpy(float)
    htf=df["htf"].to_numpy(object); dates=df["date"].to_numpy("datetime64[ns]")
    elig=np.where(np.isfinite(atr)&(atr>0)&((htf=="BULLISH") if htf_on else True))[0]
    elig=elig[elig>60]
    out=[]
    for i in np.sort(rng.choice(elig,size=n_entries_per_period,replace=False)):
        entry=cl[i]; sd=max(atr[i]*k,4.0); stop=entry-sd; j=i+1; xp=None
        while j<len(df):
            if lo[j]<=stop: xp=min(o[j],stop); break
            if j-m>=0:
                lvl=lo[j-m:j].min()
                if lo[j]<=lvl: xp=min(o[j],lvl); break
            j+=1
        if xp is None: xp,j=cl[-1],len(df)-1
        nights=int((pd.Timestamp(dates[j]).normalize()-pd.Timestamp(dates[i]).normalize()).days)
        out.append(((xp-entry)-spread-nights*fin/365*entry)/sd)
    return np.array(out)

v={"family":"A","tf":"4h","n":55,"k":2,"htf":True,"dir":"long"}
t=sv.simulate(v,fr,3.0,rbn.GOLD_ALL_IN); t=t[t.date>=sv.START]
rng=np.random.default_rng(42)
for name,(a,b) in {"discovery 2005-15":("2005-01-01","2015-12-31"),"holdout 2016-26":("2016-01-01","2026-12-31")}.items():
    real=t[(t.date>=a)&(t.date<=b)].r
    sub=df[(df.date>=a)&(df.date<=b)].reset_index(drop=True)
    null=np.array([random_longs(sub,len(real),rng=rng).mean() for _ in range(300)])
    print(f"{name}: breakout long mean {real.mean():+.3f} (n={len(real)}) | random long entries (same exits, HTF bullish only): mean {null.mean():+.3f}, 95% band [{np.percentile(null,2.5):+.3f}, {np.percentile(null,97.5):+.3f}] | share of random >= real: {(null>=real.mean()).mean():.2f}")
# silver buy-and-hold for context
c=fr["1D"].set_index("date")["close"]
for a,b in (("2005-01-01","2015-12-31"),("2016-01-01","2026-09-21")):
    x=c[a:b]; print(f"silver {a[:4]}-{b[:4]}: ${x.iloc[0]:.2f} -> ${x.iloc[-1]:.2f} ({(x.iloc[-1]/x.iloc[0]-1)*100:+.0f}%)")
