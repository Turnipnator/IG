#!/usr/bin/env python3
"""IG-native breakout HTF ladder (2026-09-09). Usage: IG_ARCHIVE_DIR=<dir> scripts/backtest_breakout_htf_ladder_ignative.py
IG-native breakout ladder: HTF gate {NONE, HOUR, DAY} x fill model {level+0.286ATR (script convention), close+spread (live mechanism)}
for every market with a BREAKOUT_CONFIG and an archive. Same entry/stop/trail as scripts/backtest_index_breakout_ignative.py."""
import os, sys, json, dataclasses, importlib
import numpy as np, pandas as pd
ARCH=os.environ.get('IG_ARCHIVE_DIR','data/candle_archive')
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE)); sys.path.insert(0,HERE)
bt=importlib.import_module('backtest_index_breakout_ignative')
from config import MARKETS
from src.breakout import BREAKOUT_CONFIGS
from src.indicators import calculate_atr
SPREAD={'IX.D.SPTRD.DAILY.IP':0.61,'IX.D.NASDAQ.CASH.IP':2.28,'IX.D.NIKKEI.DAILY.IP':9.63,'IX.D.HANGSENG.DAILY.IP':7.11,'IX.D.DOW.DAILY.IP':3.2,'IX.D.FTSE.DAILY.IP':1.3,
        'IX.D.AIIDX.DAILY.IP':7.9,'IX.D.RUSSELL.DAILY.IP':0.3,'CS.D.USCGC.TODAY.IP':0.4,'CS.D.EURUSD.TODAY.IP':0.6,'CS.D.GBPUSD.TODAY.IP':0.9,'CC.D.CL.USS.IP':3.0,'CC.D.DX.USS.IP':8.1}

def run(epic, mk, htf_mode, fill):
    cfg=BREAKOUT_CONFIGS[epic]; df=bt.load_1h(epic)
    df['atr']=calculate_atr(df['high'],df['low'],df['close'],14)
    if htf_mode=='NONE': df['htf']='ANY'
    else: df['htf']=bt.htf_trend_series(df, htf_mode, pd.Index(df['date'])).values
    N,M,k=cfg.n,cfg.m,cfg.stop_atr_mult
    h_lo,h_hi=(mk.trading_start+1)%24,(mk.trading_end+1)%24
    trades=[]; i=N+1; n=len(df)
    while i<n:
        r=df.iloc[i]; atr=r['atr']
        if not np.isfinite(atr) or atr<=0: i+=1; continue
        prior=df.iloc[i-N:i]; upper,lower=prior['high'].max(),prior['low'].min()
        d=None
        if r['high']>=upper and r['htf'] in ('BULLISH','ANY'): d,level='BUY',upper
        elif r['low']<=lower and r['htf'] in ('BEARISH','ANY'): d,level='SELL',lower
        if d is None: i+=1; continue
        hh=r['date'].hour
        if (hh<h_lo or hh>=h_hi) if h_lo<h_hi else (h_hi<=hh<h_lo): i+=1; continue
        if fill=='level': entry=max(r['open'],level) if d=='BUY' else min(r['open'],level); cost=0.286*atr
        else: entry=r['close']; cost=SPREAD.get(epic,0.0)
        sd=max(atr*k,mk.min_stop_distance); stop=entry-sd if d=='BUY' else entry+sd
        j=i+1; xp=None
        while j<n:
            b=df.iloc[j]
            if d=='BUY' and b['low']<=stop: xp=min(b['open'],stop); why='stop'; break
            if d=='SELL' and b['high']>=stop: xp=max(b['open'],stop); why='stop'; break
            if j-M>=0:
                pm=df.iloc[j-M:j]
                if d=='BUY':
                    lvl=pm['low'].min()
                    if b['low']<=lvl: xp=min(b['open'],lvl); why='trail'; break
                else:
                    lvl=pm['high'].max()
                    if b['high']>=lvl: xp=max(b['open'],lvl); why='trail'; break
            j+=1
        if xp is None: xp=df.iloc[-1]['close']; why='eod'; j=n-1
        g=(xp-entry) if d=='BUY' else (entry-xp)
        trades.append(dict(d=d,r=(g-cost)/sd,why=why,date=r['date'],entry=entry,bars=j-i))
        i=j+1
    return pd.DataFrame(trades)

def summ(t):
    if len(t)==0: return 'n=0'
    w=t.r[t.r>0].sum(); l=-t.r[t.r<=0].sum(); pf=w/l if l>0 else float('inf')
    return f'n={len(t):3d} {int((t.r>0).sum()):2d}W ΣR={t.r.sum():+7.2f} R/t={t.r.mean():+6.3f} PF={pf:4.2f}'
mks=[m for m in MARKETS if m.epic in BREAKOUT_CONFIGS and os.path.exists(f'{ARCH}/{m.epic}.jsonl') and os.path.getsize(f'{ARCH}/{m.epic}.jsonl')>200000]
print('window: archive 2026-06-12 -> 2026-09-09; DAY HTF needs 21 daily bars so DAY signals start ~07-05; HOUR from ~06-14')
for fill in ('level','close'):
    print(f'\n===== FILL = {"level + 0.286xATR (script convention)" if fill=="level" else "bar CLOSE + trading-hours spread (live mechanism)"} =====')
    print(f'{"market":18s} | {"HTF=NONE":42s} | {"HTF=HOUR (indices live cfg)":42s} | {"HTF=DAY":42s}')
    pooled={'NONE':[], 'HOUR':[], 'DAY':[]}
    for mk in mks:
        cells=[]
        for hm in ('NONE','HOUR','DAY'):
            t=run(mk.epic,mk,hm,fill); pooled[hm].append(t); cells.append(summ(t))
        print(f'{mk.name:18s} | {cells[0]:42s} | {cells[1]:42s} | {cells[2]:42s}')
    print(f'{"** POOLED **":18s} | ' + ' | '.join(f'{summ(pd.concat(pooled[hm])) if any(len(x) for x in pooled[hm]) else "n=0":42s}' for hm in ('NONE','HOUR','DAY')))
# Gold DAY-HTF close-fill trades vs the journal's live Gold breakout entries (sanity check of the engine against reality)
t=run('CS.D.USCGC.TODAY.IP',[m for m in MARKETS if m.epic=='CS.D.USCGC.TODAY.IP'][0],'DAY','close')
print('\n### Gold DAY/close-fill replay trades (compare with journal live Gold breakout entries 08-03 15:05 S, 08-05 13:05 B, 08-06 01:05 B, 08-07 10:05 B, 08-07 13:35 B, 08-10 20:05 B, 08-17 16:05 B, 08-19 14:15 B, 09-09 02:05 S)')
print(t[['date','d','entry','why','bars','r']].to_string(index=False))
