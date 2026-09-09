#!/usr/bin/env python3
"""Replay every momentum row in rejected_signals against the IG candle archive with the live
engine's rules, per blocking gate — "what would each gate have cost or saved?".

Usage (from the repo root, venv active):
    scripts/replay_rejected.py --archive-dir <dir of <EPIC>.jsonl> --journal-json <dump.json> [--out-dir <dir>]

<dump.json> = {"rejected_signals": [...], "benched_outcomes": [...], "trades": [...]} — produce it on the VPS with
    docker exec -i ig-trading-bot python3 - <<'PY'
    import sqlite3, json
    db = sqlite3.connect('/app/data/trade_journal.db'); db.row_factory = sqlite3.Row
    print(json.dumps({t: [dict(r) for r in db.execute(f'select * from {t} order by id')]
                      for t in ('rejected_signals', 'benched_outcomes', 'trades')}, default=str))
    PY
and copy /root/ig-bot/data/candle_archive/ locally. Never run this inside the live container.

Rules LIVE: ATR stop (profile mult, min_stop floor, price cap), R:R limit, RSI-extreme exit, MACD-n exit
suppressed on the entry candle, BE at breakeven_trigger_pct -> stop = entry + lock, ATR trail after BE.
Rules RESOLVER: MACD-3, no BE/trail, 96-candle horizon (the bench resolver's model).
Candles are mid; spread charged once at exit. Archive and journal timestamps are both container-local.
Two calibrations print at the end: resolved shadow benches vs RESOLVER, real closed trades vs LIVE.
First run 2026-09-08 (research_notes.md "Per-gate replay of every rejected signal"): 77% / 91% agreement,
engine ≈ +0.04R/trade optimistic vs real fills (tick-level BE/trail is kinder at candle level).
"""
import argparse, os

import json, sys, math, collections, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETS
from src.strategy import get_strategy_for_market, MACD_EXIT_BARS
from src.indicators import add_all_indicators

_ap = argparse.ArgumentParser(); _ap.add_argument('--archive-dir', required=True); _ap.add_argument('--journal-json', required=True); _ap.add_argument('--out-dir', default='.')
_args = _ap.parse_args(); ARCHIVE = _args.archive_dir; JOURNAL = _args.journal_json; OUT = _args.out_dir
MK = {m.epic: m for m in MARKETS}
GBP_PER_R = 23.13
MACD_CHANGE = pd.Timestamp('2026-09-01 10:01')   # 9e2dd0c: MACD exit 3 -> 5
COOLDOWN = pd.Timedelta(minutes=30)
# spread per epic: benched_outcomes medians (live, at signal time); memory trading-hours figures where verified
SPREAD = {'IX.D.SPTRD.DAILY.IP': 0.61, 'IX.D.NASDAQ.CASH.IP': 2.28, 'IX.D.NIKKEI.DAILY.IP': 9.63, 'IX.D.HANGSENG.DAILY.IP': 7.11,
          'IX.D.DOW.DAILY.IP': 3.2, 'IX.D.FTSE.DAILY.IP': 1.3, 'IX.D.AIIDX.DAILY.IP': 7.9, 'IX.D.RUSSELL.DAILY.IP': 0.3,
          'CS.D.USCGC.TODAY.IP': 0.4, 'CS.D.EURUSD.TODAY.IP': 0.6, 'CS.D.GBPUSD.TODAY.IP': 0.9, 'CS.D.BITCOIN.TODAY.IP': 32.7,
          'CC.D.CL.USS.IP': 3.0, 'EN.D.CL.Month1.IP': 6.0, 'CC.D.DX.USS.IP': 8.1, 'CO.D.DX.Month1.IP': 8.1}

_IND = {}
def ind(epic):
    if epic in _IND: return _IND[epic]
    mk = MK[epic]; st = get_strategy_for_market(mk)
    rows = []
    for l in open(os.path.join(ARCHIVE, f'{epic}.jsonl')):
        l = l.strip()
        if l:
            try: rows.append(json.loads(l))
            except Exception: pass
    df = pd.DataFrame(rows); df['date'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    df = add_all_indicators(df, {'ema_fast': st.ema_fast, 'ema_medium': st.ema_medium, 'ema_slow': st.ema_slow, 'rsi_period': st.rsi_period})
    A = {c: df[c].to_numpy(dtype=float) for c in ('open', 'high', 'low', 'close', 'atr', 'rsi', 'macd_hist')}
    A['date'] = df['date'].to_numpy(); A['pos'] = {pd.Timestamp(d): i for i, d in enumerate(df['date'])}
    _IND[epic] = A; return A

def sig_index(A, mk, ts):
    """index of the candle that CLOSED at ts (open = floor(ts) - interval); tolerate small gaps."""
    ts = pd.Timestamp(ts); ci = mk.candle_interval
    fl = ts.floor(f'{ci}min'); so = fl - pd.Timedelta(minutes=ci)
    for cand in (so, fl - pd.Timedelta(minutes=2 * ci)):
        if cand in A['pos']: return A['pos'][cand]
    return None

def walk(A, i, direction, stop, lim, rules, entry=None, spread=0.0, macd_n=None, be_trig=0.7, trail_mult=1.5, lock_pct=0.0,
         rsi_ob=70, rsi_os=30, use_macd=True, horizon=None):
    """Walk forward from signal candle i. Entry = next candle open unless given. Returns result dict."""
    o, h, l, c, atr, rsi, mh = (A[k] for k in ('open', 'high', 'low', 'close', 'atr', 'rsi', 'macd_hist'))
    N = len(c)
    if i + 1 >= N: return {'status': 'no-follow'}
    if entry is None: entry = o[i + 1]
    buy = direction == 'BUY'; sgn = 1.0 if buy else -1.0
    stop_lvl = entry - sgn * stop; lim_lvl = entry + sgn * lim
    use_be = rules == 'LIVE'
    if macd_n is None: macd_n = MACD_EXIT_BARS if rules == 'LIVE' else 3
    be = False; mfe = mae = 0.0; out = None
    last = N if horizon is None else min(N, i + 1 + horizon)
    for j in range(i + 1, last):
        hj, lj, cj = h[j], l[j], c[j]
        fav = ((hj - entry) if buy else (entry - lj)) / stop
        adv = ((entry - lj) if buy else (hj - entry)) / stop
        mfe = max(mfe, fav); mae = max(mae, adv)
        n = j - i
        if (buy and lj <= stop_lvl) or ((not buy) and hj >= stop_lvl):
            out = ('stop' if not be else 'trail', (stop_lvl - entry) * sgn / stop, j, n); break
        if (buy and hj >= lim_lvl) or ((not buy) and lj <= lim_lvl):
            out = ('limit', lim / stop, j, n); break
        r = rsi[j]
        if r == r and ((buy and r > rsi_ob) or ((not buy) and r < rsi_os)):
            out = ('rsi', (cj - entry) * sgn / stop, j, n); break
        if use_macd and n >= 2 and j >= macd_n - 1:
            win = mh[j - macd_n + 1: j + 1]
            if not np.isnan(win).any() and ((buy and (win < 0).all()) or ((not buy) and (win > 0).all())):
                out = ('macd', (cj - entry) * sgn / stop, j, n); break
        if use_be and not be and fav >= be_trig:
            be = True; stop_lvl = entry + sgn * lock_pct * stop
        if be:
            tr = atr[j] * trail_mult
            cand = (hj - tr) if buy else (lj + tr)
            if (buy and cand > stop_lvl and cand - stop_lvl >= 0.2 * tr) or ((not buy) and cand < stop_lvl and stop_lvl - cand >= 0.2 * tr):
                stop_lvl = cand
    cost = spread / stop
    if out is None:
        if horizon is not None and last == i + 1 + horizon:   # resolver horizon: close at market
            j = last - 1; rg = (c[j] - entry) * sgn / stop
            return {'status': 'closed', 'exit': 'horizon', 'r_gross': rg, 'r_net': rg - cost, 'j': j, 'n': j - i,
                    'exit_time': A['date'][j], 'mfe': mfe, 'mae': mae, 'entry': entry, 'stop': stop}
        mtm = (c[N - 1] - entry) * sgn / stop
        return {'status': 'open', 'exit': 'open', 'r_gross': mtm, 'r_net': mtm - cost, 'j': N - 1, 'n': N - 1 - i,
                'exit_time': A['date'][N - 1], 'mfe': mfe, 'mae': mae, 'entry': entry, 'stop': stop}
    why, rg, j, n = out
    return {'status': 'closed', 'exit': why, 'r_gross': rg, 'r_net': rg - cost, 'j': j, 'n': n, 'exit_time': A['date'][j],
            'mfe': mfe, 'mae': mae, 'entry': entry, 'stop': stop}

def live_stop(A, i, mk, st):
    atr = A['atr'][i]; close = A['close'][i]
    if not (atr == atr): return None
    stop = max(atr * st.stop_atr_mult, mk.min_stop_distance)
    cap = max(mk.min_stop_distance * 20, close * 0.05)
    return min(stop, cap)

def family(reason):
    r = reason
    if r.startswith('Breakout'): return 'BREAKOUT(skip)'
    if r.startswith('Regime BEARISH') or r.startswith('Regime BULLISH'): return 'Regime-direction'
    if r.startswith('Regime NEUTRAL'): return 'Regime-NEUTRAL'
    if r.startswith('Regime RANGING'): return 'Regime-RANGING'
    if r.startswith('Direction-restricted'): return 'Direction-restricted'
    if r.startswith('Outside hours'): return 'Outside-hours'
    if r.startswith('Confidence'): return 'Confidence'
    if r.startswith('Shadow-only'): return 'Shadow-only'
    if r.startswith('Screener-inactive'): return 'Screener-inactive'
    if r.startswith('Pullback-entry-expired'): return 'Pullback-expired'
    if r.startswith('Already'): return 'Already-in-position'
    if r.startswith('Re-entry cooldown'): return 'Reentry-cooldown'
    if r.startswith('Loss cooldown'): return 'Loss-cooldown'
    if r.startswith('Startup cooldown'): return 'Startup-cooldown'
    if r.startswith('Position sizing'): return 'Position-sizing'
    if 'would-block' in r: return 'Observational(traded)'
    return 'Other:' + r.split(':')[0][:30]

def utc_hour(ts):  # container-local BST for the whole archive window (>= 2026-06-12)
    return (pd.Timestamp(ts) - pd.Timedelta(hours=1)).hour

def in_hours(mk, ts):
    h = utc_hour(ts); a, b = mk.trading_start, mk.trading_end
    return (a <= h < b) if a < b else (h >= a or h < b)

def other_gates(row, mk, st, fam):
    g = []
    if fam != 'Outside-hours' and not in_hours(mk, row['timestamp']): g.append('hours')
    if fam != 'Direction-restricted' and mk.allowed_direction and mk.allowed_direction != row['direction']: g.append('direction')
    if fam != 'Confidence' and row['confidence'] is not None and st.min_confidence and row['confidence'] < st.min_confidence - 1e-9: g.append('confidence')
    if fam != 'Shadow-only' and getattr(mk, 'shadow_only', False): g.append('shadow')
    if getattr(mk, 'default_mode', '') and 'breakout' in str(getattr(mk, 'default_mode', '')): g.append('mode-breakout')
    return g

MODE_OVERRIDE = {'CC.D.CL.USS.IP': 'breakout', 'CS.D.USCGC.TODAY.IP': 'breakout', 'CC.D.DX.USS.IP': 'breakout-shadow'}

def main():
    D = json.load(open(JOURNAL))
    rej = D['rejected_signals']; bench = D['benched_outcomes']; trades = D['trades']
    results = []
    skipped = collections.Counter()
    for row in rej:
        fam = family(row['reject_reason'])
        if fam.startswith('BREAKOUT') or fam.startswith('Observational'): skipped[fam] += 1; continue
        epic = row['epic']
        if epic not in MK: skipped['epic-not-configured'] += 1; continue
        mk = MK[epic]; st = get_strategy_for_market(mk)
        ts = pd.Timestamp(row['timestamp'])
        if ts < pd.Timestamp('2026-06-13'): skipped['pre-archive'] += 1; continue
        A = ind(epic); i = sig_index(A, mk, ts)
        if i is None: skipped['no-signal-candle'] += 1; continue
        stop = live_stop(A, i, mk, st)
        if stop is None: skipped['no-atr'] += 1; continue
        og = other_gates(row, mk, st, fam)
        if epic in MODE_OVERRIDE and 'mode-breakout' not in og: og.append('mode-breakout')
        base = dict(id=row['id'], epic=epic, market=row['market_name'], direction=row['direction'], ts=ts, conf=row['confidence'],
                    family=fam, reason=row['reject_reason'][:60], other_gates='+'.join(og), placeable=(len(og) == 0))
        for rules in ('LIVE', 'RESOLVER'):
            res = walk(A, i, row['direction'], stop, stop * st.reward_risk, rules, spread=SPREAD.get(epic, 0.0),
                       be_trig=st.breakeven_trigger_pct, trail_mult=st.atr_trail_mult, lock_pct=st.breakeven_lock_pct, rsi_ob=st.rsi_overbought, rsi_os=st.rsi_oversold,
                       use_macd=st.use_macd_exit, horizon=None if rules == 'LIVE' else 96)
            if res.get('status') == 'no-follow': skipped['no-follow'] += 1; continue
            results.append({**base, 'rules': rules, **{k: res[k] for k in ('status', 'exit', 'r_gross', 'r_net', 'n', 'exit_time', 'mfe', 'mae', 'entry', 'stop')}})
    R = pd.DataFrame(results)
    R.to_csv(os.path.join(OUT, 'replay_rejected_results.csv'), index=False)
    print('skipped:', dict(skipped)); print('replayed rows:', R.id.nunique())

    # ---- sequential (non-overlapping) flag per family x epic x rules
    R['seq'] = False
    for (fam, epic, rules), g in R.sort_values('ts').groupby(['family', 'epic', 'rules']):
        last_exit = None
        for idx, r in g.iterrows():
            if last_exit is None or r['ts'] >= last_exit + COOLDOWN:
                R.at[idx, 'seq'] = True; last_exit = pd.Timestamp(r['exit_time'])
    R.to_csv(os.path.join(OUT, 'replay_rejected_results.csv'), index=False)

    def table(df, by, title):
        print(f'\n### {title}')
        print(f'{"group":34s} {"n":>4s} {"cls":>4s} {"W":>3s} {"L":>3s} {"hit%":>5s} {"sumR":>7s} {"meanR":>6s} {"±SE":>5s} {"£":>6s}  exits')
        for k, g in df.groupby(by):
            cl = g[g.status == 'closed']; n = len(g); nc = len(cl)
            if nc == 0: print(f'{str(k):34s} {n:4d} {0:4d}'); continue
            w = (cl.r_net > 0.05).sum(); l = (cl.r_net < -0.05).sum()
            sr = cl.r_net.sum(); mr = cl.r_net.mean(); se = cl.r_net.std(ddof=1) / math.sqrt(nc) if nc > 1 else float('nan')
            ex = ' '.join(f'{a}:{b}' for a, b in cl.exit.value_counts().items())
            print(f'{str(k):34s} {n:4d} {nc:4d} {w:3d} {l:3d} {100*w/max(1,w+l):5.0f} {sr:+7.2f} {mr:+6.2f} {se:5.2f} {sr*GBP_PER_R:+6.0f}  {ex}')

    for rules in ('LIVE', 'RESOLVER'):
        X = R[R.rules == rules]
        table(X, 'family', f'[{rules}] ALL rows by blocking gate (independent signals)')
        table(X[X.seq], 'family', f'[{rules}] by gate, SEQUENTIAL (non-overlapping, 30-min cooldown)')
        table(X[X.placeable & X.seq], 'family', f'[{rules}] by gate, sequential AND placeable under current config (no other gate, market live)')
    X = R[(R.rules == 'LIVE') & R.seq]
    for fam in ('Regime-direction', 'Regime-NEUTRAL', 'Direction-restricted', 'Outside-hours', 'Confidence', 'Shadow-only', 'Screener-inactive'):
        table(X[X.family == fam], 'market', f'[LIVE seq] {fam} by market')
    table(X[X.family == 'Regime-direction'], ['market', 'direction'], '[LIVE seq] Regime-direction by market x direction')
    table(X[X.family == 'Direction-restricted'], ['market', 'direction'], '[LIVE seq] Direction-restricted by market x direction')
    table(X[X.family == 'Outside-hours'], ['market', 'direction'], '[LIVE seq] Outside-hours by market x direction')
    # other-gate overlap for the regime family
    print('\n### Regime-direction rows: what ELSE would block them today')
    print(R[(R.rules == 'LIVE') & (R.family == 'Regime-direction')].groupby(['market', 'other_gates']).size().to_string())

    # ---- CALIBRATION A: resolved shadow/quality benches vs RESOLVER replay
    print('\n### CALIBRATION A — resolved momentum benches (bench_type shadow/quality) vs RESOLVER rules')
    agree = 0; tot = 0; diffs = []
    for b in bench:
        if b['bench_type'] not in ('shadow', 'quality') or b['outcome'] is None or b['epic'] not in MK: continue
        mk = MK[b['epic']]; st = get_strategy_for_market(mk); A = ind(b['epic']); ts = pd.Timestamp(b['benched_at'])
        if ts < pd.Timestamp('2026-06-13'): continue
        i = sig_index(A, mk, ts)
        if i is None: continue
        res = walk(A, i, b['direction'], b['stop_distance'], b['limit_distance'], 'RESOLVER', entry=b['entry_price'], spread=0.0,
                   rsi_ob=st.rsi_overbought, rsi_os=st.rsi_oversold, use_macd=st.use_macd_exit, horizon=96)
        if res.get('status') == 'no-follow': continue
        tot += 1; same = (res['exit'] == b['outcome']) and abs(res['r_gross'] - b['r_multiple']) < 0.15
        agree += same
        diffs.append((b['market_name'], b['direction'], b['benched_at'][:16], b['outcome'], round(b['r_multiple'], 2), res['exit'], round(res['r_gross'], 2), res['n'], b['candles_to_resolve'], 'OK' if same else 'DIFF'))
    print(f'benches compared: {tot}, exit+R agree: {agree} ({100*agree/max(1,tot):.0f}%)')
    for d in diffs: print('  ', d)

    # ---- CALIBRATION B: real closed momentum trades vs LIVE replay (their stop, their entry)
    print('\n### CALIBRATION B — real closed momentum trades vs LIVE rules (given entry+stop; MACD-3 before 09-01)')
    rows = []
    for t in trades:
        if t['status'] != 'CLOSED' or t['strategy'] == 'breakout' or t['epic'] not in MK or not t['exit_time']: continue
        et = pd.Timestamp(t['entry_time'])
        if et < pd.Timestamp('2026-06-13'): continue
        mk = MK[t['epic']]; st = get_strategy_for_market(mk); A = ind(t['epic'])
        i = sig_index(A, mk, et)
        if i is None: continue
        macd_n = 3 if et < MACD_CHANGE else 5
        res = walk(A, i, t['direction'], t['stop_distance'], t['limit_distance'] or t['stop_distance'] * st.reward_risk, 'LIVE',
                   entry=t['entry_price'], spread=0.0, macd_n=macd_n, be_trig=st.breakeven_trigger_pct, trail_mult=st.atr_trail_mult, lock_pct=st.breakeven_lock_pct,
                   rsi_ob=st.rsi_overbought, rsi_os=st.rsi_oversold, use_macd=st.use_macd_exit)
        if res.get('status') != 'closed': continue
        real_r = (t['exit_price'] - t['entry_price']) * (1 if t['direction'] == 'BUY' else -1) / t['stop_distance']
        ex = t['exit_reason'] or ''
        real_exit = 'macd' if 'MACD' in ex else 'rsi' if 'RSI' in ex else 'stop/limit' if 'Stop' in ex else ex[:12]
        my_exit = 'stop/limit' if res['exit'] in ('stop', 'limit', 'trail') else res['exit']
        rows.append(dict(market=t['market_name'], dir=t['direction'], entry=t['entry_time'][:16], real_exit=real_exit, real_r=round(real_r, 2),
                         my_exit=my_exit, my_r=round(res['r_gross'], 2), d=round(res['r_gross'] - real_r, 2), pnl=t['pnl']))
    B = pd.DataFrame(rows)
    if len(B):
        B.to_csv(os.path.join(OUT, 'replay_rejected_calib_trades.csv'), index=False)
        print(f'trades compared: {len(B)} | exit-type agree: {(B.real_exit == B.my_exit).mean()*100:.0f}% | '
              f'sum real R {B.real_r.sum():+.2f} vs replay {B.my_r.sum():+.2f} | mean |dR| {B.d.abs().mean():.2f} | median |dR| {B.d.abs().median():.2f}')
        print('by real exit type:'); print(B.groupby('real_exit').agg(n=('d', 'size'), real=('real_r', 'sum'), mine=('my_r', 'sum'), mad=('d', lambda x: x.abs().mean())).round(2).to_string())
        print('worst 8 disagreements:'); print(B.reindex(B.d.abs().sort_values(ascending=False).index).head(8).to_string(index=False))

if __name__ == '__main__':
    main()
