import sqlite3
con = sqlite3.connect("/app/data/trade_journal.db")
con.row_factory = sqlite3.Row
for mkt in ("FTSE 100", "NASDAQ 100"):
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM trades WHERE market_name=? AND pnl IS NOT NULL ORDER BY id DESC LIMIT 8",
        (mkt,))]
    print("=" * 78)
    print(f"{mkt} — last {len(rows)} closed trades (newest first)")
    print("=" * 78)
    for t in rows:
        sd = t["stop_distance"] or 0
        # how far price travelled vs the stop (exit move / stop) — signed in trade dir
        ep, xp = t["entry_price"], t["exit_price"] or t["entry_price"]
        move = (xp - ep) if t["direction"] == "BUY" else (ep - xp)
        rmult = (move / sd) if sd else 0
        tag = "WIN " if t["pnl"] > 0.5 else ("LOSS" if t["pnl"] < -0.5 else "scr ")
        print(f"[{tag}] {t['entry_time'][5:16]} {t['direction']:4s} conf={t['confidence']:.2f} "
              f"ADX={t['adx']:.0f} RSI={t['rsi']:.0f} HTF={t['htf_trend']:7s} "
              f"stop={sd:.1f} held={t['duration_mins']:.0f}m  {t['exit_reason'][:22]:22s} "
              f"R={rmult:+.2f} GBP{t['pnl']:+.2f}")
    print()
