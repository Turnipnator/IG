"""
Alternative streaming-group probe.

L1:{epic} currently fails with [21] Invalid group (missing L1 entitlement).
This tests whether OTHER IG Lightstreamer groups work for the same epic, which
would (a) give the bot a possible workaround and (b) localise the entitlement gap.

Groups tested: L1, MARKET, CHART:TICK, CHART:1MINUTE.
Read-only: subscribes, captures result + whether a tick arrives, unsubscribes, logs out.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

API_KEY = os.environ["IG_API_KEY"]
USERNAME = os.environ["IG_USERNAME"]
PASSWORD = os.environ["IG_PASSWORD"]
ACC_TYPE = os.environ.get("IG_ACC_TYPE", "DEMO").upper()
REST_BASE = "https://demo-api.ig.com/gateway/deal" if ACC_TYPE == "DEMO" else "https://api.ig.com/gateway/deal"
LS_FALLBACK = "https://demo-apd.marketdatasystems.com" if ACC_TYPE == "DEMO" else "https://apd.marketdatasystems.com"

req = urllib.request.Request(
    f"{REST_BASE}/session",
    data=json.dumps({"identifier": USERNAME, "password": PASSWORD}).encode(),
    headers={"X-IG-API-KEY": API_KEY, "Content-Type": "application/json", "Version": "2"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        body = json.loads(r.read().decode())
        cst = r.headers.get("CST")
        xst = r.headers.get("X-SECURITY-TOKEN")
        account_id = body.get("currentAccountId")
        ls = body.get("lightstreamerEndpoint", LS_FALLBACK)
except urllib.error.HTTPError as e:
    print(f"REST login FAIL: HTTP {e.code} -- {e.read().decode()}")
    sys.exit(1)
print(f"REST login OK: account={account_id}")
print()

from lightstreamer.client import LightstreamerClient, Subscription, SubscriptionListener  # type: ignore

client = LightstreamerClient(ls, "DEFAULT")
client.connectionDetails.setUser(account_id)
client.connectionDetails.setPassword(f"CST-{cst}|XST-{xst}")
client.connect()
time.sleep(2)
print(f"LS status: {client.getStatus()}")
print()

EPIC = "CS.D.EURUSD.TODAY.IP"
# The exact field list the bot subscribes with (src/streaming.py).
BOT_FIELDS = ["UPDATE_TIME", "BID", "OFFER", "CHANGE", "CHANGE_PCT", "HIGH", "LOW", "MARKET_STATE"]
tests = [
    ("MERGE", [f"L1:{EPIC}"],     BOT_FIELDS, "L1     + bot's exact 8 fields"),
    ("MERGE", [f"MARKET:{EPIC}"], BOT_FIELDS, "MARKET + bot's exact 8 fields"),
]


def probe(mode, items, fields, label):
    st = {"r": None, "code": None, "msg": None, "tick": False}
    ev = threading.Event()

    class _L(SubscriptionListener):
        def onSubscription(self):
            st["r"] = "ACTIVE"
            ev.set()

        def onSubscriptionError(self, code, message):
            st["r"], st["code"], st["msg"] = "ERROR", code, message
            ev.set()

        def onItemUpdate(self, u):
            st["tick"] = True

    sub = Subscription(mode=mode, items=items, fields=fields)
    sub.addListener(_L())
    client.subscribe(sub)
    ev.wait(6)
    if st["r"] == "ACTIVE":
        time.sleep(2)  # wait for a tick
    try:
        client.unsubscribe(sub)
    except Exception:
        pass
    if st["r"] == "ACTIVE":
        return f"  {label:34s} -> ACTIVE  ({'tick received' if st['tick'] else 'no tick in window'})"
    if st["r"] == "ERROR":
        return f"  {label:34s} -> ERROR [{st['code']}] {st['msg']}"
    return f"  {label:34s} -> TIMEOUT"


print(f"=== ALTERNATIVE GROUP PROBES (epic: {EPIC}) ===")
for t in tests:
    print(probe(*t))

try:
    client.disconnect()
except Exception:
    pass
try:
    urllib.request.urlopen(
        urllib.request.Request(
            f"{REST_BASE}/session",
            headers={"X-IG-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": xst,
                     "Version": "1", "_method": "DELETE"},
            method="DELETE",
        ),
        timeout=5,
    )
except Exception:
    pass
