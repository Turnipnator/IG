"""
Streaming entitlement probe.

REST-logs in, then attempts several DIFFERENT Lightstreamer subscriptions to
determine whether the [21] Invalid group failure is:
  - market-data specific (only L1:* items fail), or
  - account-wide (ACCOUNT:/TRADE: streams also fail -> entitlement problem).

Read-only: subscribes, captures the result, unsubscribes, logs out. Trades nothing.
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

if ACC_TYPE == "DEMO":
    REST_BASE = "https://demo-api.ig.com/gateway/deal"
    LS_URL = "https://demo-apd.marketdatasystems.com"
else:
    REST_BASE = "https://api.ig.com/gateway/deal"
    LS_URL = "https://apd.marketdatasystems.com"

# ---------- REST login ----------
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
        ls_endpoint = body.get("lightstreamerEndpoint", LS_URL)
except urllib.error.HTTPError as e:
    print(f"REST login FAIL: HTTP {e.code} -- {e.read().decode()}")
    sys.exit(1)

print(f"REST login OK: account={account_id}  ls_endpoint={ls_endpoint}")
print()

from lightstreamer.client import LightstreamerClient, Subscription, SubscriptionListener  # type: ignore

client = LightstreamerClient(ls_endpoint, "DEFAULT")
client.connectionDetails.setUser(account_id)
client.connectionDetails.setPassword(f"CST-{cst}|XST-{xst}")
client.connect()
time.sleep(2)
print(f"LS connection status: {client.getStatus()}")
print()

tests = [
    ("MERGE",    ["L1:CS.D.EURUSD.TODAY.IP"],  ["BID", "OFFER"],            "EUR/USD  L1 market"),
    ("MERGE",    ["L1:CS.D.GBPUSD.TODAY.IP"],  ["BID", "OFFER"],            "GBP/USD  L1 market"),
    ("MERGE",    ["L1:IX.D.FTSE.DAILY.IP"],    ["BID", "OFFER"],            "FTSE100  L1 market"),
    ("MERGE",    [f"ACCOUNT:{account_id}"],    ["PNL", "AVAILABLE_CASH"],   "ACCOUNT  balance stream"),
    ("DISTINCT", [f"TRADE:{account_id}"],      ["CONFIRMS", "OPU"],         "TRADE    confirms stream"),
]


def probe(mode, items, fields, label):
    st = {"r": None, "code": None, "msg": None}
    ev = threading.Event()

    class _L(SubscriptionListener):
        def onSubscription(self):
            st["r"] = "ACTIVE"
            ev.set()

        def onSubscriptionError(self, code, message):
            st["r"], st["code"], st["msg"] = "ERROR", code, message
            ev.set()

    sub = Subscription(mode=mode, items=items, fields=fields)
    sub.addListener(_L())
    client.subscribe(sub)
    ev.wait(5)
    try:
        client.unsubscribe(sub)
    except Exception:
        pass
    if st["r"] == "ACTIVE":
        return f"  {label:28s} -> ACTIVE"
    if st["r"] == "ERROR":
        return f"  {label:28s} -> ERROR [{st['code']}] {st['msg']}"
    return f"  {label:28s} -> TIMEOUT (no response in 5s)"


print("=== SUBSCRIPTION PROBES ===")
for t in tests:
    print(probe(*t))

# Cleanup
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
