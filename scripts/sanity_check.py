"""
Post-IG-fix sanity check.

Runs two minimal tests against the IG demo account without bringing the bot up:
  1. REST login   -> obtains CST + X-SECURITY-TOKEN
  2. Lightstreamer subscribe to MARKET:CS.D.EURUSD.TODAY.IP (BID, OFFER only)

Exits 0 only if both pass. Logs out cleanly so it doesn't leave a stale session.

Designed to be invoked from the VPS as a one-shot inside the bot's image:
  cd /root/ig-bot && docker compose run --rm --no-deps \
      --entrypoint python3 ig-bot /app/scripts/sanity_check.py
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

print(f"Account type: {ACC_TYPE}")
print(f"Key prefix:   {API_KEY[:8]}...")
print(f"Username:     {USERNAME}")
print()

# ---------- Test 1: REST login ----------
print("=== TEST 1: REST LOGIN ===")
print(f"POST {REST_BASE}/session")

req = urllib.request.Request(
    f"{REST_BASE}/session",
    data=json.dumps({"identifier": USERNAME, "password": PASSWORD}).encode(),
    headers={
        "X-IG-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "Version": "2",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        body = json.loads(r.read().decode())
        cst = r.headers.get("CST")
        xst = r.headers.get("X-SECURITY-TOKEN")
        account_id = body.get("currentAccountId")
        if not (cst and xst and account_id):
            print(f"  FAIL: incomplete response (CST={bool(cst)}, XST={bool(xst)}, acct={account_id})")
            sys.exit(1)
        print(f"  PASS: HTTP 200, account={account_id}, CST/XST received")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"  FAIL: HTTP {e.code} -- {body}")
    print("  REST login still broken. Do NOT restart the bot.")
    sys.exit(1)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    sys.exit(2)

print()

# ---------- Test 2: Lightstreamer subscribe ----------
print("=== TEST 2: LIGHTSTREAMER SUBSCRIBE ===")
print(f"Endpoint:     {LS_URL}")
print(f"Adapter set:  DEFAULT")
print(f"Item:         MARKET:CS.D.EURUSD.TODAY.IP")
print(f"Fields:       BID, OFFER")

try:
    from lightstreamer.client import (  # type: ignore
        LightstreamerClient,
        Subscription,
        SubscriptionListener,
    )
except ImportError:
    print("  FAIL: lightstreamer-client-lib not available. Must run inside bot image.")
    sys.exit(3)

state = {"status": None, "code": None, "msg": None, "tick": False}
event = threading.Event()


class _L(SubscriptionListener):
    def onSubscription(self):
        state["status"] = "active"
        event.set()

    def onSubscriptionError(self, code, message):
        state["status"] = "error"
        state["code"] = code
        state["msg"] = message
        event.set()

    def onItemUpdate(self, update):
        state["tick"] = True


ls_password = f"CST-{cst}|XST-{xst}"
client = LightstreamerClient(LS_URL, "DEFAULT")
client.connectionDetails.setUser(account_id)
client.connectionDetails.setPassword(ls_password)
client.connect()
time.sleep(2)

sub = Subscription(
    mode="MERGE",
    items=["MARKET:CS.D.EURUSD.TODAY.IP"],
    fields=["BID", "OFFER"],
)
sub.addListener(_L())
client.subscribe(sub)

event.wait(6)
if state["status"] == "active":
    time.sleep(1)  # give it a moment to receive a tick

if state["status"] == "active":
    tick_info = "tick received" if state["tick"] else "no tick yet (market may be closed)"
    print(f"  PASS: subscription ACTIVE ({tick_info})")
    rc = 0
elif state["status"] == "error":
    print(f"  FAIL: subscription ERROR [{state['code']}] {state['msg']}")
    print("  Streaming still broken.")
    rc = 4
else:
    print("  FAIL: TIMEOUT (no response from Lightstreamer in 6s)")
    rc = 5

# Clean up
try:
    client.disconnect()
except Exception:
    pass

# Log out so we don't leave stale sessions on IG
try:
    req2 = urllib.request.Request(
        f"{REST_BASE}/session",
        headers={
            "X-IG-API-KEY": API_KEY,
            "CST": cst,
            "X-SECURITY-TOKEN": xst,
            "Version": "1",
            "_method": "DELETE",
        },
        method="DELETE",
    )
    urllib.request.urlopen(req2, timeout=5)
except Exception:
    pass

print()
if rc == 0:
    print("RESULT: All clear. Safe to restart the bot.")
else:
    print(f"RESULT: One or more checks failed (rc={rc}). Keep the bot stopped.")
sys.exit(rc)
