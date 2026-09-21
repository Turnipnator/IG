"""Which EPIC is killing the streaming subscription?

The bot subscribes to all 13 markets as ONE Lightstreamer subscription, and IG
rejects the whole thing if a single item in it is unsubscribable. This repo has
been bitten by that three times now:

  * CC.D.* CFD-only epics on a SPREADBET account -> "Invalid account type"
  * SI.D.ITBUS.DAILY.IP                          -> "[-1] Incorrect instrument setup"
  * 2026-09-18 19:31, all 13 at once             -> "[21] Invalid group"

Each time the symptom was total: every market went dark, and the group looked
like the culprit because the group is what the error names. This script tells
the two apart by subscribing to each epic ALONE and reporting which ones bind.

If every epic fails, it really is the group/entitlement. If one fails and twelve
bind, that one epic was taking the whole feed down with it.

Read-only: subscribes, waits for a verdict, unsubscribes, logs out. No orders,
no historical fetches (streaming costs nothing against the 10k/week allowance).

    docker exec -i -w /app ig-trading-bot python3 - < scripts/stream_epic_bisect.py
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/app" if os.path.exists("/app") else ".")
from config import MARKETS  # noqa: E402

API_KEY = os.environ["IG_API_KEY"]
USERNAME = os.environ["IG_USERNAME"]
PASSWORD = os.environ["IG_PASSWORD"]
ACC_TYPE = os.environ.get("IG_ACC_TYPE", "DEMO").upper()
REST_BASE = (
    "https://demo-api.ig.com/gateway/deal" if ACC_TYPE == "DEMO"
    else "https://api.ig.com/gateway/deal"
)

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
        ls_endpoint = body.get("lightstreamerEndpoint")
except urllib.error.HTTPError as e:
    print(f"REST login FAILED: HTTP {e.code} — {e.read().decode()}")
    sys.exit(1)

print(f"REST login OK — account={account_id} type={ACC_TYPE}")
print(f"Lightstreamer endpoint: {ls_endpoint}\n")

from lightstreamer.client import (  # noqa: E402
    LightstreamerClient, Subscription, SubscriptionListener,
)

client = LightstreamerClient(ls_endpoint, "DEFAULT")
client.connectionDetails.setUser(account_id)
client.connectionDetails.setPassword(f"CST-{cst}|XST-{xst}")
client.connect()
time.sleep(2)
print(f"LS status: {client.getStatus()}\n")

FIELDS = ["UPDATE_TIME", "BID", "OFFER", "CHANGE", "CHANGE_PCT", "HIGH", "LOW", "MARKET_STATE"]


def probe(group: str, epics: list[str], label: str, wait: float = 6.0) -> tuple[str, str]:
    """Subscribe to one item set, wait for IG's verdict, tear it down."""
    done = threading.Event()
    state = {"result": "TIMEOUT", "detail": "", "ticked": False}

    class _L(SubscriptionListener):
        def onSubscription(self):
            state["result"] = "ACTIVE"
            done.set()

        def onSubscriptionError(self, code, message):
            state["result"] = "REFUSED"
            state["detail"] = f"[{code}] {message}"
            done.set()

        def onItemUpdate(self, update):
            state["ticked"] = True

    sub = Subscription(mode="MERGE", items=[f"{group}:{e}" for e in epics], fields=FIELDS)
    sub.addListener(_L())
    client.subscribe(sub)
    done.wait(wait)
    if state["result"] == "ACTIVE":
        time.sleep(1.5)  # give a tick a chance to land
    try:
        client.unsubscribe(sub)
    except Exception:
        pass
    time.sleep(0.3)
    tick = " +tick" if state["ticked"] else ""
    return state["result"], (state["detail"] or "") + tick


epics = [m.epic for m in MARKETS]
names = {m.epic: m.name for m in MARKETS}

print("=" * 72)
print("STEP 1 — all 13 at once (reproduce the bot's own subscription)")
print("=" * 72)
for group in ("MARKET", "L1"):
    result, detail = probe(group, epics, group)
    print(f"  {group:8s} all {len(epics)} epics -> {result} {detail}")

print()
print("=" * 72)
print("STEP 2 — each epic ALONE via MARKET (finds a single poisoning item)")
print("=" * 72)
good, bad = [], []
for epic in epics:
    result, detail = probe("MARKET", [epic], epic, wait=5.0)
    flag = "OK " if result == "ACTIVE" else "!! "
    print(f"  {flag}{names.get(epic, ''):20s} {epic:24s} -> {result} {detail}")
    (good if result == "ACTIVE" else bad).append(epic)

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
print(f"  bind OK : {len(good)}/{len(epics)}")
print(f"  refused : {len(bad)}/{len(epics)}")
if bad and good:
    print(f"\n  CULPRIT EPIC(S): {bad}")
    print("  These are taking the whole subscription down. Removing them from")
    print("  MARKETS (or fixing the epic) restores the feed for everything else.")
elif not good:
    print("\n  Every epic refused individually -> genuinely a group/entitlement")
    print("  problem, not a bad epic. Check the API key at labs.ig.com.")
else:
    print("\n  All epics bind individually. If the combined subscribe in STEP 1")
    print("  also bound, the feed is healthy and the fault was transient.")

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
print("\ndone (session closed)")
