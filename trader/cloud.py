"""Running the bot on GitHub Actions (so your PC can be off).

How it works
  * A GitHub workflow starts before every US session, restores the bot's memory, trades all
    session, does the after-close review + strategy search, saves its memory and stops.
  * GitHub stops any single job after 6 hours, a session plus review is longer, so the bot
    hands over to a fresh job ("continuation") before that, never close to the closing bell.
  * The bot's memory (data/brain.db) is saved ENCRYPTED (with your STATE_KEY secret) on a
    separate 'state' branch, so nobody can read your strategies even though the repo is public.
    A small status file (balances, trades - paper money, no keys) is saved next to it for
    your Projects Dashboard.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from .config import DATA_DIR, ROOT


def on_github():
    return os.environ.get("GITHUB_ACTIONS") == "true"


def status_dir():
    d = os.environ.get("NASDAQ_STATUS_DIR")
    return Path(d) if d else None


def save_state(log=print, reason="checkpoint"):
    """Snapshot brain.db safely (SQLite backup) and let the workflow script encrypt + push it."""
    if not on_github():
        return False
    src = DATA_DIR / "brain.db"
    if not src.exists():
        return False
    snap = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "brain_snapshot.db"
    try:
        s = sqlite3.connect(str(src))
        d = sqlite3.connect(str(snap))
        with d:
            s.backup(d)
        s.close()
        d.close()
        script = ROOT / ".github" / "scripts" / "save_state.sh"
        r = subprocess.run(["bash", str(script), str(snap), reason], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            log(f"  ! could not save memory to GitHub: {(r.stderr or r.stdout)[-300:]}")
            return False
        log(f"  memory saved to GitHub ({reason})")
        return True
    except Exception as e:
        log(f"  ! could not save memory to GitHub: {e}")
        return False


def dispatch_continuation(log=print):
    """Ask GitHub to start the next job of this session (it queues until this one has finished)."""
    if not on_github():
        return False
    import requests
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    wf = os.environ.get("NASDAQ_WORKFLOW_FILE", "trade.yml")
    ref = os.environ.get("GITHUB_REF_NAME", "main")
    if not (repo and token):
        log("  ! cannot start the continuation job (no GITHUB_TOKEN)")
        return False
    for attempt in range(3):
        try:
            r = requests.post(f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/dispatches",
                              headers={"Authorization": f"Bearer {token}",
                                       "Accept": "application/vnd.github+json"},
                              json={"ref": ref, "inputs": {"reason": "continuation"}}, timeout=30)
            if r.status_code in (201, 204):
                log("  handed over to a fresh GitHub job (continuation queued)")
                return True
            log(f"  ! continuation request failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            log(f"  ! continuation request failed: {e}")
        time.sleep(5 * (attempt + 1))
    return False


def import_seed(registry, log=print):
    """First run on GitHub: bring over what the bot learned on your PC (counts of strategies
    tried - so the luck test stays honest - search history and current strategies)."""
    f = ROOT / "state_seed.json"
    if not f.exists() or registry.meta("seed_imported"):
        return False
    seed = json.loads(f.read_text(encoding="utf-8"))
    for k, v in (seed.get("meta") or {}).items():
        registry.set_meta(k, v)
    n = 0
    for s in seed.get("strategies") or []:
        if registry.get(s["id"]):
            continue
        registry.db.execute(
            "INSERT INTO strategies (id,family,parent,version,status,created,paper_start,genome,report,fitness,note,health)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["id"], s["family"], s.get("parent"), s["version"], s["status"], s["created"], s.get("paper_start"),
             json.dumps(s["genome"]), json.dumps(s["report"]), float(s.get("fitness") or 0), s.get("note") or "",
             s.get("health") or "new"))
        n += 1
    registry.db.commit()
    registry.set_meta("seed_imported", True)
    log(f"  imported the bot's memory from your PC: {n} strategies, "
        f"{int(seed.get('meta', {}).get('total_trials', 0)):,} strategies tested so far")
    return True


# ---------------- PC search -> GitHub trading ("inbox") ----------------
# You search for strategies on your PC; GitHub trades them. After a search the PC sends
# the strategies it validated (plus the luck-test counts) as inbox/strategies.enc to the
# repo's main branch. It is encrypted with your STATE_KEY (stdlib only: SHA-256 counter-mode
# keystream + HMAC-SHA256 tag), so the public repo shows nothing readable.
import base64
import hashlib
import hmac
import secrets as _secrets
import zlib

INBOX_PATH = "inbox/strategies.enc"
SEARCH_META = ("total_trials", "families", "sr_stats", "search_runs", "independence", "independence_stats")


def _keys(key_text):
    k = hashlib.sha256(("nasdaq-bot-inbox:" + key_text.strip()).encode()).digest()
    return hashlib.sha256(k + b"enc").digest(), hashlib.sha256(k + b"mac").digest()


def _stream(k, nonce, n):
    out = bytearray()
    c = 0
    while len(out) < n:
        out += hashlib.sha256(k + nonce + c.to_bytes(8, "big")).digest()
        c += 1
    return bytes(out[:n])


def seal(obj, key_text):
    ke, km = _keys(key_text)
    raw = zlib.compress(json.dumps(obj).encode(), 9)
    nonce = _secrets.token_bytes(16)
    ct = bytes(a ^ b for a, b in zip(raw, _stream(ke, nonce, len(raw))))
    tag = hmac.new(km, nonce + ct, hashlib.sha256).digest()
    return base64.b64encode(b"NQ1" + nonce + tag + ct)


def unseal(blob, key_text):
    b = base64.b64decode(blob)
    if b[:3] != b"NQ1":
        raise ValueError("not a strategies file")
    nonce, tag, ct = b[3:19], b[19:51], b[51:]
    ke, km = _keys(key_text)
    if not hmac.compare_digest(tag, hmac.new(km, nonce + ct, hashlib.sha256).digest()):
        raise ValueError("wrong STATE_KEY or damaged file")
    return json.loads(zlib.decompress(bytes(a ^ b for a, b in zip(ct, _stream(ke, nonce, len(ct))))))


def export_inbox(registry):
    rows = registry.db.execute(
        "SELECT * FROM strategies WHERE status IN ('champion','reserve') ORDER BY created").fetchall()
    strategies = [{k: r[k] for k in r.keys()} for r in rows]
    for s in strategies:
        s["genome"] = json.loads(s["genome"]) if isinstance(s["genome"], str) else s["genome"]
        s["report"] = json.loads(s["report"]) if isinstance(s["report"], str) else s["report"]
    from datetime import datetime, timezone
    return {"published": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "meta": {k: registry.meta(k) for k in SEARCH_META if registry.meta(k) is not None},
            "strategies": strategies}


def import_inbox(registry, cfg, payload, log=print):
    """GitHub side: add strategies found on the PC (as reserves - free slots are filled at once).
    Strategies GitHub already knows (e.g. retired by self-healing) are left as GitHub decided."""
    if payload.get("published") and payload["published"] == registry.meta("inbox_published"):
        return 0
    meta = payload.get("meta") or {}
    if float(meta.get("total_trials") or 0) >= float(registry.meta("total_trials", 0) or 0):
        for k, v in meta.items():          # the PC does the searching, so its luck counts are the real ones
            registry.set_meta(k, v)
    n = 0
    for s in payload.get("strategies") or []:
        if registry.get(s["id"]):
            continue
        registry.db.execute(
            "INSERT INTO strategies (id,family,parent,version,status,created,paper_start,genome,report,fitness,note,health)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["id"], s["family"], s.get("parent"), s["version"], "reserve", s["created"], None,
             json.dumps(s["genome"]), json.dumps(s["report"]), float(s.get("fitness") or 0),
             s.get("note") or "", "new"))
        registry.event(s["id"], "received", "Strategy found by the search on your PC.")
        n += 1
    registry.db.commit()
    registry.set_meta("inbox_published", payload.get("published"))
    from .healer import fill_slots
    fill_slots(cfg, registry, log)
    log(f"  strategies from your PC (search of {payload.get('published', '?')[:16]}): {n} new")
    return n


def check_inbox(registry, cfg, log=print):
    """GitHub side: look for a newer strategies file from the PC (main branch)."""
    key = os.environ.get("STATE_KEY", "")
    if not key:
        return 0
    blob = None
    if on_github():
        try:
            subprocess.run(["git", "fetch", "-q", "--depth=1", "origin", os.environ.get("GITHUB_REF_NAME", "main")],
                           cwd=str(ROOT), capture_output=True, timeout=60)
            r = subprocess.run(["git", "show", f"FETCH_HEAD:{INBOX_PATH}"], cwd=str(ROOT),
                               capture_output=True, timeout=30)
            if r.returncode == 0:
                blob = r.stdout
        except Exception:
            blob = None
    if blob is None and (ROOT / INBOX_PATH).exists():
        blob = (ROOT / INBOX_PATH).read_bytes()
    if not blob:
        return 0
    try:
        return import_inbox(registry, cfg, unseal(blob, key), log)
    except Exception as e:
        log(f"  ! could not read the strategies sent from your PC: {e}")
        return 0


def publish(cfg, registry, log=print):
    """PC side: send this PC's validated strategies to GitHub (git must be installed)."""
    repo = ((cfg.get("github") or {}).get("repo") or "").strip()
    keyf = ROOT / "STATE_KEY.txt"
    if not repo:
        log("! no GitHub repo set (config.yaml -> github: repo:)")
        return False
    if not keyf.exists():
        log("! STATE_KEY.txt is missing - it must be the same text as the STATE_KEY secret on GitHub")
        return False
    if not shutil.which("git"):
        log("! Git is not installed - get it from https://git-scm.com/download/win, then run this again")
        return False
    sync = ROOT / "github_sync"

    def git(*a, check=True):
        r = subprocess.run(["git", *a], cwd=str(sync), capture_output=True, text=True, timeout=300)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {a[0]} failed: {(r.stderr or r.stdout).strip()[-400:]}")
        return r

    try:
        if not (sync / ".git").exists():
            log(f"  first time: downloading the GitHub repo {repo} into github_sync ...")
            r = subprocess.run(["git", "clone", "-q", f"https://github.com/{repo}.git", str(sync)],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"git clone failed: {(r.stderr or r.stdout).strip()[-400:]}")
        git("pull", "-q", "--ff-only", "origin", "main")
        payload = export_inbox(registry)
        (sync / "inbox").mkdir(exist_ok=True)
        (sync / INBOX_PATH).write_bytes(seal(payload, keyf.read_text(encoding="utf-8")))
        git("add", INBOX_PATH)
        if not git("status", "--porcelain", check=False).stdout.strip():
            log("  GitHub already has these strategies")
            return True
        git("-c", "user.name=nasdaq-bot-pc", "-c", "user.email=nasdaq-bot@users.noreply.github.com",
            "commit", "-q", "-m", f"strategies from PC search {payload['published'][:16]}")
        git("push", "-q", "origin", "HEAD:main")
        log(f"  sent {len(payload['strategies'])} strategies to GitHub - the next session trades them "
            f"(a running session picks them up within 30 minutes)")
        return True
    except Exception as e:
        log(f"! could not send the strategies to GitHub: {e}")
        return False
