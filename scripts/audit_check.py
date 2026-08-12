"""一页式全量审计：predictions / standings / fixtures / odds / xg / asian / backtest / signals"""
import json, sys
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent / "data"
issues = []

# ── 1. predictions.json ──
pj = json.loads((ROOT / "predictions.json").read_text(encoding="utf-8"))
preds = pj.get("predictions", [])
print(f"predictions: {len(preds)}  generatedAt={pj['generatedAt']}")

for p in preds:
    h, a = p["homeTeam"], p["awayTeam"]
    comp = p.get("competition", {})
    l = comp if isinstance(comp, str) else comp.get("code", "?")

    probs = p.get("probabilities", {})
    s = sum(probs.values()) if probs else 0
    if s and abs(s - 1.0) > 0.01:
        issues.append(f"[{l}] {h}vs{a}: prob sum={s:.4f}")

    dc = p.get("dcProb", {})
    dcs = sum(dc.values()) if dc else 0
    if dcs and abs(dcs - 1.0) > 0.01:
        issues.append(f"[{l}] {h}vs{a}: dcProb sum={dcs:.4f}")

    ps = p.get("predictedScore", "")
    cs = p.get("correctScores", [])
    if cs and cs[0] and ps:
        if ps.replace("-", "") != cs[0].get("score", "").replace("-", ""):
            issues.append(f"[{l}] {h}vs{a}: predScore={ps} vs correctScores[0]={cs[0]['score']}")

    odds = p.get("odds", {})
    for k in ("home", "draw", "away"):
        v = odds.get(k)
        if v is not None and v <= 1.0:
            issues.append(f"[{l}] {h}vs{a}: odds.{k}={v}")

    if not p.get("analysis"):
        issues.append(f"[{l}] {h}vs{a}: no analysis")
    if not p.get("status"):
        issues.append(f"[{l}] {h}vs{a}: no status")

    asian = p.get("asian", {})
    if not asian or asian.get("point") is None:
        issues.append(f"[{l}] {h}vs{a}: no asian")

    ou = p.get("overUnder", {})
    if not ou or ou.get("line") is None:
        issues.append(f"[{l}] {h}vs{a}: no overUnder")

    ai = p.get("aiJudge", {})
    if ai and ai.get("confidence") is not None:
        model_c = max(probs.get("home", 0), dc.get("home", 0))
        ac = ai["confidence"]
        if abs(ac - model_c) > 0.155:
            issues.append(f"[{l}] {h}vs{a}: AI conf={ac:.3f} vs model={model_c:.4f} diff={ac-model_c:+.4f}")

# prevH2h
missing_h2h = sum(1 for p in preds if p.get("prevH2h") is None)
if missing_h2h:
    issues.append(f"prevH2h: {missing_h2h}/58 is None")

print(f"  pred internal issues: {len(issues)}")

# ── 2. standings ──
st = json.loads((ROOT / "standings.json").read_text(encoding="utf-8"))
standings = st.get("standings", st) if isinstance(st, dict) else st
for league, data in standings.items():
    played = sum(1 for r in data if r.get("played", 0) > 0)
    placeholder = sum(1 for r in data if r.get("played", 0) == 0)
    print(f"standings [{league}]: {played} real / {placeholder} placeholder")
    for r in data:
        if r.get("played", 0) > 0:
            pos = r.get("position", 0)
            if pos < 1 or pos > 20:
                issues.append(f"[{league}] {r['team']}: position={pos}")

# ── 3. fixtures ──
fx = json.loads((ROOT / "fixtures.json").read_text(encoding="utf-8"))
matches = fx.get("matches", [])
future = [m for m in matches if m.get("status") in ("TIMED", "SCHEDULED")]
finished = [m for m in matches if m.get("status") == "FINISHED"]
print(f"fixtures: total={len(matches)} future={len(future)} finished={len(finished)}")

now = datetime.utcnow()
for m in future:
    dt = m.get("utcDate", "")
    if dt:
        try:
            md = datetime.fromisoformat(dt.replace("Z", ""))
            if md < now:
                issues.append(f"[{m['competition']['code']}] {m['homeTeam']['shortName']}vs{m['awayTeam']['shortName']}: past but TIMED ({dt})")
        except:
            pass

# ── 4. odds files ──
for fname in ["PL", "PD", "BL1", "SA", "FL1", "CL"]:
    fp = ROOT / "odds" / f"{fname}.json"
    if fp.exists():
        od = json.loads(fp.read_text(encoding="utf-8"))
        print(f"odds/{fname}.json: {len(od.get('matches',[]))} matches")
    else:
        issues.append(f"MISSING: odds/{fname}.json")

# ── 5. xG ──
for fname in ["PL", "PD", "BL1", "SA", "FL1", "CL"]:
    fp = ROOT / "xg" / f"{fname}.json"
    if fp.exists():
        xg = json.loads(fp.read_text(encoding="utf-8"))
        print(f"xg/{fname}.json: teams={xg.get('teams',0)} players={xg.get('players',0)}")
    else:
        issues.append(f"MISSING: xg/{fname}.json")

# ── 6. asian ──
for fp in sorted((ROOT / "asian").glob("*.json")):
    ah = json.loads(fp.read_text(encoding="utf-8"))
    n = len(ah) if isinstance(ah, list) else len(ah.get("matches", ah))
    print(f"asian/{fp.name}: {n} records")

# ── 7. backtest ──
bf = ROOT / "backtest_results.json"
if bf.exists():
    bt = json.loads(bf.read_text(encoding="utf-8"))
    s_ = bt.get("summary", {})
    print(f"backtest: {s_.get('total_matches','?')} matches, acc={s_.get('accuracy',0):.3f}, roi={s_.get('roi',0):.3f}")
else:
    issues.append("MISSING: backtest_results.json")

# ── 8. signals ──
mf = ROOT / "message_signals.json"
if mf.exists():
    ms = json.loads(mf.read_text(encoding="utf-8"))
    print(f"message_signals: {len(ms)} entries")
else:
    issues.append("MISSING: message_signals.json")

# ── 9. advanced ──
for fp in sorted((ROOT / "advanced").glob("*.json")):
    adv = json.loads(fp.read_text(encoding="utf-8"))
    n_teams = len(adv) if isinstance(adv, list) else len(adv.get("teams", adv))
    print(f"advanced/{fp.name}: {n_teams} teams")

# ── 10. prediction_history ──
phf = ROOT / "prediction_history.json"
if phf.exists():
    ph = json.loads(phf.read_text(encoding="utf-8"))
    snapshots = len(ph) if isinstance(ph, list) else len(ph.get("snapshots", ph))
    print(f"prediction_history: {snapshots} snapshots")
else:
    issues.append("MISSING: prediction_history.json")

# ── FINAL ──
print(f"\n===== TOTAL ISSUES: {len(issues)} =====")
for i in issues:
    print(f"  X {i}")
