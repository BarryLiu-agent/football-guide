"""Quick predictions-only check with issue printing"""
import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent / "data"
issues = []

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
            issues.append(f"[{l}] {h}vs{a}: predScore={ps} vs top1={cs[0]['score']}")

    odds = p.get("odds", {})
    for k in ("home", "draw", "away"):
        v = odds.get(k)
        if v is not None and v <= 1.0:
            issues.append(f"[{l}] {h}vs{a}: odds.{k}={v}")

    if not p.get("analysis"):
        issues.append(f"[{l}] {h}vs{a}: no analysis")
    if not p.get("status"):
        issues.append(f"[{l}] {h}vs{a}: no status")

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

# signalLevel vs valuePicks
for p in preds:
    sl = p.get("signalLevel", "")
    vp = p.get("valuePicks", [])
    if sl == "gold" and (not vp or len(vp) == 0):
        issues.append(f"[{p.get('competition','?')}] {p['homeTeam']}vs{p['awayTeam']}: signalLevel=gold but no valuePicks")

print(f"\nTOTAL ISSUES: {len(issues)}")
for i in issues[:50]:
    print(f"  X {i}")
if len(issues) > 50:
    print(f"  ... and {len(issues)-50} more")
