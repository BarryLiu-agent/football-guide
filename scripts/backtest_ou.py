# -*- coding: utf-8 -*-
"""
backtest_ou.py - 大小球/让球模型校准回测
验证 Elo+泊松 模型的盘口概率是否可信（无历史盘口，校准模型自身）：
  - 模型大球概率(>2.5球) 分组 vs 实际大球率
  - 模型让球赢盘率(-0.5/-1.5 固定线) 分组 vs 实际赢盘率
若校准良好，则「模型 vs 当前盘口」的 edge 有依据；若偏乐观/悲观，edge 需打折。

用法:
  python scripts/backtest_ou.py
输出: data/calibration_ou.json
"""
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from elo import EloModel  # noqa: E402
from predict import ScoreModel  # noqa: E402  # 复用泊松比分模型

DATA_DIR = ROOT / "data"


def main():
    path = DATA_DIR / "season_2025.json"
    if not path.exists():
        print("缺少 data/season_2025.json")
        return 1

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    per_league = {}
    for m in data.get("matches", []):
        if m.get("homeGoals") is None or m.get("awayGoals") is None:
            continue
        per_league.setdefault(m["league"], []).append(m)

    ou_rows = []   # {prob, hit} 大球(>2.5)
    ah_rows = []   # {prob, hit} 主队 -0.5 赢盘
    ah2_rows = []  # {prob, hit} 主队 -1.5 赢盘

    for league, ms in per_league.items():
        elo = EloModel()
        for m in reversed(ms):  # 时间正序（头部=最近）
            home, away = m["homeTeam"], m["awayTeam"]
            ph, pd, pa = elo.predict(home, away)
            sm = ScoreModel()
            sm.fit(ph, pd, pa)
            hg, ag = m["homeGoals"], m["awayGoals"]
            total = hg + ag
            diff = hg - ag

            # 大球概率 >2.5
            dist = sm.total_goals_dist()
            over_p = sum(v for k, v in dist.items() if int(k) > 2.5)
            ou_rows.append({"prob": round(min(over_p, 0.999), 3), "hit": total > 2.5})

            # 主队 -0.5 赢盘（净胜 ≥1）与 -1.5 赢盘（净胜 ≥2）
            ah_rows.append({"prob": round(sm.cover_prob(-0.5), 3), "hit": diff >= 1})
            ah2_rows.append({"prob": round(sm.cover_prob(-1.5), 3), "hit": diff >= 2})

            elo.update([{
                "utcDate": m.get("utcDate", ""),
                "homeTeam": {"name": home},
                "awayTeam": {"name": away},
                "score": {"fullTime": {"home": hg, "away": ag}},
            }])

    def buckets(rows, labels):
        out = []
        for lo, hi, label in labels:
            rs = [r for r in rows if lo <= r["prob"] < hi]
            if not rs:
                continue
            hit = sum(1 for r in rs if r["hit"])
            avg = sum(r["prob"] for r in rs) / len(rs)
            out.append({
                "label": label, "n": len(rs),
                "hitRate": round(hit / len(rs), 4),
                "avgProb": round(avg, 3),
                "bias": round(hit / len(rs) - avg, 4),
            })
        return out

    labels = [(0.0, 0.4, "≤40%"), (0.4, 0.5, "40-50%"), (0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 1.01, "≥70%")]

    out = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "2025/26 赛季 Elo+泊松 盘口模型校准",
        "total": len(ou_rows),
        "ou25": {
            "overall": round(sum(1 for r in ou_rows if r["hit"]) / len(ou_rows), 4),
            "buckets": buckets(ou_rows, labels),
        },
        "ah05": {
            "overall": round(sum(1 for r in ah_rows if r["hit"]) / len(ah_rows), 4),
            "buckets": buckets(ah_rows, labels),
        },
        "ah15": {
            "overall": round(sum(1 for r in ah2_rows if r["hit"]) / len(ah2_rows), 4),
            "buckets": buckets(ah2_rows, labels),
        },
    }

    with open(DATA_DIR / "calibration_ou.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
