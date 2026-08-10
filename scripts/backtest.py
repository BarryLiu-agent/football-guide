# -*- coding: utf-8 -*-
"""
backtest.py - 模型校准回测（Elo 滚动预测上赛季）
对 data/season_2025.json 的每场做"赛前预测 → 赛后更新"，输出分组命中率校准报告。
输出: data/calibration.json（前端战绩页展示）

用法:
  python scripts/backtest.py
"""
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from elo import EloModel  # noqa: E402

DATA_DIR = ROOT / "data"


def main():
    path = DATA_DIR / "season_2025.json"
    if not path.exists():
        print("缺少 data/season_2025.json，先运行 _run_seasons.py 抓取上赛季数据")
        return 1

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    per_league = {}
    for m in data.get("matches", []):
        if m.get("homeGoals") is None or m.get("awayGoals") is None:
            continue
        per_league.setdefault(m["league"], []).append(m)

    rows = []
    for league, ms in per_league.items():
        elo = EloModel()
        # 抓取顺序 = 从最新往回翻 → 反转即时间正序（赛前信息只能来自更早的比赛）
        for m in reversed(ms):
            home, away = m["homeTeam"], m["awayTeam"]
            ph, pd, pa = elo.predict(home, away)
            best = max([("home", ph), ("draw", pd), ("away", pa)], key=lambda x: x[1])
            hg, ag = m["homeGoals"], m["awayGoals"]
            actual = "home" if hg > ag else ("draw" if hg == ag else "away")
            rows.append({
                "league": league, "prob": round(best[1], 4),
                "dir": best[0], "hit": best[0] == actual,
            })
            elo.update([{
                "utcDate": m.get("utcDate", ""),
                "homeTeam": {"name": home},
                "awayTeam": {"name": away},
                "score": {"fullTime": {"home": hg, "away": ag}},
            }])

    if not rows:
        print("无有效赛果数据")
        return 1

    # ── 分组校准（模型最高概率 vs 实际命中率）──
    buckets = [
        (0.0, 0.45, "≤45%"),
        (0.45, 0.55, "45-55%"),
        (0.55, 0.65, "55-65%"),
        (0.65, 1.01, "≥65%"),
    ]
    groups = []
    for lo, hi, label in buckets:
        rs = [r for r in rows if lo <= r["prob"] < hi]
        if not rs:
            continue
        hit = sum(1 for r in rs if r["hit"])
        groups.append({
            "label": label, "n": len(rs),
            "hitRate": round(hit / len(rs), 4),
            "avgProb": round(sum(r["prob"] for r in rs) / len(rs), 3),
            "bias": round(hit / len(rs) - sum(r["prob"] for r in rs) / len(rs), 4),
        })

    by_league = {}
    for lg in sorted(set(r["league"] for r in rows)):
        rs = [r for r in rows if r["league"] == lg]
        by_league[lg] = {
            "n": len(rs),
            "hitRate": round(sum(1 for r in rs if r["hit"]) / len(rs), 4),
        }

    out = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "2025/26 赛季回测（Elo 滚动预测）",
        "total": len(rows),
        "overall": round(sum(1 for r in rows if r["hit"]) / len(rows), 4),
        "buckets": groups,
        "byLeague": by_league,
        "directionDist": dict(Counter(r["dir"] for r in rows)),
    }

    with open(DATA_DIR / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
