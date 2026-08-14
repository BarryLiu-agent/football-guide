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
    # 多赛季支持：读取 season_2021.json ~ season_2025.json
    seasons = []
    for year in range(2021, 2026):
        path = DATA_DIR / f"season_{year}.json"
        if path.exists():
            seasons.append((year, path))
    
    if not seasons:
        # 回退：只读 season_2025.json
        path = DATA_DIR / "season_2025.json"
        if path.exists():
            seasons = [(2025, path)]
        else:
            print("缺少赛季数据，先运行 fbref_seasons.py 抓取多赛季")
            return 1

    print(f'回测 {len(seasons)} 个赛季: {[y for y,_ in seasons]}')

    # 每联赛维护一个跨赛季 Elo
    per_league_matches = {}
    for year, path in seasons:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for m in data.get('matches', []):
            if m.get('homeGoals') is None or m.get('awayGoals') is None:
                continue
            lg = m['league']
            if lg not in per_league_matches:
                per_league_matches[lg] = []
            per_league_matches[lg].append(m)

    rows = []
    for league, ms in per_league_matches.items():
        elo = EloModel()
        # 按日期排序（同一联赛内跨赛季按时间正序）
        sorted_ms = sorted(ms, key=lambda m: m.get('utcDate', '') or '')
        for m in sorted_ms:
            home, away = m['homeTeam'], m['awayTeam']
            ph, pd, pa = elo.predict(home, away)
            best = max([('home', ph), ('draw', pd), ('away', pa)], key=lambda x: x[1])
            hg, ag = m['homeGoals'], m['awayGoals']
            actual = 'home' if hg > ag else ('draw' if hg == ag else 'away')
            season = m.get('season', '')
            rows.append({
                'league': league, 'prob': round(best[1], 4),
                'dir': best[0], 'hit': best[0] == actual,
                'season': season,
            })
            elo.update([{
                'utcDate': m.get('utcDate', ''),
                'homeTeam': {'name': home},
                'awayTeam': {'name': away},
                'score': {'fullTime': {'home': hg, 'away': ag}},
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

    # 按赛季分组
    by_season = {}
    for s in sorted(set(r.get("season", "") for r in rows)):
        rs = [r for r in rows if r.get("season") == s]
        if rs:
            by_season[s] = {
                "n": len(rs),
                "hitRate": round(sum(1 for r in rs if r["hit"]) / len(rs), 4),
            }

    out = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": f"{len(seasons)} 赛季回测（Elo 滚动预测，跨赛季 Elo 连续）",
        "total": len(rows),
        "overall": round(sum(1 for r in rows if r["hit"]) / len(rows), 4),
        "buckets": groups,
        "byLeague": by_league,
        "bySeason": by_season,
        "directionDist": dict(Counter(r["dir"] for r in rows)),
    }

    with open(DATA_DIR / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
