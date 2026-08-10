# -*- coding: utf-8 -*-
"""
postprocess.py - 预测后处理：Elo 独立概率 + 价值检测
读取 predictions.json，为每场比赛计算：
  - eloProbs:   Elo 独立胜率（不依赖赔率）
  - modelProbs: 融合概率（60% 市场 + 40% Elo）
  - valueMarks: 价值检测（融合概率 vs 市场概率，差值 >= 5% → 有价值）

用法: python scripts/postprocess.py
"""
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from elo import EloModel

DATA_DIR = ROOT / "data"
PRED_FILE = DATA_DIR / "predictions.json"
VALUE_THRESHOLD = 0.05  # 模型 vs 市场差值 >= 5% 视为有价值


def norm_team(s):
    import re
    return re.sub(r"\b(fc|afc|cf|sc)\b", "", (s or "").lower()).replace("&", "").strip()


def main():
    if not PRED_FILE.exists():
        print("无 predictions.json")
        return 1

    predictions = json.loads(PRED_FILE.read_text(encoding="utf-8"))
    preds = predictions.get("predictions", [])

    # Elo：从积分榜初始化 + 已完赛赛果更新
    elo = EloModel()
    try:
        st_path = DATA_DIR / "standings.json"
        if st_path.exists():
            elo.init_from_standings(json.loads(st_path.read_text(encoding="utf-8")).get("standings", {}))
        fix_path = DATA_DIR / "fixtures.json"
        if fix_path.exists():
            fixtures = json.loads(fix_path.read_text(encoding="utf-8")).get("matches", [])
            finished = [m for m in fixtures if m.get("status") == "FINISHED"]
            elo.update(finished)
    except Exception as e:
        print(f"Elo 初始化失败: {e}")

    n_elo = n_value = 0
    for p in preds:
        home, away = p.get("homeTeam", ""), p.get("awayTeam", "")
        if not home or not away:
            continue
        ep_h, ep_d, ep_a = elo.predict(home, away)
        p["eloProbs"] = {"home": round(ep_h, 3), "draw": round(ep_d, 3), "away": round(ep_a, 3)}
        p["eloRatings"] = {"home": round(elo.get_rating(home), 0), "away": round(elo.get_rating(away), 0)}
        n_elo += 1

        market = p.get("probabilities")
        if market and market.get("home") is not None:
            m_ph = 0.6 * market["home"] + 0.4 * ep_h
            m_pd = 0.6 * market.get("draw", 0.25) + 0.4 * ep_d
            m_pa = 0.6 * market.get("away", 0.25) + 0.4 * ep_a
            p["modelProbs"] = {"home": round(m_ph, 3), "draw": round(m_pd, 3), "away": round(m_pa, 3)}
            edges = {
                "home": round(m_ph - market["home"], 3),
                "draw": round(m_pd - market.get("draw", 0.25), 3),
                "away": round(m_pa - market.get("away", 0.25), 3),
            }
            picks = [k for k, v in edges.items() if v >= VALUE_THRESHOLD]
            p["valueMarks"] = {"edges": edges, "picks": picks}
            if picks:
                n_value += 1
        else:
            p["modelProbs"] = p["eloProbs"]
            p["valueMarks"] = {"edges": None, "picks": []}

    PRED_FILE.write_text(json.dumps(predictions, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"后处理完成: {n_elo} 场含 Elo 概率, {n_value} 场标记有价值")
    for p in preds[:5]:
        if p.get("valueMarks", {}).get("picks"):
            print(f"  💰 {p['homeTeam']} vs {p['awayTeam']}: {p['valueMarks']['picks']} 有价值")
    return 0


if __name__ == "__main__":
    sys.exit(main())
