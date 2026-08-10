# -*- coding: utf-8 -*-
"""predict.py 最小补丁：输出 eloProbs/modelProbs/valueMarks 字段。"""
from pathlib import Path

p = Path("scripts/predict.py")
src = p.read_text(encoding="utf-8")

ANCHOR = '            pred = predictor.predict(home, away, odds_result, msg_result)\n            pred["league"] = league'
INSERT = '''            pred = predictor.predict(home, away, odds_result, msg_result)
            # ── Elo 独立模型输出 + 价值检测 ──
            pred["eloProbs"] = {"home": round(pred_elo["home"], 3), "draw": round(pred_elo["draw"], 3), "away": round(pred_elo["away"], 3)}
            if odds_result and odds_result.get("prob"):
                op = odds_result["prob"]
                m_ph = round(0.6 * op["home"] + 0.4 * pred_elo["home"], 3)
                m_pd = round(0.6 * op["draw"] + 0.4 * pred_elo["draw"], 3)
                m_pa = round(0.6 * op["away"] + 0.4 * pred_elo["away"], 3)
                pred["modelProbs"] = {"home": m_ph, "draw": m_pd, "away": m_pa}
                # 价值检测：融合模型概率 vs 市场概率，差值 >= 5% 为有价值
                edges = {k: round(m_ph - op["home"], 3) if k == "home" else
                               round(m_pd - op["draw"], 3) if k == "draw" else
                               round(m_pa - op["away"], 3) for k in ("home", "draw", "away")}
                picks = [k for k, v in edges.items() if v >= 0.05]
                pred["valueMarks"] = {"edges": edges, "picks": picks}
            else:
                pred["modelProbs"] = pred["eloProbs"]
                pred["valueMarks"] = {"edges": None, "picks": []}
            pred["league"] = league'''

assert ANCHOR in src, "anchor not found"
src = src.replace(ANCHOR, INSERT, 1)
p.write_text(src, encoding="utf-8")
print("补丁应用 OK")
