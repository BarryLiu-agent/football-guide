# -*- coding: utf-8 -*-
"""重写 predict.py 主循环（521-639 行）为单一清晰版本，修复补丁堆叠混乱。"""
from pathlib import Path

p = Path("scripts/predict.py")
lines = p.read_text(encoding="utf-8").splitlines()

# 521 行（for league...）到 639 行（predictions.append 后）替换
start = 520  # 0-based: 521 行
end = 639    # 0-based: 640 行前

new_loop = '''    for league, matches in odds_by_league.items():
        for m in matches:
            home = m.get("homeTeam", "")
            away = m.get("awayTeam", "")
            context = {
                "homeTeam": home, "awayTeam": away,
                "odds": m.get("markets", {}).get("h2h"),
                "messages": messages,
            }
            results = {}
            for a in analyzers:
                results[a.name] = a.analyze(context)

            odds_result = results.get("odds")
            msg_result = results.get("message")

            # ── Elo 独立概率 ──
            ep_home, ep_draw, ep_away = elo.predict(home, away)
            pred_elo = {"home": ep_home, "draw": ep_draw, "away": ep_away}
            market_prob = None
            if odds_result and odds_result.get("prob"):
                market_prob = odds_result["prob"]

            # ── 融合概率：60% 市场 + 40% Elo（供波胆/大小球建模）──
            if market_prob:
                f_home = 0.6 * market_prob["home"] + 0.4 * ep_home
                f_draw = 0.6 * market_prob["draw"] + 0.4 * ep_draw
                f_away = 0.6 * market_prob["away"] + 0.4 * ep_away
            else:
                f_home, f_draw, f_away = ep_home, ep_draw, ep_away

            score_model = ScoreModel()
            score_model.fit(f_home, f_draw, f_away)

            pred = predictor.predict(home, away, odds_result, msg_result)
            pred["league"] = league
            pred["kickoff"] = m.get("kickoff", "")
            pred["matchUrl"] = m.get("matchUrl", "")

            # 真实赔率 + 涨跌 + 凯利
            raw_odds = odds_result.get("rawOdds") if odds_result else None
            pred["rawOdds"] = raw_odds
            prev_h2h = m.get("markets", {}).get("prevH2h") or m.get("prevH2h")
            odds_change = None
            if prev_h2h and raw_odds:
                def _delta(cur, prev):
                    if isinstance(cur, (int, float)) and isinstance(prev, (int, float)):
                        return round(cur - prev, 2)
                    return None
                odds_change = {
                    "home": _delta(raw_odds.get("home"), prev_h2h.get("home") or prev_h2h.get("Home")),
                    "draw": _delta(raw_odds.get("draw"), prev_h2h.get("draw") or prev_h2h.get("Draw")),
                    "away": _delta(raw_odds.get("away"), prev_h2h.get("away") or prev_h2h.get("Away")),
                }
            pred["oddsChange"] = odds_change

            kelly = None
            if raw_odds and market_prob:
                kelly = {}
                for k in ("home", "draw", "away"):
                    o = raw_odds.get(k)
                    pp = market_prob.get(k)
                    if isinstance(o, (int, float)) and o > 1 and pp:
                        kelly[k] = round((o * pp - 1) / (o - 1), 3)
                    else:
                        kelly[k] = None
            pred["kelly"] = kelly
            pred["spreads"] = m.get("markets", {}).get("spreads")

            # ── Elo 输出 + 价值检测 ──
            pred["eloProb"] = {k: round(v, 3) for k, v in pred_elo.items()}
            pred["eloRatings"] = {
                "home": round(elo.get_rating(home), 0),
                "away": round(elo.get_rating(away), 0),
            }
            pred["modelProbs"] = {
                "home": round(f_home, 3), "draw": round(f_draw, 3), "away": round(f_away, 3)
            }
            value_picks = []
            if market_prob:
                for k, label in (("home", "主胜"), ("draw", "平局"), ("away", "客胜")):
                    diff = pred_elo[k] - market_prob[k]
                    if abs(diff) >= 0.05:
                        value_picks.append({
                            "side": k, "label": label,
                            "modelProb": round(pred_elo[k], 3),
                            "oddsProb": round(market_prob[k], 3),
                            "edge": round(diff, 3),
                        })
            pred["valuePicks"] = value_picks

            # Dixon-Coles 波胆 + 大小球 + 分布
            pred["correctScores"] = score_model.dc_scores(6)
            top_cs = score_model.correct_scores(1)
            if top_cs:
                pred["predictedScore"] = top_cs[0]["score"]

            ou = None
            totals_mkt = m.get("markets", {}).get("totals")
            if totals_mkt and totals_mkt.get("over") and totals_mkt.get("under"):
                po, pu = 1 / totals_mkt["over"], 1 / totals_mkt["under"]
                s = po + pu
                ou = {"line": totals_mkt["line"], "over": round(po / s, 3), "under": round(pu / s, 3),
                      "overPrice": totals_mkt["over"], "underPrice": totals_mkt["under"]}
            if not ou:
                ou = score_model.over_under(2.5)
            pred["overUnder"] = ou

            sh = find_standing(league, home)
            sa = find_standing(league, away)
            pred["standings"] = ({"home": sh, "away": sa} if sh and sa else None)
            pred["totalGoalsDist"] = score_model.total_goals_dist()
            pred["btts"] = score_model.btts_prob()
            pred["expectedGoals"] = {
                "home": round(score_model.lam_h, 2), "away": round(score_model.lam_a, 2)
            }
            pred["analysis"] = AnalysisWriter.generate(home, away, odds_result, msg_result, score_model, ou, pred["spreads"], pred["standings"], value_picks)
            predictions.append(pred)
'''

lines[start:end] = new_loop.splitlines()
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"主循环已重写 (原 {end-start} 行 -> 新 {len(new_loop.splitlines())} 行)")
