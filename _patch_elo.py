# -*- coding: utf-8 -*-
"""predict.py: 集成 Elo 独立模型 + Dixon-Coles + 价值检测。"""
import re
from pathlib import Path

p = Path("scripts/predict.py")
src = p.read_text(encoding="utf-8")
changed = []

# 1. import elo
OLD = "ROOT = Path(__file__).resolve().parent.parent"
NEW = "ROOT = Path(__file__).resolve().parent.parent\nsys.path.insert(0, str(ROOT / \"scripts\"))\n\nfrom elo import EloModel, dc_probs"
if OLD in src:
    src = src.replace(OLD, NEW, 1); changed.append("import elo")
else:
    print("import 锚点未匹配")

# 2. ScoreModel 增加 dc_probs 方法（Dixon-Coles 修正的波胆）
OLD2 = """    def over_under(self, line=2.5):"""
NEW2 = """    def dc_scores(self, top_n=6, rho=-0.08):
        \"\"\"Dixon-Coles 修正的波胆分布（修正 0-0/1-0/0-1/1-1 低比分依赖）。\"\"\"
        probs = dc_probs(self.lam_h, self.lam_a, rho=rho)
        entries = [{\"score\": f\"{i}-{j}\", \"prob\": round(pp, 4)} for (i, j), pp in probs.items()]
        entries.sort(key=lambda x: -x[\"prob\"])
        return entries[:top_n]

    def dc_1x2(self, rho=-0.08):
        \"\"\"Dixon-Coles 修正的胜平负概率。\"\"\"
        probs = dc_probs(self.lam_h, self.lam_a, rho=rho)
        p_home = sum(v for (i, j), v in probs.items() if i > j)
        p_draw = sum(v for (i, j), v in probs.items() if i == j)
        p_away = sum(v for (i, j), v in probs.items() if i < j)
        return round(p_home, 4), round(p_draw, 4), round(p_away, 4)

    def over_under(self, line=2.5):"""
if OLD2 in src:
    src = src.replace(OLD2, NEW2); changed.append("ScoreModel DC")
else:
    print("ScoreModel 锚点未匹配")

# 3. main(): 构建 Elo 模型
OLD3 = """    odds_by_league = load_odds()
    messages = load_messages()
    print(f"赔率联赛: {list(odds_by_league.keys())}, 消息: {len(messages)} 条\")"""
NEW3 = """    odds_by_league = load_odds()
    messages = load_messages()
    print(f"赔率联赛: {list(odds_by_league.keys())}, 消息: {len(messages)} 条\")

    # Elo 独立模型：积分榜初始化 + 本赛季赛果迭代
    elo_model = EloModel()
    standings_data = {}
    try:
        with open(DATA_DIR / "standings.json", encoding="utf-8") as f:
            standings_data = json.load(f).get("standings", {})
        elo_model.init_from_standings(standings_data)
    except Exception:
        pass
    try:
        with open(DATA_DIR / "fixtures.json", encoding="utf-8") as f:
            fixtures_all = json.load(f).get("matches", [])
        finished = [m for m in fixtures_all if m.get("status") == "FINISHED"]
        elo_model.update(finished)
        print(f"Elo: {len(elo_model.ratings)} 队, 已用 {len(finished)} 场赛果迭代")
    except Exception:
        pass"""
if OLD3 in src:
    src = src.replace(OLD3, NEW3); changed.append("Elo 构建")
else:
    print("Elo 构建锚点未匹配")

# 4. 主循环：融合概率 fit + 价值检测
OLD4 = """            # 泊松模型: 波胆 + 大小球
            score_model = ScoreModel()
            if odds_result and odds_result.get("prob"):
                p = odds_result["prob"]
                score_model.fit(p["home"], p["draw"], p["away"])"""
NEW4 = """            # 泊松模型: 波胆 + 大小球（融合赔率与 Elo 概率 fit）
            score_model = ScoreModel()
            if odds_result and odds_result.get("prob"):
                op = odds_result["prob"]
                # Elo 独立概率
                ep_home, ep_draw, ep_away = elo_model.predict(home, away)
                pred_elo = {"home": ep_home, "draw": ep_draw, "away": ep_away}
                # 融合: 60% 赔率 + 40% Elo
                f_home = 0.6 * op["home"] + 0.4 * ep_home
                f_draw = 0.6 * op["draw"] + 0.4 * ep_draw
                f_away = 0.6 * op["away"] + 0.4 * ep_away
                score_model.fit(f_home, f_draw, f_away)
            else:
                ep_home, ep_draw, ep_away = elo_model.predict(home, away)
                pred_elo = {"home": ep_home, "draw": ep_draw, "away": ep_away}
                score_model.fit(ep_home, ep_draw, ep_away)"""
if OLD4 in src:
    src = src.replace(OLD4, NEW4); changed.append("融合概率")
else:
    print("融合概率锚点未匹配")

# 5. pred 输出：eloProb + 价值检测
OLD5 = """            # 预测比分 = 泊松模型最可能波胆（比期望值四舍五入更有区分度）
            top_cs = score_model.correct_scores(1)
            if top_cs:
                pred["predictedScore"] = top_cs[0]["score"]"""
NEW5 = """            # Elo 概率 + 价值检测（模型概率 vs 赔率隐含概率）
            pred["eloProb"] = pred_elo
            pred["eloRatings"] = {
                "home": round(elo_model.get_rating(home), 0),
                "away": round(elo_model.get_rating(away), 0),
            }
            value_picks = []
            if odds_result and odds_result.get("prob"):
                op = odds_result["prob"]
                for k, label in (("home", "主胜"), ("draw", "平局"), ("away", "客胜")):
                    diff = pred_elo[k] - op[k]
                    if abs(diff) >= 0.05:
                        value_picks.append({
                            "side": k, "label": label,
                            "modelProb": round(pred_elo[k], 3),
                            "oddsProb": round(op[k], 3),
                            "edge": round(diff, 3),
                        })
            pred["valuePicks"] = value_picks
            # Dixon-Coles 修正波胆（低比分修正）
            pred["correctScores"] = score_model.dc_scores(6)
            # 预测比分 = 最可能波胆
            top_cs = score_model.correct_scores(1)
            if top_cs:
                pred["predictedScore"] = top_cs[0]["score"]"""
if OLD5 in src:
    src = src.replace(OLD5, NEW5); changed.append("价值检测")
else:
    print("价值检测锚点未匹配")

p.write_text(src, encoding="utf-8")
print("应用:", changed if changed else "全部未匹配!")
