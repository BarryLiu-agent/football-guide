# -*- coding: utf-8 -*-
"""还原 predict.py 的 import 与 API 调用到与 scripts/elo.py 匹配。"""
from pathlib import Path

p = Path("scripts/predict.py")
src = p.read_text(encoding="utf-8")

# 1. import 还原
src = src.replace(
    "from elo_model import EloModel, detect_value\nfrom elo_model import dc_probs",
    "from elo import EloModel, dc_probs")

# 2. API 调用还原（我改坏的）
src = src.replace(
    "elo_pp = elo_model.match_probs(league, home, away)\n                ep_home, ep_draw, ep_away = elo_pp['home'], elo_pp['draw'], elo_pp['away']",
    "ep_home, ep_draw, ep_away = elo_model.predict(home, away)")
src = src.replace("elo_model.match_probs(league, home, away)", "elo_model.predict(home, away)")
src = src.replace("elo_model.rating(league, home)", "elo_model.get_rating(home)")
src = src.replace("elo_model.rating(league, away)", "elo_model.get_rating(away)")

p.write_text(src, encoding="utf-8")
print("还原完成")
print("import 行:", [l for l in src.splitlines() if l.startswith("from elo")])
print("match_probs 残留:", "match_probs" in src)
print("get_rating 调用:", src.count("get_rating"))
