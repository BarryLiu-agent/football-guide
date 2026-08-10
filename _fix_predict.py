# -*- coding: utf-8 -*-
"""修复 predict.py 的 Elo/dc 调用，匹配 elo_model.py 实际 API。"""
import re
from pathlib import Path

# 1. elo_model.py 增加 Dixon-Coles 函数
em = Path("scripts/elo_model.py")
esrc = em.read_text(encoding="utf-8")
if "def dc_probs" not in esrc:
    esrc += '''

def dc_probs(lam_h: float, lam_a: float, rho: float = 0.03, max_goals: int = 6):
    """Dixon-Coles 低比分修正的比分概率矩阵。
    P(i,j) = Poisson(i;λh) * Poisson(j;λa) * τ(i,j)
    τ: 修正 (0,0)/(1,0)/(0,1)/(1,1) 低比分，修正平局被低估的问题。
    """
    import math
    po = lambda k, lam: math.exp(-lam) * lam ** k / math.factorial(k)
    m = [[po(i, lam_h) * po(j, lam_a) for j in range(max_goals)] for i in range(max_goals)]
    # τ 修正
    if lam_h > 0.01 and lam_a > 0.01:
        m[0][0] *= (1 - lam_h * lam_a * rho)
        m[1][0] *= (1 - lam_h * rho)
        m[0][1] *= (1 - lam_a * rho)
        m[1][1] *= (1 + rho)
        # 防负值
        for i in (0, 1):
            for j in (0, 1):
                m[i][j] = max(0.0, m[i][j])
    # 归一化
    s = sum(sum(row) for row in m)
    return [[v / s for v in row] for row in m]
'''
    em.write_text(esrc, encoding="utf-8")
    print("dc_probs added")

# 2. predict.py import 修复
p = Path("scripts/predict.py")
src = p.read_text(encoding="utf-8")
src = src.replace(
    "from elo import EloModel, dc_probs",
    "from elo_model import EloModel, detect_value\nfrom elo_model import dc_probs")
print("import fixed")

# 3. predict()/get_rating() -> match_probs()/rating()
src = src.replace("elo_model.predict(home, away)", "elo_model.match_probs(league, home, away)")
src = src.replace("elo_model.get_rating(home)", "elo_model.rating(league, home)")
src = src.replace("elo_model.get_rating(away)", "elo_model.rating(league, away)")
print("API calls fixed")

# 4. 主循环：elo 概率结果变量名检查（可能 elo_model.predict 返回 tuple 或 dict）
# match_probs 返回 dict {'home','draw','away'}
src = re.sub(r"ep_home, ep_draw, ep_away = elo_model\.match_probs\(league, home, away\)",
             "elo_pp = elo_model.match_probs(league, home, away)\n                ep_home, ep_draw, ep_away = elo_pp['home'], elo_pp['draw'], elo_pp['away']",
             src)
p.write_text(src, encoding="utf-8")
print("done")
