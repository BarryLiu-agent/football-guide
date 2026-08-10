# -*- coding: utf-8 -*-
"""在 predict.py 中插入 EloModel 类定义（在 ScoreModel 注释前）。"""
from pathlib import Path

p = Path("scripts/predict.py")
src = p.read_text(encoding="utf-8")

MODEL_CODE = '''
# ── Elo 独立战力模型（与赔率无关的独立观点）─────────────

def _norm_team(s):
    import re
    return re.sub(r"\\b(fc|afc|cf|sc|ac)\\b", "", (s or "").lower()).replace("&", "").strip()


class EloModel:
    """Elo 评分模型：积分榜(上赛季)初始化 → 本赛季赛果持续更新。"""

    BASE = 1500.0
    HOME_ADV = 100.0
    K = 32.0

    def __init__(self, standings_by_league=None):
        self.ratings = {}
        for league, rows in (standings_by_league or {}).items():
            for row in rows:
                team = row.get("team") or ""
                pos = row.get("position") or 20
                self.ratings[(league, _norm_team(team))] = self.BASE + (20 - pos) * 20

    def get_rating(self, league, team):
        return self.ratings.get((league, _norm_team(team)), self.BASE)

    def expected(self, r_a, r_b):
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    def match_probs(self, league, home, away):
        """返回 (p_home, p_draw, p_away)：Elo 独立估计。"""
        rh = self.get_rating(league, home) + self.HOME_ADV
        ra = self.get_rating(league, away)
        ph = self.expected(rh, ra)
        pa = self.expected(ra, rh)
        closeness = 1.0 - abs(rh - ra) / 800.0
        pd = 0.24 * closeness + 0.06
        s = ph + pd + pa
        return ph / s, pd / s, pa / s

    def update(self, league, home, away, gh, ga):
        rh = self.get_rating(league, home) + self.HOME_ADV
        ra = self.get_rating(league, away)
        eh = self.expected(rh, ra)
        ea = self.expected(ra, rh)
        sh = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        sa = 1.0 - sh
        self.ratings[(league, _norm_team(home))] = self.get_rating(league, home) + self.K * (sh - eh)
        self.ratings[(league, _norm_team(away))] = self.get_rating(league, away) + self.K * (sa - ea)


def dc_tau(x, y, rho=-0.1):
    """Dixon-Coles 低比分修正（提升平局/1-0/0-1 精度）。"""
    if x == 0 and y == 0:
        return 1 - rho
    if (x == 0 and y == 1) or (x == 1 and y == 0):
        return 1 + rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0
'''

if "class EloModel" in src:
    print("已存在，跳过")
else:
    # 定位 ScoreModel 类的注释行
    idx = src.find("# ── 泊松比分模型")
    if idx == -1:
        print("锚点未找到!")
    else:
        src = src[:idx] + MODEL_CODE + "\n" + src[idx:]
        p.write_text(src, encoding="utf-8")
        print("EloModel 类已插入")
