# -*- coding: utf-8 -*-
"""
elo_model.py - 独立球队战力模型（Elo 评分）
从积分榜初始化评分，产出独立于赔率的胜/平/负概率，用于价值检测。
"""


class EloModel:
    """Elo 评分 → 独立胜率。与盘口无关，用于找'模型概率 vs 市场概率'的价值差。"""

    K = 24          # 更新系数
    HOME_ADV = 55   # 主场优势（Elo 分）

    def __init__(self, standings: dict = None):
        self.ratings = {}          # (code, team) -> elo
        self.played = {}           # (code, team) -> 已赛场次（更新时用）
        if standings:
            self.init_from_standings(standings)

    def init_from_standings(self, standings: dict):
        """按积分榜初始化：第1名 ~1650，每降一位减 (400/n)。"""
        for code, rows in (standings or {}).items():
            n = max(1, len(rows))
            for r in rows:
                pos = r.get("position")
                if not pos:
                    continue
                elo = 1650 - (pos - 1) * (400 // n)
                team = (r.get("team") or "").lower()
                if team:
                    self.ratings[(code, team)] = elo
                short = (r.get("shortName") or "").lower()
                if short:
                    self.ratings[(code, short)] = elo

    def rating(self, code: str, team: str) -> float:
        return self.ratings.get((code, (team or "").lower()), 1500.0)

    def expected(self, r_home: float, r_away: float, home_adv: bool = True) -> float:
        diff = (r_home + (self.HOME_ADV if home_adv else 0)) - r_away
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    def match_probs(self, code: str, home: str, away: str) -> dict:
        """Elo 推导 主胜/平/客胜 概率。平局用经验值 26% 按实力接近度微调。"""
        rh = self.rating(code, home)
        ra = self.rating(code, away)
        eh = self.expected(rh, ra, home_adv=True)      # 主队期望得分（含平局）
        ea = self.expected(ra, rh, home_adv=False)     # 客队期望得分
        # 期望得分 ≈ P(win) + 0.5*P(draw)，按实力接近度分配平局
        closeness = 1 - abs(rh - ra) / 800
        pd = 0.26 * (0.5 + 0.5 * closeness)
        ph = max(0.02, min(0.95, eh - pd / 2))
        pa = max(0.02, min(0.95, ea - pd / 2))
        s = ph + pd + pa
        return {"home": ph / s, "draw": pd / s, "away": pa / s}

    def update(self, code: str, home: str, away: str, hg: int, ag: int):
        """赛后更新评分（胜负平）。"""
        rh = self.rating(code, home)
        ra = self.rating(code, away)
        eh = self.expected(rh, ra, home_adv=True)
        ea = 1 - eh
        # 实际得分
        if hg > ag:
            sh, sa = 1.0, 0.0
        elif hg == ag:
            sh, sa = 0.5, 0.5
        else:
            sh, sa = 0.0, 1.0
        # 已赛场次越多 K 值越小（收敛）
        kh = max(6, self.K - (self.played.get((code, home.lower()), 0) // 5) * 2)
        ka = max(6, self.K - (self.played.get((code, away.lower()), 0) // 5) * 2)
        self.ratings[(code, home.lower())] = rh + kh * (sh - eh)
        self.ratings[(code, away.lower())] = ra + ka * (sa - ea)
        self.played[(code, home.lower())] = self.played.get((code, home.lower()), 0) + 1
        self.played[(code, away.lower())] = self.played.get((code, away.lower()), 0) + 1


def detect_value(model_prob: dict, market_prob: dict, threshold: float = 0.05) -> list:
    """价值检测：模型概率 vs 市场隐含概率，差 >= 阈值 标为有价值。"""
    picks = []
    labels = {"home": "主胜", "draw": "平局", "away": "客胜"}
    for k, label in labels.items():
        mp = model_prob.get(k)
        op = market_prob.get(k)
        if mp is None or op is None:
            continue
        diff = mp - op
        if abs(diff) >= threshold:
            picks.append({
                "side": k, "label": label,
                "modelProb": round(mp, 3),
                "oddsProb": round(op, 3),
                "edge": round(diff, 3),
            })
    return picks
