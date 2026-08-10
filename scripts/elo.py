# -*- coding: utf-8 -*-
"""
elo.py - 独立球队战力模型（Elo 评分）
产出独立于博彩赔率的胜率估计，用于价值检测。

用法:
  from elo import EloModel
  model = EloModel()
  model.init_from_standings(standings)      # 休赛期用排名初始化
  model.update(finished_matches)            # 赛季中用赛果迭代
  p_home, p_draw, p_away = model.predict("Arsenal", "Coventry")
"""

import math

INIT_ELO = 1500.0
K = 32.0            # 常规 K 因子
HOME_ADV = 100.0    # 主场优势（Elo 分）
SPREAD = 400.0      # Elo 换算尺度


class EloModel:
    def __init__(self):
        self.ratings = {}  # team(lower) -> elo

    # ── 初始化 ─────────────────────────────

    def init_from_standings(self, standings: dict) -> None:
        """用积分榜排名初始化 Elo（休赛期无赛果时使用）。"""
        for league, rows in (standings or {}).items():
            for r in rows:
                # 排名 1-20 → 1650-1400
                pos = r.get("position") or 20
                elo = INIT_ELO + (21 - min(pos, 20)) * 15
                self.ratings[self._key(r.get("team", ""))] = elo

    def init_from_players(self, xg_teams: list) -> None:
        """用 xG 榜初始化（按 xG 排序 → Elo）。"""
        if not xg_teams:
            return
        sorted_teams = sorted(xg_teams, key=lambda t: -(t.get("xG") or 0))
        for i, t in enumerate(sorted_teams[:20]):
            self.ratings.setdefault(self._key(t.get("title", "")), INIT_ELO + (21 - i) * 15)

    def update(self, finished_matches: list) -> None:
        """用已完赛比赛迭代更新 Elo（按时间顺序）。"""
        matches = sorted(finished_matches, key=lambda m: m.get("utcDate", ""))
        for m in matches:
            home = self._key((m.get("homeTeam") or {}).get("name", "") or (m.get("homeTeam") or {}).get("shortName", ""))
            away = self._key((m.get("awayTeam") or {}).get("name", "") or (m.get("awayTeam") or {}).get("shortName", ""))
            ft = (m.get("score") or {}).get("fullTime") or {}
            hg, ag = ft.get("home"), ft.get("away")
            if not home or not away or hg is None or ag is None:
                continue
            self._play(home, away, hg, ag)

    def _play(self, home, away, hg, ag) -> None:
        e_home = self.ratings.setdefault(home, INIT_ELO)
        e_away = self.ratings.setdefault(away, INIT_ELO)
        exp_home = 1 / (1 + 10 ** ((e_away + HOME_ADV - e_home) / SPREAD))
        # 实际结果（含平局折算）
        if hg > ag:
            s_home = 1.0
        elif hg < ag:
            s_home = 0.0
        else:
            s_home = 0.5
        diff = K * (s_home - exp_home)
        self.ratings[home] = e_home + diff
        self.ratings[away] = e_away - diff

    # ── 预测 ───────────────────────────────

    def predict(self, home: str, away: str):
        """返回 (p_home, p_draw, p_away)。基于 Elo 差 + 经验平局率。"""
        e_home = self._find_rating(home)
        e_away = self._find_rating(away)
        d = e_home + HOME_ADV - e_away
        p_home_raw = 1 / (1 + 10 ** (-d / SPREAD))
        # 平局概率随实力差递减（足球经验公式）
        p_draw = 0.30 * math.exp(-abs(d) / 400.0)
        p_home = p_home_raw * (1 - p_draw)
        p_away = (1 - p_home_raw) * (1 - p_draw)
        return round(p_home, 4), round(p_draw, 4), round(p_away, 4)

    def get_rating(self, team: str) -> float:
        return self._find_rating(team)

    def _find_rating(self, team: str) -> float:
        """精确匹配 → 首词匹配 → 默认 1500（解决 München/Munich 等译名差异）。"""
        k = self._key(team)
        if k in self.ratings:
            return self.ratings[k]
        first = k.split()[0]
        if first:
            for key, val in self.ratings.items():
                if key.split()[0] == first:
                    return val
        return INIT_ELO

    def ratings_table(self, top_n: int = 20) -> list:
        """返回 Elo 排行榜。"""
        return sorted(self.ratings.items(), key=lambda kv: -kv[1])[:top_n]

    ALIASES = {
        "inter milan": "inter", "internazionale": "inter", "inter milano": "inter",
        "ac milan": "milan", "milan": "milan",
        "atletico madrid": "atletico madrid", "atletico de madrid": "atletico madrid", "atleti": "atletico madrid",
        "manchester united": "manchester united", "man united": "manchester united", "man utd": "manchester united",
        "manchester city": "manchester city", "man city": "manchester city",
        "bayern munich": "bayern munich", "bayern munchen": "bayern munich",
        "paris saint germain": "psg", "paris saint-germain": "psg", "psg": "psg",
        "bodo glimt": "bodoglimt", "bodo/glimt": "bodoglimt",
        "red bull leipzig": "leipzig", "rb leipzig": "leipzig", "leipzig": "leipzig",
        "borussia monchengladbach": "gladbach", "borussia mgladbach": "gladbach", "mgladbach": "gladbach",
        "fc koln": "koln", "koln": "koln",
        "bayer leverkusen": "leverkusen", "bayer 04 leverkusen": "leverkusen",
        "porto": "porto", "fc porto": "porto",
        "sporting lisbon": "sporting", "sporting cp": "sporting",
        "benfica": "benfica", "sl benfica": "benfica",
        "athletic bilbao": "athletic", "athletic club": "athletic",
    }

    @staticmethod
    def _key(name: str) -> str:
        import re
        import unicodedata
        raw = re.sub(r"(fc|afc|cf|sc|ac)", "", (name or "").lower()).replace("&", "").strip()
        raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
        s = re.sub(r"[^a-z0-9 ]", " ", raw)
        s = re.sub(r"\s+", " ", s).strip()
        return EloModel.ALIASES.get(s, s)



# ── Dixon-Coles 低比分修正 ────────────────

def dc_tau(x: int, y: int, lam: float, mu: float, rho: float = -0.08) -> float:
    """Dixon-Coles tau 修正因子（修正低比分依赖）。"""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def dc_probs(lam: float, mu: float, rho: float = -0.08, max_goals: int = 8) -> list:
    """带 Dixon-Coles 修正的比分概率矩阵。返回 {(i,j): prob}。"""
    import math
    pois = lambda k, l: math.exp(-l) * l ** k / math.factorial(k)
    probs = {}
    for i in range(max_goals):
        for j in range(max_goals):
            p = dc_tau(i, j, lam, mu, rho) * pois(i, lam) * pois(j, mu)
            probs[(i, j)] = max(p, 0.0)
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}
