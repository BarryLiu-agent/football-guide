# -*- coding: utf-8 -*-
"""
elo.py - 独立球队战力模型（Elo 评分）+ Dixon-Coles 修正
产出独立于博彩赔率的胜率估计，用于价值检测。

用法:
  from elo import EloModel, dc_probs
  model = EloModel()
  model.init_from_standings(standings)      # 休赛期用积分榜排名初始化
  model.update(finished_matches)            # 赛季中用赛果迭代
  p_home, p_draw, p_away = model.predict("Arsenal", "Coventry City")
"""

import math
import unicodedata

INIT_ELO = 1500.0
K = 32.0            # 常规 K 因子
HOME_ADV = 100.0    # 主场优势（Elo 分）
SPREAD = 400.0      # Elo 换算尺度

# 常见队名变体 -> 标准归一化名（解决 Inter Milan / Internazionale 等差异）
ALIASES = {
    "inter milan": "inter", "internazionale": "inter", "inter milano": "inter",
    "ac milan": "milan", "milan": "milan",
    "atletico madrid": "atletico madrid", "atletico de madrid": "atletico madrid", "atleti": "atletico madrid", "atletico": "atletico madrid",
    "manchester united": "manchester united", "man united": "manchester united", "man utd": "manchester united",
    "manchester city": "manchester city", "man city": "manchester city",
    "bayern munich": "bayern munich", "bayern munchen": "bayern munich",
    "paris saint germain": "psg", "paris saint-germain": "psg", "psg": "psg",
    # 意甲/法甲常见带冠词名（odds 全称 → 赛季数据常用名）
    "atalanta bc": "atalanta", "as roma": "roma", "as monaco": "monaco",
    "ca osasuna": "osasuna", "rc lens": "lens", "parma calcio 1913": "parma",
    "parma calcio": "parma", "us lecce": "lecce", "us salernitana": "salernitana",
    "a cf fiorentina": "fiorentina", "acf fiorentina": "fiorentina", "cf fiorentina": "fiorentina",
    "fiorentina": "fiorentina", "genoa cfc": "genoa", "genoa": "genoa",
    "torino fc": "torino", "torino": "torino", "cagliari": "cagliari", "udinese": "udinese",
    "verona": "verona", "hellas verona": "verona", "bologna": "bologna", "empoli": "empoli",
    "monza": "monza", "ac monza": "monza", "como": "como", "venezia": "venezia",
    "frosinone": "frosinone", "lecce": "lecce", "sassuolo": "sassuolo", "spezia": "spezia",
    "salernitana": "salernitana", "cremonese": "cremonese", "sampdoria": "sampdoria",
    "crotone": "crotone", "cesena": "cesena", "palermo": "palermo", "pisa": "pisa",
    "reggiana": "reggiana", "bari": "bari", "brescia": "brescia", "carrarese": "carrarese",
    "mantova": "mantova", "modena": "modena", "southampton": "southampton",
    # 法甲
    "marseille": "marseille", "olympique de marseille": "marseille", "olympique marseille": "marseille",
    "stade rennais": "rennes", "rennes": "rennes", "lille osc": "lille", "lille": "lille",
    "rc strasbourg": "strasbourg", "strasbourg": "strasbourg", "stade brestois": "brest",
    "brest": "brest", "fc nantes": "nantes", "nantes": "nantes", "toulouse": "toulouse",
    "stade de reims": "reims", "reims": "reims", "ajaccio": "ajaccio", "lorient": "lorient",
    "le havre": "le havre", "havre ac": "le havre", "auxerre": "auxerre", "angers": "angers",
    "montpellier": "montpellier", "nice": "nice", "ogc nice": "nice", "le mans": "le mans",
    "paris fc": "paris fc", "troyes": "troyes", "metz": "metz", "gignac": "gignac",
    # 西甲
    "espanyol": "espanyol", "rcd espanyol": "espanyol", "osasuna": "osasuna",
    "deportivo la coruna": "deportivo", "rc deportivo": "deportivo", "deportivo": "deportivo",
    "elche cf": "elche", "elche": "elche", "malaga cf": "malaga", "malaga": "malaga",
    "real racing club de santander": "racing santander", "racing santander": "racing santander",
    "real racing santander": "racing santander", "racing club": "racing santander",
    "valladolid": "valladolid", "real valladolid": "valladolid", "girona": "girona",
    "leganes": "leganes", "cd leganes": "leganes", "tenerife": "tenerife", "mirandes": "mirandes",
    "oviedo": "oviedo", "real oviedo": "oviedo", "eibar": "eibar", "sporting gijon": "sporting gijon",
    "granada": "granada", "castellon": "castellon", "cordoba": "cordoba", "huesca": "huesca",
    "albacete": "albacete", "burgos": "burgos", "cartagena": "cartagena", "ferrol": "ferrol",
    "eldense": "eldense", "racing ferrol": "ferrol",
    # 德甲
    "fc augsburg": "augsburg", "augsburg": "augsburg", "sc paderborn": "paderborn",
    "paderborn": "paderborn", "sv elversberg": "elversberg", "elversberg": "elversberg",
    "fc schalke 04": "schalke", "schalke 04": "schalke", "schalke": "schalke",
    "fsv mainz 05": "mainz", "mainz 05": "mainz", "mainz": "mainz",
    "holstein kiel": "kiel", "holstein": "kiel", "fc st pauli 1910": "st pauli",
    "st pauli": "st pauli", "sv darmstadt 98": "darmstadt", "darmstadt": "darmstadt",
    "fortuna dusseldorf": "fortuna dusseldorf", "fc kaiserslautern": "kaiserslautern",
    "kaiserslautern": "kaiserslautern", "hamburger sv": "hamburg", "hamburg": "hamburg",
    "fc koln": "koln", "koln": "koln", "fc heidenheim 1846": "heidenheim",
    "heidenheim": "heidenheim", "1 fc union berlin": "union berlin", "union berlin": "union berlin",
    "borussia monchengladbach": "gladbach", "borussia mgladbach": "gladbach", "mgladbach": "gladbach",
    "borussia m.gladbach": "gladbach", "borussia m'gladbach": "gladbach",
    "fc koln": "koln", "koln": "koln",
    "bayer leverkusen": "leverkusen", "bayer 04 leverkusen": "leverkusen",
    "porto": "porto", "fc porto": "porto",
    "sporting lisbon": "sporting", "sporting cp": "sporting",
    "benfica": "benfica", "sl benfica": "benfica",
    "athletic bilbao": "athletic", "athletic club": "athletic",
}


def norm_team(name: str) -> str:
    """归一化队名：去后缀/重音/变体 → 标准键。"""
    words = (name or "").lower().replace("&", " ").split()
    words = [w for w in words if w not in ("fc", "afc", "cf", "sc", "ac", "1", "2", "0", "4", "05", "1846", "1910", "04", "01", "1.")]
    raw = " ".join(words)
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    raw = " ".join(raw.split())
    return ALIASES.get(raw, raw)


class EloModel:
    def __init__(self):
        self.ratings = {}  # team(norm) -> elo

    # ── 初始化 ─────────────────────────────

    def init_from_standings(self, standings: dict) -> None:
        """用积分榜排名初始化 Elo（休赛期无赛果时使用）。"""
        for rows in (standings or {}).values():
            for r in rows:
                pos = r.get("position") or 20
                elo = INIT_ELO + (21 - min(pos, 20)) * 15
                self.ratings.setdefault(norm_team(r.get("team", "")), elo)

    def init_from_players(self, xg_teams: list) -> None:
        """用 xG 榜初始化（按 xG 排序 → Elo）。"""
        if not xg_teams:
            return
        sorted_teams = sorted(xg_teams, key=lambda t: -(t.get("xG") or 0))
        for i, t in enumerate(sorted_teams[:20]):
            self.ratings.setdefault(norm_team(t.get("title", "")), INIT_ELO + (21 - i) * 15)

    def update(self, finished_matches: list) -> None:
        """用已完赛比赛迭代更新 Elo（按时间顺序）。"""
        matches = sorted(finished_matches, key=lambda m: m.get("utcDate", ""))
        for m in matches:
            home_t = (m.get("homeTeam") or {})
            away_t = (m.get("awayTeam") or {})
            home = norm_team(home_t.get("name") or home_t.get("shortName", ""))
            away = norm_team(away_t.get("name") or away_t.get("shortName", ""))
            ft = (m.get("score") or {}).get("fullTime") or {}
            hg, ag = ft.get("home"), ft.get("away")
            if not home or not away or hg is None or ag is None:
                continue
            self._play(home, away, hg, ag)

    def _play(self, home: str, away: str, hg: int, ag: int) -> None:
        e_home = self.ratings.setdefault(home, INIT_ELO)
        e_away = self.ratings.setdefault(away, INIT_ELO)
        exp_home = 1 / (1 + 10 ** ((e_away + HOME_ADV - e_home) / SPREAD))
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
        p_draw = 0.30 * math.exp(-abs(d) / 400.0)
        p_home = p_home_raw * (1 - p_draw)
        p_away = (1 - p_home_raw) * (1 - p_draw)
        return round(p_home, 4), round(p_draw, 4), round(p_away, 4)

    def get_rating(self, team: str) -> float:
        return self._find_rating(team)

    def _find_rating(self, team: str) -> float:
        """精确匹配 → 全词包含匹配 → 默认 1500。"""
        k = norm_team(team)
        if k in self.ratings:
            return self.ratings[k]
        # 全词包含（防首词误配："real racing santander" 不命中 "real madrid"）
        words = [w for w in k.split() if w]
        if len(words) >= 2:
            for key, val in self.ratings.items():
                if all(w in key.split() for w in words):
                    return val
        return INIT_ELO

    def ratings_table(self, top_n: int = 20) -> list:
        return sorted(self.ratings.items(), key=lambda kv: -kv[1])[:top_n]


# ── Dixon-Coles 低比分修正 ────────────────

def dc_tau(x: int, y: int, lam: float, mu: float, rho: float = -0.08) -> float:
    """Dixon-Coles tau 修正因子（修正低比分依赖，提升平局精度）。"""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dc_probs(lam: float, mu: float, rho: float = -0.08, max_goals: int = 8) -> list:
    """Dixon-Coles 修正的比分概率矩阵。返回 [{score, prob}] 按概率降序。"""
    mat = {}
    for i in range(max_goals):
        for j in range(max_goals):
            p = _poisson(i, lam) * _poisson(j, mu) * dc_tau(i, j, lam, mu, rho)
            mat[(i, j)] = p
    s = sum(mat.values())
    if s > 0:
        mat = {k: v / s for k, v in mat.items()}
    entries = [{"score": f"{i}-{j}", "prob": round(p, 4)} for (i, j), p in mat.items()]
    entries.sort(key=lambda x: -x["prob"])
    return entries


def dc_1x2_from(lam: float, mu: float, rho: float = -0.08, max_goals: int = 8):
    """从 λ/μ 计算 Dixon-Coles 胜平负概率 (p_home, p_draw, p_away)。"""
    entries = dc_probs(lam, mu, rho, max_goals)
    p_home = sum(e["prob"] for e in entries if int(e["score"].split("-")[0]) > int(e["score"].split("-")[1]))
    p_draw = sum(e["prob"] for e in entries if e["score"].split("-")[0] == e["score"].split("-")[1])
    p_away = sum(e["prob"] for e in entries if int(e["score"].split("-")[0]) < int(e["score"].split("-")[1]))
    return p_home, p_draw, p_away


class XgModel:
    """基于 xG 的攻防强度模型。
    用历史比赛 xG 算每队进攻强度(场均 xG 创造)与防守强度(场均 xG 丢球)，
    预测时用泊松参数 λ_home = 攻(主) × 防(客) / 联赛均值，再经 Dixon-Coles 得 1X2。
    与 Elo 互补：Elo 看赛果（运气成分），xG 看过程（真实强度）。
    """

    def __init__(self, max_age: int = 20):
        self.attack = {}   # team(norm) -> 场均 xG 创造
        self.defense = {}  # team(norm) -> 场均 xG 丢球
        self.avg_home_xg = 1.45  # 五大联赛主场场均 xG 基准（回退值）
        self.avg_away_xg = 1.15  # 客场基准
        self._home_sum = 0.0
        self._away_sum = 0.0
        self._home_n = 0
        self._away_n = 0
        self.max_age = max_age  # 每队只用最近 N 场（越近越反映当前状态）
        self._team_recent = {}  # team -> list[(xg_for, xg_against)]

    def add_match(self, home: str, away: str, xg_home, xg_away):
        """累计一场比赛的 xG（按时间顺序调用）。"""
        if xg_home is None or xg_away is None:
            return
        for t, xgf, xga in ((home, xg_home, xg_away), (away, xg_away, xg_home)):
            rec = self._team_recent.setdefault(norm_team(t), [])
            rec.append((float(xgf), float(xga)))
            if len(rec) > self.max_age:
                rec.pop(0)
        self._home_sum += float(xg_home)
        self._away_sum += float(xg_away)
        self._home_n += 1
        self._away_n += 1

    def finalize(self):
        """训练结束：计算每队攻防强度与联赛均值。"""
        if self._home_n:
            self.avg_home_xg = self._home_sum / self._home_n
        if self._away_n:
            self.avg_away_xg = self._away_sum / self._away_n
        for t, rec in self._team_recent.items():
            if rec:
                self.attack[t] = sum(r[0] for r in rec) / len(rec)
                self.defense[t] = sum(r[1] for r in rec) / len(rec)

    def predict(self, home: str, away: str):
        """返回 (p_home, p_draw, p_away)。两队缺数据时回退均匀 1/3。"""
        h, a = norm_team(home), norm_team(away)
        atk_h = self.attack.get(h)
        def_a = self.defense.get(a)
        atk_a = self.attack.get(a)
        def_h = self.defense.get(h)
        if None in (atk_h, def_a, atk_a, def_h):
            return 1 / 3, 1 / 3, 1 / 3
        # 主队 λ：主场基准 × 攻(主)/主场均值 × 防(客)/客场均值（防守弱=丢球多）
        lam_home = self.avg_home_xg * (atk_h / self.avg_home_xg) * (def_a / self.avg_away_xg)
        lam_away = self.avg_away_xg * (atk_a / self.avg_away_xg) * (def_h / self.avg_home_xg)
        p_home, p_draw, p_away = dc_1x2_from(lam_home, lam_away)
        return round(p_home, 4), round(p_draw, 4), round(p_away, 4)

    def lam(self, home: str, away: str):
        """返回 (λ_home, λ_away) 或 (None, None)。"""
        h, a = norm_team(home), norm_team(away)
        atk_h, def_a, atk_a, def_h = self.attack.get(h), self.defense.get(a), self.attack.get(a), self.defense.get(h)
        if None in (atk_h, def_a, atk_a, def_h):
            return None, None
        lam_home = self.avg_home_xg * (atk_h / self.avg_home_xg) * (def_a / self.avg_away_xg)
        lam_away = self.avg_away_xg * (atk_a / self.avg_away_xg) * (def_h / self.avg_home_xg)
        return lam_home, lam_away
