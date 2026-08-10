"""
predict.py - 比分预测引擎
结合赔率隐含概率与消息信号，输出每场比赛的比分预测、置信度与依据摘要。

用法:
  python scripts/predict.py

可扩展接口:
  Analyzer (抽象基类) - 实现 analyze(context) -> dict
    已有实现: OddsAnalyzer (赔率→隐含概率), MessageAnalyzer (消息→信号)
  新分析器: 继承 Analyzer, 在 ANALYSIS_REGISTRY 注册即可, 主流程不改
"""

import io
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from elo import EloModel, dc_probs
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
ODDS_DIR = DATA_DIR / "odds"


# ── 抽象接口 ─────────────────────────────────────────────

class Analyzer:
    """分析器抽象基类。新分析器继承并实现 analyze() 即可。"""

    name = "base"

    def __init__(self, rules: dict):
        self.rules = rules

    def analyze(self, context: dict) -> dict:
        raise NotImplementedError


# ── 赔率分析器 ───────────────────────────────────────────

class OddsAnalyzer(Analyzer):
    """赔率 → 去水后隐含概率。"""

    name = "odds"

    def analyze(self, context: dict) -> dict:
        odds = context.get("odds")
        if not odds or not isinstance(odds, dict):
            return {"prob": None, "fairOdds": None, "margin": None}

        # 支持两种结构: 数字 {home,draw,away} 或 名称映射 {Home: 1.5, ...}
        home_o = odds.get("home") or odds.get("Home")
        draw_o = odds.get("draw") or odds.get("Draw")
        away_o = odds.get("away") or odds.get("Away")

        values = [v for v in [home_o, draw_o, away_o] if isinstance(v, (int, float)) and v > 1]
        if len(values) < 3:
            return {"prob": None, "fairOdds": None, "margin": None}

        # 去水 (remove bookmaker margin)
        raw_probs = [1 / v for v in values]
        margin = sum(raw_probs) - 1
        fair = [p / sum(raw_probs) for p in raw_probs]

        return {
            "prob": {"home": fair[0], "draw": fair[1], "away": fair[2]},
            "fairOdds": {"home": 1 / fair[0], "draw": 1 / fair[1], "away": 1 / fair[2]},
            "margin": round(margin, 4),
            "rawOdds": {"home": home_o, "draw": draw_o, "away": away_o},
        }


# ── 消息分析器 ───────────────────────────────────────────

class MessageAnalyzer(Analyzer):
    """消息 → 每队信号分（-1 ~ +1）。"""

    name = "message"

    # 常见球队名别名 → 标准化名
    TEAM_ALIASES = {
        "manchester united": "manchester united", "man utd": "manchester united",
        "manchester city": "manchester city", "man city": "manchester city",
        "real madrid": "real madrid", "barcelona": "barcelona", "barça": "barcelona",
        "bayern": "bayern munich", "bayern munich": "bayern munich",
        "psg": "paris saint-germain", "paris saint-germain": "paris saint-germain", "paris st-germain": "paris saint-germain",
        "inter milan": "inter",
        "atletico": "atletico madrid", "atlético": "atletico madrid",
        "juventus": "juventus", "juve": "juventus",
        "dortmund": "borussia dortmund", "liverpool": "liverpool",
        "arsenal": "arsenal", "chelsea": "chelsea", "tottenham": "tottenham",
        "napoli": "napoli", "roma": "roma", "lazio": "lazio",
        "marseille": "marseille", "lyon": "lyon", "monaco": "monaco",
    }

    # 球员名 → 球队（新闻标题常提球员不提队名，如 "Vini Jr. blow"）
    # 只收录归属确定的球星，宁缺毋滥（错误归属会污染信号）
    PLAYER_MAP = {
        "vinicius": "real madrid", "vinícius": "real madrid", "vini jr": "real madrid",
        "mbappe": "real madrid", "mbappé": "real madrid", "bellingham": "real madrid",
        "rodrygo": "real madrid", "valverde": "real madrid", "courtois": "real madrid",
        "modric": "real madrid", "modrić": "real madrid",
        "rodri": "manchester city", "haaland": "manchester city", "de bruyne": "manchester city",
        "foden": "manchester city", "grealish": "manchester city", "doku": "manchester city",
        "saka": "arsenal", "odegaard": "arsenal", "ødegaard": "arsenal", "rice": "arsenal",
        "saliba": "arsenal", "martinelli": "arsenal", "havertz": "arsenal",
        "salah": "liverpool", "van dijk": "liverpool", "alisson": "liverpool",
        "szoboszlai": "liverpool", "mac allister": "liverpool", "nunez": "liverpool", "gakpo": "liverpool",
        "bruno fernandes": "manchester united", "rashford": "manchester united",
        "garnacho": "manchester united", "mainoo": "manchester united", "hojlund": "manchester united",
        "palmer": "chelsea", "enzo fernandez": "chelsea", "caicedo": "chelsea", "jackson": "chelsea",
        "isak": "newcastle united", "gordon": "newcastle united", "guimaraes": "newcastle united",
        "son": "tottenham", "kane": "bayern munich", "musiala": "bayern munich",
        "kimmich": "bayern munich", "neuer": "bayern munich", "sane": "bayern munich",
        "sané": "bayern munich", "gnabry": "bayern munich", "olise": "bayern munich",
        "yamal": "barcelona", "raphinha": "barcelona", "pedri": "barcelona", "gavi": "barcelona",
        "ter stegen": "barcelona", "lewandowski": "barcelona", "lewa": "barcelona",
        "griezmann": "atletico madrid", "julian alvarez": "atletico madrid",
        "julián álvarez": "atletico madrid", "lautaro": "inter", "lautaro martinez": "inter",
        "thuram": "inter", "barella": "inter", "calhanoglu": "inter",
        "leao": "ac milan", "leão": "ac milan", "pulisic": "ac milan",
        "theo hernandez": "ac milan", "maignan": "ac milan",
        "vlahovic": "juventus", "yildiz": "juventus", "chiesa": "juventus",
        "kvaratskhelia": "psg", "kvara": "psg", "dembele": "psg", "dembélé": "psg",
        "hakimi": "psg", "donnarumma": "psg",
        "wirtz": "bayer leverkusen", "grimaldo": "bayer leverkusen",
        "openda": "rb leipzig", "guirassy": "borussia dortmund", "brandt": "borussia dortmund",
    }

    def analyze(self, context: dict) -> dict:
        messages = context.get("messages", [])
        home = (context.get("homeTeam") or "").lower()
        away = (context.get("awayTeam") or "").lower()
        if not home or not away:
            return {"signals": {}, "mentions": 0}

        weights = self.rules.get("keywordWeights", {})
        scores = {home: 0.0, away: 0.0}
        mentions = {home: 0, away: 0}
        evidence = []

        for msg in messages:
            text = (msg.get("text", "") or "").lower()
            if not text:
                continue
            # 该消息涉及哪支球队（队名 + 别名 + 球员名）
            involved = set()
            home_terms = [home, self.TEAM_ALIASES.get(home, home)] + [p for p, t in self.PLAYER_MAP.items() if t == home]
            away_terms = [away, self.TEAM_ALIASES.get(away, away)] + [p for p, t in self.PLAYER_MAP.items() if t == away]
            if any(tok in text for tok in home_terms):
                involved.add(home)
            if any(tok in text for tok in away_terms):
                involved.add(away)
            if not involved:
                continue

            # 关键词打分
            score = 0.0
            matched = []
            for kw, w in weights.items():
                if kw in text:
                    score += w
                    matched.append(kw)
            for team in involved:
                scores[team] += score
                mentions[team] += 1
            if matched:
                evidence.append({
                    "text": msg.get("text", "")[:120],
                    "teams": list(involved),
                    "keywords": matched[:5],
                    "score": round(score, 3),
                })

        # 归一化到 [-1, 1]（tanh 压缩，防单条消息爆炸）
        signals = {}
        for team, s in scores.items():
            normalized = math.tanh(s / max(1, mentions[team] or 1))
            signals[team] = {"score": round(normalized, 3), "mentions": mentions[team]}

        return {"signals": signals, "mentions": sum(mentions.values()), "evidence": evidence[:10]}


# ── 比分预测器（融合层）──────────────────────────────────

class ScorePredictor:
    """融合赔率概率 + 消息信号 + 历史比分先验，输出预测。"""

    def __init__(self, rules: dict):
        self.rules = rules

    def predict(self, home, away, odds_result, msg_result):
        fusion = self.rules.get("fusion", {})
        w_odds = fusion.get("oddsWeight", 0.6)
        w_msg = fusion.get("messageWeight", 0.3)
        w_prior = fusion.get("priorWeight", 0.1)

        prob = odds_result.get("prob") if odds_result else None
        signals = msg_result.get("signals", {}) if msg_result else {}

        # 基础胜平负概率（默认 1/3 均匀）
        p_home = p_draw = p_away = 1 / 3
        if prob:
            p_home, p_draw, p_away = prob["home"], prob["draw"], prob["away"]

        # 消息信号调整
        h_sig = signals.get(home.lower(), {}).get("score", 0)
        a_sig = signals.get(away.lower(), {}).get("score", 0)
        adj = w_msg * (h_sig - a_sig) / 2  # 信号差映射到概率增量

        final_home = min(0.9, max(0.05, p_home * (1 - w_msg) + (p_home + adj) * w_msg + w_prior / 3))
        final_away = min(0.9, max(0.05, p_away * (1 - w_msg) + (p_away - adj) * w_msg + w_prior / 3))
        final_draw = 1 - final_home - final_away

        # 预测比分: 期望总进球按概率份额分配
        total_goals = 2.5  # 足球比赛均值
        share_home = final_home / max(0.01, final_home + final_away)
        share_away = 1 - share_home
        exp_home = total_goals * share_home + adj * 0.8
        exp_away = total_goals * share_away - adj * 0.8
        pred_score = f"{max(0, round(exp_home))}-{max(0, round(exp_away))}"

        # 置信度: 概率集中度 + 消息一致性
        top = max(final_home, final_draw, final_away)
        msg_conf = abs(h_sig - a_sig)
        confidence = min(0.95, 0.4 + top * 0.4 + msg_conf * 0.15)

        reasons = []
        if prob:
            reasons.append(f"赔率隐含概率: 主{prob['home']:.0%} 平{prob['draw']:.0%} 客{prob['away']:.0%}")
        if h_sig != 0 or a_sig != 0:
            reasons.append(f"消息信号: 主队{h_sig:+.2f} 客队{a_sig:+.2f}")
        if not reasons:
            reasons.append("无赔率与消息数据, 使用先验")

        return {
            "homeTeam": home, "awayTeam": away,
            "predictedScore": pred_score,
            "expectedGoals": {"home": round(exp_home, 2), "away": round(exp_away, 2)},
            "probabilities": {
                "home": round(final_home, 3), "draw": round(final_draw, 3), "away": round(final_away, 3)
            },
            "confidence": round(confidence, 3),
            "reasons": reasons,
            "messageEvidence": msg_result.get("evidence", []) if msg_result else [],
        }


# ── 分析器注册表 ─────────────────────────────────────────

ANALYSIS_REGISTRY = {
    "odds": OddsAnalyzer,
    "message": MessageAnalyzer,
}


# ── 泊松比分模型（波胆/大小球推导）──────────────────────

class ScoreModel:
    """从 1X2 隐含概率反推主客期望进球，生成波胆分布与大小球概率。
    标准方法：泊松分布 P(i,j) = Poisson(i;λh) × Poisson(j;λa)。"""

    MAX_GOALS = 6  # 截断 0-6 球

    def __init__(self):
        self.lam_h = 1.3
        self.lam_a = 1.1

    def fit(self, p_home, p_draw, p_away):
        """数值求解 λh/λa 使模型 1X2 概率最接近输入概率。"""
        if not p_home or not p_draw or not p_away:
            return self
        best_err, best = 1e9, (self.lam_h, self.lam_a)
        # 网格搜索: 总进球 1.8~3.2, 主队份额 0.35~0.75
        for total in [x * 0.1 for x in range(18, 33)]:
            for share in [x * 0.01 for x in range(35, 76)]:
                lh, la = total * share, total * (1 - share)
                ph, pd, pa = self._probs(lh, la)
                err = abs(ph - p_home) + abs(pd - p_draw) + abs(pa - p_away)
                if err < best_err:
                    best_err, best = err, (lh, la)
        self.lam_h, self.lam_a = best
        return self

    def _poisson(self, k, lam):
        return math.exp(-lam) * lam ** k / math.factorial(k)

    def _probs(self, lh=None, la=None):
        lh, la = lh or self.lam_h, la or self.lam_a
        # 网格计算 P(i,j) = Poisson(i;lh) × Poisson(j;la)
        n = self.MAX_GOALS
        m = [[self._poisson(i, lh) * self._poisson(j, la) for j in range(n)] for i in range(n)]
        p_home = sum(m[i][j] for i in range(n) for j in range(n) if i > j)
        p_draw = sum(m[i][i] for i in range(n))
        p_away = sum(m[i][j] for i in range(n) for j in range(n) if i < j)
        return p_home, p_draw, p_away

    def correct_scores(self, top_n=6):
        """返回 Top N 波胆: [{score: '1-0', prob: 0.12}]"""
        n = self.MAX_GOALS
        m = [[self._poisson(i, self.lam_h) * self._poisson(j, self.lam_a) for j in range(n)] for i in range(n)]
        entries = [{"score": f"{i}-{j}", "prob": round(m[i][j], 4)} for i in range(n) for j in range(n)]
        entries.sort(key=lambda x: -x["prob"])
        return entries[:top_n]

    def dc_scores(self, top_n=6, rho=-0.08):
        """Dixon-Coles 修正的波胆分布（修正 0-0/1-0/0-1/1-1 低比分依赖）。"""
        return dc_probs(self.lam_h, self.lam_a, rho=rho)[:top_n]

    def dc_1x2(self, rho=-0.08):
        """Dixon-Coles 修正的胜平负概率。"""
        entries = dc_probs(self.lam_h, self.lam_a, rho=rho)
        p_home = sum(e["prob"] for e in entries if int(e["score"].split("-")[0]) > int(e["score"].split("-")[1]))
        p_draw = sum(e["prob"] for e in entries if e["score"].split("-")[0] == e["score"].split("-")[1])
        p_away = sum(e["prob"] for e in entries if int(e["score"].split("-")[0]) < int(e["score"].split("-")[1]))
        return round(p_home, 4), round(p_draw, 4), round(p_away, 4)

    def over_under(self, line=2.5):
        """大小球: 返回 {over, under} 概率。"""
        n = self.MAX_GOALS
        m = [[self._poisson(i, self.lam_h) * self._poisson(j, self.lam_a) for j in range(n)] for i in range(n)]
        p_over = sum(m[i][j] for i in range(n) for j in range(n) if i + j > line)
        return {"line": line, "over": round(p_over, 3), "under": round(1 - p_over, 3)}

    def total_goals_dist(self):
        """总进球数概率分布: {'0': 0.08, '1': 0.2, ...}（0-6球+，7球及以上合并）"""
        n = self.MAX_GOALS
        m = [[self._poisson(i, self.lam_h) * self._poisson(j, self.lam_a) for j in range(n)] for i in range(n)]
        dist = {}
        for i in range(n):
            for j in range(n):
                t = i + j
                dist[t] = dist.get(t, 0) + m[i][j]
        out = {str(k): round(v, 4) for k, v in sorted(dist.items())}
        return out

    def btts_prob(self):
        """双方进球概率（BTTS）。"""
        n = self.MAX_GOALS
        both = sum(self._poisson(i, self.lam_h) * self._poisson(j, self.lam_a)
                   for i in range(1, n) for j in range(1, n))
        return round(both, 4)


# ── 文字分析生成（规则模板）──────────────────────────────

class AnalysisWriter:
    """基于赔率/波胆/大小球/消息信号生成中文文字分析。"""

    @staticmethod
    def generate(home, away, odds_result, msg_result, score_model, ou=None, spreads=None, standings=None, context=None):
        lines = []
        prob = odds_result.get("prob") if odds_result else None
        raw = odds_result.get("rawOdds") if odds_result else None

        # 0. 排名对比（若积分榜可用）
        if standings and standings.get("home") and standings.get("away"):
            h = standings["home"]
            a = standings["away"]
            diff = a["position"] - h["position"]
            pos_txt = f"主队{home}排名第{h['position']}（{h['points']}分/{h['playedGames']}场），客队{away}排名第{a['position']}（{a['points']}分/{a['playedGames']}场）"
            if diff >= 3:
                pos_txt += f"，主队排名领先 {diff} 位"
            elif diff <= -3:
                pos_txt += f"，客队排名领先 {abs(diff)} 位"
            else:
                pos_txt += "，排名接近"
            lines.append(f"【排名】{pos_txt}。")

        # 1. 胜负分析（含真实赔率）
        if prob:
            ph, pd, pa = prob["home"], prob["draw"], prob["away"]
            odds_str = ""
            if raw and all(raw.get(k) for k in ("home", "draw", "away")):
                odds_str = f"（赔率 {raw['home']} / {raw['draw']} / {raw['away']}）"
            if ph >= 0.55:
                verdict = f"市场高度看好主队{home}，主胜隐含概率 {ph:.0%}{odds_str}"
            elif pa >= 0.55:
                verdict = f"市场明显看好客队{away}，客胜隐含概率 {pa:.0%}{odds_str}"
            elif ph >= pa and ph - pa < 0.15:
                verdict = f"双方实力接近，主队{home}略占优（主胜 {ph:.0%} vs 客胜 {pa:.0%}）{odds_str}"
            else:
                verdict = f"比赛悬念较大，主胜 {ph:.0%} / 平 {pd:.0%} / 客胜 {pa:.0%}{odds_str}"
            lines.append(f"【胜负】{verdict}。")

        # 2. 让球分析
        if spreads and spreads.get("home") and spreads.get("away"):
            hp, ap = spreads["home"], spreads["away"]
            # The Odds API: home point 为负表示主队让球（如 -1.5）
            if hp["point"] < 0:
                line_txt = f"主队{home}让 {abs(hp['point'])} 球（盘口赔率 {hp['price']} / {ap['price']}）"
            elif ap["point"] < 0:
                line_txt = f"客队{away}让 {abs(ap['point'])} 球（盘口赔率 {hp['price']} / {ap['price']}）"
            else:
                line_txt = f"平手盘（主 {hp['point']} / 客 {ap['point']}，赔率 {hp['price']} / {ap['price']}）"
            lines.append(f"【让球】{line_txt}。")

        # 3. 波胆分析
        cs = score_model.correct_scores(3)
        if cs:
            cs_str = "、".join(f"{c['score']}（{c['prob']:.0%}）" for c in cs)
            lines.append(f"【波胆】泊松模型推算最可能比分：{cs_str}。")

        # 4. 大小球（真实赔率优先，否则泊松）
        if not ou:
            ou = score_model.over_under(2.5)
        if ou:
            direction = "大球" if ou["over"] >= 0.5 else "小球"
            price_str = ""
            if ou.get("overPrice") and ou.get("underPrice"):
                price_str = f"（赔率 大 {ou['overPrice']} / 小 {ou['underPrice']}）"
            lines.append(f"【大小球】{ou['line']} 球盘口：大球 {ou['over']:.0%} / 小球 {ou['under']:.0%}{price_str}，倾向{direction}。")

        # 5. 总进球分布 + 双方进球
        dist = score_model.total_goals_dist()
        if dist:
            dist_str = "、".join(f"{k}球 {v:.0%}" for k, v in list(dist.items())[:5])
            lines.append(f"【进球数】总进球概率：{dist_str}。")
        btts = score_model.btts_prob()
        if btts is not None:
            btts_txt = f"双方都进球概率 {btts:.0%}" + ("，倾向双方有球" if btts >= 0.55 else "，倾向至少一方零封")
            lines.append(f"【双方进球】{btts_txt}。")

        # 5. 消息信号
        if msg_result:
            signals = msg_result.get("signals", {})
            h_sig = signals.get(home.lower(), {}).get("score", 0)
            a_sig = signals.get(away.lower(), {}).get("score", 0)
            h_men = signals.get(home.lower(), {}).get("mentions", 0)
            a_men = signals.get(away.lower(), {}).get("mentions", 0)
            if h_men or a_men:
                parts = []
                if h_men:
                    parts.append(f"{home}信号 {h_sig:+.2f}（{h_men}条提及）")
                if a_men:
                    parts.append(f"{away}信号 {a_sig:+.2f}（{a_men}条提及）")
                lines.append(f"【消息面】{'，'.join(parts)}。")
            ev = msg_result.get("evidence", [])[:2]
            for e in ev:
                lines.append(f"📰 {e['text']}（{'、'.join(e['keywords'][:3])}）")

        # 5.5 独立模型 + 价值信号
        if context and context.get("valuePicks"):
            vp = context["valuePicks"]
            if vp:
                first = vp[0]
                lines.append(f"【价值】独立模型(Elo)与市场存在分歧：{first['label']} 模型 {first['modelProb']:.0%} vs 盘口 {first['oddsProb']:.0%}（差 {first['edge']:+.0%}），值得关注。")

        # 6. 综合结论
        if prob:
            winner = home if prob["home"] >= prob["away"] else away
            cs0 = cs[0]["score"] if cs else ""
            lines.append(f"【结论】综合赔率与消息，{winner}不败概率更高，关注波胆 {cs0} 方向。")

        if not lines:
            lines.append("暂无足够数据生成分析。")
        return "\n".join(lines)


# ── 分析器注册表（保持兼容）──────────────────────────────


# ── 主流程 ───────────────────────────────────────────────

def load_odds():
    """加载 data/odds/*.json, 返回 {league: [matches]}"""
    result = {}
    if not ODDS_DIR.exists():
        return result
    for f in ODDS_DIR.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        result[data.get("league", f.stem)] = data.get("matches", [])
    return result


def load_messages():
    path = DATA_DIR / "messages.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("messages", [])


def load_standings():
    """加载 data/standings.json → {league_code: [rows]}。"""
    path = DATA_DIR / "standings.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("standings", {})


def norm_team(s):
    """队名归一化（与前端一致）。"""
    import re
    return re.sub(r"\b(fc|afc|cf|sc)\b", "", (s or "").lower()).replace("&", "").strip()


FORM_LAST = 5  # 近 5 场状态


def load_form():
    """每队最近 N 场场均积分（0~3）。数据源 data/season_2025.json（上赛季末段状态，
    赛季开打后由新赛果自然延续）。抓取顺序为从最新往回翻，故数组头部即最近场次。"""
    path = DATA_DIR / "season_2025.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    form = {}
    per_league = {}
    for m in data.get("matches", []):
        if m.get("homeGoals") is None or m.get("awayGoals") is None:
            continue
        per_league.setdefault(m["league"], []).append(m)
    for ms in per_league.values():
        for m in ms[:FORM_LAST]:
            for side, gf, ga in (("home", "homeGoals", "awayGoals"), ("away", "awayGoals", "homeGoals")):
                pts = 3 if m[gf] > m[ga] else (1 if m[gf] == m[ga] else 0)
                t = norm_team(m["homeTeam" if side == "home" else "awayTeam"])
                acc = form.setdefault(t, [0.0, 0])
                acc[0] += pts
                acc[1] += 1
    return {t: round(s / n, 3) for t, (s, n) in form.items() if n}


# The Odds API 联赛代码 → Football-Data.org 联赛代码（积分榜用）
LEAGUE_ALIAS = {
    "CH": "ELC", "ED": "DED", "BDF": "BPL", "JLG": "J1", "KL1": "KLE",
}


def main():
    with open(CONFIG_DIR / "prediction_rules.json", "r", encoding="utf-8") as f:
        rules = json.load(f)

    analyzers = [cls(rules) for cls in ANALYSIS_REGISTRY.values()]
    print(f"分析器: {[a.name for a in analyzers]}")

    odds_by_league = load_odds()
    messages = load_messages()
    standings_by_league = load_standings()
    form = load_form()
    print(f"赔率联赛: {list(odds_by_league.keys())}, 消息: {len(messages)} 条, 积分榜联赛: {len(standings_by_league)}, 近5场form球队: {len(form)}")

    # Elo 独立模型：积分榜初始化 + 本赛季赛果迭代
    elo = EloModel()
    elo.init_from_standings(standings_by_league)
    try:
        with open(DATA_DIR / "fixtures.json", encoding="utf-8") as f:
            fixtures_all = json.load(f).get("matches", [])
        finished = [m for m in fixtures_all if m.get("status") == "FINISHED"]
        elo.update(finished)
        # 上赛季 1752 场迭代（与回测同源）：消除 Football-Data 积分榜与赔率源
        # 球队构成不一致导致的无历史评级（国米/马竞/里昂等被误判为 1500 初始）
        season_path = DATA_DIR / "season_2025.json"
        if season_path.exists():
            try:
                with open(season_path, encoding="utf-8") as f:
                    season = json.load(f)
                ms_by_lg = {}
                for sm in season.get("matches", []):
                    if sm.get("homeGoals") is None or sm.get("awayGoals") is None:
                        continue
                    ms_by_lg.setdefault(sm["league"], []).append(sm)
                season_total = 0
                for lg_ms in ms_by_lg.values():
                    # 抓取顺序 = 从最新往回 → 反转即时间正序
                    elo.update([{
                        "utcDate": sm.get("utcDate", ""),
                        "homeTeam": {"name": sm["homeTeam"]},
                        "awayTeam": {"name": sm["awayTeam"]},
                        "score": {"fullTime": {"home": sm["homeGoals"], "away": sm["awayGoals"]}},
                    } for sm in reversed(lg_ms)])
                    season_total += len(lg_ms)
                print(f"Elo: {len(elo.ratings)} 队, 已用 {len(finished)} 场本赛季 + {season_total} 场上赛季赛果迭代")
            except Exception as e:
                print(f"Elo 上赛季迭代失败: {e}")
        else:
            print(f"Elo: {len(elo.ratings)} 队, 已用 {len(finished)} 场赛果迭代")
    except Exception as e:
        print(f"Elo 赛果迭代跳过: {e}")

    predictor = ScorePredictor(rules)
    predictions = []

    def find_standing(league_code, team_name):
        """在积分榜中按队名模糊匹配排名。"""
        table = standings_by_league.get(league_code) or standings_by_league.get(LEAGUE_ALIAS.get(league_code, league_code))
        if not table:
            return None
        n = norm_team(team_name)
        for row in table:
            if norm_team(row["team"]) == n or norm_team(row.get("shortName", "")) == n:
                return {"position": row["position"], "points": row["points"],
                        "playedGames": row["playedGames"], "goalDifference": row["goalDifference"]}
        # 首词匹配
        first = n.split()[0] if n else ""
        for row in table:
            if first and norm_team(row["team"]).startswith(first):
                return {"position": row["position"], "points": row["points"],
                        "playedGames": row["playedGames"], "goalDifference": row["goalDifference"]}
        return None

    for league, matches in odds_by_league.items():
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

            # ── Elo 独立概率（近 5 场 form 微调，幅度 ≤±5%）──
            ep_home, ep_draw, ep_away = elo.predict(home, away)
            fh, fa = form.get(norm_team(home)), form.get(norm_team(away))
            if fh is not None and fa is not None:
                adj = (fh - fa) * 0.02
                ep_home += adj
                ep_away -= adj
                _s = ep_home + ep_draw + ep_away
                ep_home, ep_draw, ep_away = ep_home / _s, ep_draw / _s, ep_away / _s
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

            # ── 多模型分歧：Elo vs Dixon-Coles vs 市场（方向不一致 = 不碰）──
            dc_home, dc_draw, dc_away = score_model._probs()
            dc12 = {"home": dc_home, "draw": dc_draw, "away": dc_away}

            def _dir(probs):
                return max(probs, key=probs.get)

            dir_elo = _dir(pred_elo)
            dir_dc = _dir(dc12)
            dir_mkt = _dir(market_prob) if market_prob else dir_elo
            diverge = len({dir_elo, dir_dc, dir_mkt}) >= 2
            pred_dirs = {"elo": dir_elo, "dc": dir_dc, "market": dir_mkt}

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
            pred["prevOdds"] = prev_h2h if isinstance(prev_h2h, dict) else None

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
            pred["dcProb"] = {k: round(v, 3) for k, v in dc12.items()}
            pred["diverge"] = diverge
            pred["dirs"] = pred_dirs
            pred["form"] = {"home": fh, "away": fa}
            pred["eloRatings"] = {
                "home": round(elo.get_rating(home), 0),
                "away": round(elo.get_rating(away), 0),
            }
            pred["modelProbs"] = {
                "home": round(f_home, 3), "draw": round(f_draw, 3), "away": round(f_away, 3)
            }
            value_picks = []
            reverse_picks = []  # 模型 vs 市场对立（负 edge）：提示不碰，非价值
            if market_prob:
                sorted_probs = sorted(pred_elo.values(), reverse=True)
                model_conf = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
                for k, label in (("home", "主胜"), ("draw", "平局"), ("away", "客胜")):
                    diff = pred_elo[k] - market_prob[k]
                    if diff >= 0.10:  # 正 edge：模型比市场更看好 = 价值信号
                        level = "gold" if diff >= 0.15 and model_conf >= 0.15 else "watch"
                        value_picks.append({
                            "side": k, "label": label,
                            "modelProb": round(pred_elo[k], 3),
                            "oddsProb": round(market_prob[k], 3),
                            "edge": round(diff, 3),
                            "level": level,
                            "modelConf": round(model_conf, 3),
                        })
                    elif diff <= -0.10:  # 负 edge：模型比市场更不看好 = 反向信号（提示不碰）
                        reverse_picks.append({
                            "side": k, "label": label,
                            "modelProb": round(pred_elo[k], 3),
                            "oddsProb": round(market_prob[k], 3),
                            "edge": round(diff, 3),
                        })
            pred["valuePicks"] = value_picks
            pred["reversePicks"] = reverse_picks
            pred["signalLevel"] = "gold" if any(v["level"] == "gold" for v in value_picks) else ("watch" if value_picks else "none")

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
            pred["analysis"] = AnalysisWriter.generate(home, away, odds_result, msg_result, score_model, ou, pred["spreads"], pred["standings"], {"valuePicks": value_picks})
            predictions.append(pred)
    # 模型校准信息：赛季已进行轮次
    season_info = {"seasonStarted": False, "round": 0, "finishedMatches": 0}
    try:
        with open(DATA_DIR / "fixtures.json", encoding="utf-8") as f:
            fx = json.load(f).get("matches", [])
        fin = [m for m in fx if m.get("status") == "FINISHED"]
        season_info["finishedMatches"] = len(fin)
        if fin:
            season_info["seasonStarted"] = True
            matchdays = [m.get("matchday") for m in fin if m.get("matchday")]
            if matchdays:
                season_info["round"] = max(matchdays)
    except Exception:
        pass

    out = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(predictions),
        "rulesVersion": "1.1",
        "modelInfo": {
            "name": "Elo + Dixon-Coles + 赔率融合",
            "eloTeams": len(elo.ratings),
            "season": season_info,
        },
        "predictions": predictions,
        "disclaimer": "本结果仅用于个人数据分析与研究, 不构成任何投注建议",
    }

    # ── 预测战绩：存档 + 赛果对比 + 成功率统计 ──
    stats = evaluate_predictions(predictions)
    out["stats"] = stats

    with open(DATA_DIR / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 摘要
    notable = [p for p in predictions if p["confidence"] >= rules.get("confidenceMin", 0.4)]
    print(f"\n总计预测: {len(predictions)} 场, 高置信(≥{rules.get('confidenceMin')}): {len(notable)} 场")
    if stats and stats.get("finished"):
        print(f"战绩: 已结算 {stats['finished']} 场, 比分精确命中 {stats['exactHit']} ({stats['exactRate']:.0%}), 胜负方向命中 {stats['outcomeHit']} ({stats['outcomeRate']:.0%})")
    for p in notable[:5]:
        print(f"  {p['homeTeam']} vs {p['awayTeam']}: {p['predictedScore']} (置信度 {p['confidence']:.0%})")
    print(f"\n输出: data/predictions.json")
    return 0


def outcome_of(score_str):
    """从比分字符串推断胜负方向: home/draw/away。"""
    try:
        h, a = score_str.split("-")
        h, a = int(h), int(a)
        if h > a:
            return "home"
        if h < a:
            return "away"
        return "draw"
    except (ValueError, AttributeError):
        return None


def evaluate_predictions(predictions):
    """
    预测战绩评估：
    1. 存档预测到 data/prediction_history.json（去重，保留最早预测）
    2. 从 fixtures.json 读取已完赛赛果，与存档预测对比
    3. 统计：比分精确命中率 / 胜负方向命中率
    """
    history_path = DATA_DIR / "prediction_history.json"
    hist = {"predictions": [], "results": []}
    if history_path.exists():
        try:
            hist = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            hist = {"predictions": [], "results": []}

    # 1. 合并当前预测（去重：同队同日期只保留最早一条）
    seen = set()
    for p in hist["predictions"]:
        seen.add((norm_team(p["homeTeam"]), norm_team(p["awayTeam"]), p.get("kickoff", "")[:10]))
    for p in predictions:
        key = (norm_team(p["homeTeam"]), norm_team(p["awayTeam"]), (p.get("kickoff") or "")[:10])
        if key in seen:
            continue
        seen.add(key)
        hist["predictions"].append({
            "homeTeam": p["homeTeam"], "awayTeam": p["awayTeam"],
            "kickoff": p.get("kickoff", ""),
            "predictedScore": p["predictedScore"],
            "confidence": p.get("confidence"),
            "valuePicks": p.get("valuePicks") or [],
            "signalLevel": p.get("signalLevel", "none"),
            "predictedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    # 2. 从赛程读取已完赛结果
    fixtures_path = DATA_DIR / "fixtures.json"
    finished = []
    if fixtures_path.exists():
        try:
            fix = json.loads(fixtures_path.read_text(encoding="utf-8"))
            for m in fix.get("matches", []):
                if m.get("status") == "FINISHED" and m.get("score", {}).get("fullTime"):
                    ft = m["score"]["fullTime"]
                    if ft.get("home") is not None and ft.get("away") is not None:
                        finished.append({
                            "homeTeam": m["homeTeam"]["name"],
                            "awayTeam": m["awayTeam"]["name"],
                            "kickoff": m.get("utcDate", ""),
                            "actualScore": f"{ft['home']}-{ft['away']}",
                        })
        except Exception:
            pass

    # 3. 匹配结算
    # 匹配规则: 队名归一化精确相等 → 首词相等 → 包含关系（日期仅作辅助）
    def team_match(a, b):
        if not a or not b:
            return False
        if a == b:
            return True
        if a.split()[0] == b.split()[0]:
            return True
        return a in b or b in a

    result_keys = {}
    for r in finished:
        result_keys[(norm_team(r["homeTeam"]), norm_team(r["awayTeam"]), r["kickoff"][:10])] = r
    evaluated = 0
    exact_hit = 0
    outcome_hit = 0
    value_total = 0
    value_hit = 0
    for p in hist["predictions"]:
        if p.get("actualScore"):
            evaluated += 1
            if p.get("hitExact"):
                exact_hit += 1
            if p.get("hitOutcome"):
                outcome_hit += 1
            continue
        key = (norm_team(p["homeTeam"]), norm_team(p["awayTeam"]), p.get("kickoff", "")[:10])
        r = result_keys.get(key)
        if not r:
            # 模糊匹配：仅按队名（首词/包含），日期放宽
            for rk, rv in result_keys.items():
                if team_match(rk[0], key[0]) and team_match(rk[1], key[1]):
                    r = rv
                    break
        if r:
            p["actualScore"] = r["actualScore"]
            p["evaluatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            p["hitExact"] = (p["predictedScore"] == r["actualScore"])
            p["hitOutcome"] = (outcome_of(p["predictedScore"]) == outcome_of(r["actualScore"]))
            evaluated += 1
            if p["hitExact"]:
                exact_hit += 1
            if p["hitOutcome"]:
                outcome_hit += 1
            # 价值标记方向命中（bestOutcome）
            vp = p.get("valuePicks") or []
            if vp:
                value_total += 1
                best = max(vp, key=lambda v: abs(v.get("edge", 0)))
                if best.get("side") == outcome_of(r["actualScore"]):
                    value_hit += 1
            hist["results"].append({
                "homeTeam": p["homeTeam"], "awayTeam": p["awayTeam"],
                "kickoff": p.get("kickoff", ""),
                "predictedScore": p["predictedScore"],
                "actualScore": r["actualScore"],
                "hitExact": p["hitExact"],
                "hitOutcome": p["hitOutcome"],
            })

    # 4. 统计
    stats = {
        "total": len(hist["predictions"]),
        "finished": evaluated,
        "exactHit": exact_hit,
        "exactRate": round(exact_hit / evaluated, 4) if evaluated else 0,
        "outcomeHit": outcome_hit,
        "outcomeRate": round(outcome_hit / evaluated, 4) if evaluated else 0,
        "valueTotal": value_total,
        "valueHit": value_hit,
        "valueRate": round(value_hit / value_total, 4) if value_total else 0,
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    hist["stats"] = stats
    history_path.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


if __name__ == "__main__":
    sys.exit(main())
