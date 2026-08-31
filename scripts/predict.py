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

# ── AI 研判费用控制 ──
# 只对开赛前 72h 内的比赛调用 AI（覆盖本周末+周中窗口），更远的沿用旧研判。
AI_WINDOW_HOURS = 72
# 临场期：距开赛 <24h 时每次刷新都重新调用 AI（赛前伤停/赔率/阵容变动密集，必须及时反映）。
AI_LATE_HOURS = 24
# 远期（≥24h 且 ≤72h）同场 12h 内已有研判则不重复调用（CI 高频刷新时避免重复扣费）。
AI_COOLDOWN_HOURS = 12

import argparse
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

from elo import EloModel, dc_probs, XgModel, norm_team
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

    def _word_hit(self, term: str, text: str) -> bool:
        """词级匹配：term 作为独立词/短语出现在 text 中（避免 'angers' 命中 'rangers'）。"""
        term = term.strip().lower()
        if not term:
            return False
        # 多词短语：要求短语作为连续子串且边界为词边界（'man city' 不命中 'man cityzens'）
        if " " in term:
            import re
            return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None
        # 单词：\b 词边界匹配（'angers' 不会命中 'rangers'）
        import re
        return re.search(r"\b" + re.escape(term) + r"\b", text) is not None

    def _team_terms(self, team: str) -> list:
        """生成队名的全部匹配词：全称 + 双向别名 + 球员名 + 去后缀核心名。"""
        team = (team or "").lower()
        if not team:
            return []
        terms = [team]
        # 反向别名：全称→简称（odds 名 → 新闻常用名）
        for alias, full in self.TEAM_ALIASES.items():
            if full == team:
                terms.append(alias)
        # 正向别名（已有）：当 team 是简称时（如 "atletico"），补上对应全称（"atletico madrid"）
        if self.TEAM_ALIASES.get(team):
            terms.append(self.TEAM_ALIASES[team])
        else:
            # 反向补全称：team 是某别名的 key（简称），新闻标题常写全称
            for alias, full in self.TEAM_ALIASES.items():
                if alias == team and full != team:
                    terms.append(full)
        # 球员名（该队球员）
        terms += [p for p, t in self.PLAYER_MAP.items() if t == team]
        # 去掉城市/地名后缀的核心名（"brighton and hove albion" → "brighton"）
        core = team.replace(" and hove albion", "").replace(" hotspur", "").replace(" united", "")
        core = core.replace(" city", "").replace(" fc", "").replace(" cf", "").replace(" ac", "")
        if core != team:
            terms.append(core)
        # 去重，保持顺序
        seen, out = set(), []
        for t in terms:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

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
            # 双方向别名：全称→简称（"brighton and hove albion" 能匹配标题里的 "brighton"）
            involved = set()
            home_terms = self._team_terms(home)
            away_terms = self._team_terms(away)
            if any(self._word_hit(tok, text) for tok in home_terms):
                involved.add(home)
            if any(self._word_hit(tok, text) for tok in away_terms):
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
                # 附上摘要/全文（含伤停/状态细节），供 AI 研判参考；标题限 120 字符
                meta = msg.get("metadata") or {}
                summary = (meta.get("summary") or "").strip()[:300]
                content = (meta.get("content") or "").strip()[:400]
                evidence.append({
                    "text": msg.get("text", "")[:120],
                    "summary": summary,
                    "content": content,
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
        msg_conf_w = fusion.get("messageConfWeight", 0.1)

        prob = odds_result.get("prob") if odds_result else None
        signals = msg_result.get("signals", {}) if msg_result else {}

        # 基础胜平负概率（默认 1/3 均匀）
        p_home = p_draw = p_away = 1 / 3
        if prob:
            p_home, p_draw, p_away = prob["home"], prob["draw"], prob["away"]

        # 消息信号不直接改概率（量小滞后，加减概率只会引入噪声），
        # 仅作为置信度参考与展示字段；融合主锚是去水后的市场概率。
        h_sig = signals.get(home.lower(), {}).get("score", 0)
        a_sig = signals.get(away.lower(), {}).get("score", 0)
        final_home, final_draw, final_away = p_home, p_draw, p_away

        # 预测比分: 期望总进球按概率份额分配
        total_goals = 2.5  # 足球比赛均值
        share_home = final_home / max(0.01, final_home + final_away)
        share_away = 1 - share_home
        exp_home = total_goals * share_home
        exp_away = total_goals * share_away
        pred_score = f"{max(0, round(exp_home))}-{max(0, round(exp_away))}"

        # 置信度: 概率集中度 + 消息一致性（消息只影响置信度，权重低）
        top = max(final_home, final_draw, final_away)
        msg_conf = abs(h_sig - a_sig)
        confidence = min(0.95, 0.4 + top * 0.4 + msg_conf * msg_conf_w)
        # 模型进化：若 calibrate 配置了高置信折扣（evolve.py 自适应），应用于置信度
        disc = self.rules.get("confidenceDiscount") if isinstance(self.rules, dict) else None
        if isinstance(disc, (int, float)) and 0.5 < disc < 1.0 and confidence >= 0.65:
            confidence *= disc
            confidence = round(confidence, 4)

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
            "msgSignals": {"home": round(h_sig, 3), "away": round(a_sig, 3)},
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
        self.score_prior = {}   # 常见比分先验(scorePrior): { '1-0': 0.1, ... }，可选

    def fit(self, p_home, p_draw, p_away, xg_hint=None, min_total_goals=None):
        """数值求解 λh/λa 使模型 1X2 概率最接近输入概率。
        xg_hint: (lam_h, lam_a) 元组，来自 xG 攻防强度。有 hint 时缩小搜索范围到 ±0.5。
        min_total_goals: 联赛近期场均总进球（如 2.8）。搜索时 λh+λa 不低于该值的 85%，
        避免模型反推出过低 λ（实际联赛进球普遍被低估，导致波胆/大小球偏小）。"""
        if not p_home or not p_draw or not p_away:
            return self
        best_err, best = 1e9, (self.lam_h, self.lam_a)
        min_total = (min_total_goals or 0) * 0.85
        if xg_hint and xg_hint[0] and xg_hint[1]:
            # xG 引导搜索：以 xG λ 为中心，±0.5 范围，步长 0.05
            ch, ca = xg_hint
            for dh in [x * 0.05 for x in range(-10, 11)]:
                for da in [x * 0.05 for x in range(-10, 11)]:
                    lh, la = max(0.3, ch + dh), max(0.3, ca + da)
                    if lh + la < min_total:
                        continue
                    ph, pd, pa = self._probs(lh, la)
                    err = abs(ph - p_home) + abs(pd - p_draw) + abs(pa - p_away)
                    if err < best_err:
                        best_err, best = err, (lh, la)
        else:
            # 原始全范围网格搜索: 总进球 1.4~3.6, 主队份额 0.25~0.80
            for total in [x * 0.1 for x in range(14, 37)]:
                for share in [x * 0.01 for x in range(25, 81)]:
                    lh, la = total * share, total * (1 - share)
                    if lh + la < min_total:
                        continue
                    ph, pd, pa = self._probs(lh, la)
                    err = abs(ph - p_home) + abs(pd - p_draw) + abs(pa - p_away)
                    if err < best_err:
                        best_err, best = err, (lh, la)
        self.lam_h, self.lam_a = best
        return self

    def _poisson(self, k, lam):
        return math.exp(-lam) * lam ** k / math.factorial(k)

    def _cdf(self, k, lam):
        """泊松累积分布 P(X <= k)。"""
        return sum(self._poisson(i, lam) for i in range(k + 1))

    def _probs(self, lh=None, la=None):
        lh, la = lh or self.lam_h, la or self.lam_a
        # 网格计算 P(i,j) = Poisson(i;lh) × Poisson(j;la)
        n = self.MAX_GOALS
        m = [[self._poisson(i, lh) * self._poisson(j, la) for j in range(n)] for i in range(n)]
        p_home = sum(m[i][j] for i in range(n) for j in range(n) if i > j)
        p_draw = sum(m[i][i] for i in range(n))
        p_away = sum(m[i][j] for i in range(n) for j in range(n) if i < j)
        # 补尾概率（任一队 ≥ n 球）：主队≥n 且客队<n → 必主胜；反之必客胜；
        # 双方都≥n 的平局概率可忽略，按 λ 比例分摊给主/客胜。
        cdf_h, cdf_a = self._cdf(n - 1, lh), self._cdf(n - 1, la)
        tail_home = (1 - cdf_h) * cdf_a
        tail_away = cdf_h * (1 - cdf_a)
        tail_both = (1 - cdf_h) * (1 - cdf_a)
        share_h = lh / (lh + la)
        p_home += tail_home + tail_both * share_h
        p_away += tail_away + tail_both * (1 - share_h)
        return p_home, p_draw, p_away

    def _apply_prior(self, entries):
        """把常见比分先验(scorePrior)与泊松/DC 波胆做软融合。
        w(s) = prior.get(s, 最小先验) / 最大先验；再 归一化。
        令常见比分(1-0/2-1 等)按其在真实分布中的频率获得温和上浮，
        未出现在先验表的冷门比分获得与"最小先验"同级的相对权重，不会被清零。
        有 prior 时覆盖输入 entries 的 prob，并保持 total=1。"""
        if not self.score_prior or not entries:
            return entries
        try:
            max_p = max(self.score_prior.values())
            min_p = min(self.score_prior.values())
        except ValueError:
            return entries
        if max_p <= 0:
            return entries
        tot = 0.0
        for e in entries:
            w = self.score_prior.get(e["score"], min_p) / max_p
            e = e.copy()
            e["prob"] = e["prob"] * w
            tot += e["prob"]
        if tot <= 0:
            return entries
        return [{**e, "prob": round(e["prob"] / tot, 4)} for e in entries]

    def correct_scores(self, top_n=6):
        """返回 Top N 波胆: [{score: '1-0', prob: 0.12}]"""
        n = self.MAX_GOALS
        m = [[self._poisson(i, self.lam_h) * self._poisson(j, self.lam_a) for j in range(n)] for i in range(n)]
        entries = [{"score": f"{i}-{j}", "prob": round(m[i][j], 4)} for i in range(n) for j in range(n)]
        entries.sort(key=lambda x: -x["prob"])
        return entries[:top_n]

    def dc_scores(self, top_n=6, rho=-0.08):
        """Dixon-Coles 修正的波胆分布（修正 0-0/1-0/0-1/1-1 低比分依赖）。
        若配置了 scorePrior，则按常见比分先验做软融合后再取 Top N。"""
        entries = dc_probs(self.lam_h, self.lam_a, rho=rho)
        entries = self._apply_prior(entries)
        return entries[:top_n]

    def dc_1x2(self, rho=-0.08):
        """Dixon-Coles 修正的胜平负概率。"""
        entries = dc_probs(self.lam_h, self.lam_a, rho=rho)
        p_home = sum(e["prob"] for e in entries if int(e["score"].split("-")[0]) > int(e["score"].split("-")[1]))
        p_draw = sum(e["prob"] for e in entries if e["score"].split("-")[0] == e["score"].split("-")[1])
        p_away = sum(e["prob"] for e in entries if int(e["score"].split("-")[0]) < int(e["score"].split("-")[1]))
        return round(p_home, 4), round(p_draw, 4), round(p_away, 4)

    def over_under(self, line=2.5):
        """大小球: 返回 {over, under} 概率。
        有意用普通泊松而非 DC 修正：DC 只调整 0-0/1-0/0-1/1-1 低比分相关性，
        对大小球边际概率影响 <1%，保持普通泊松避免过度拟合。"""
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

    def cover_prob(self, point):
        """主队让球 point（负=让球）时的赢盘概率 P(主队进球差 > -point)。
        整球盘（point 为整数）含走盘：返回 (win, push)，push=净胜恰等于 -point。
        与 over_under 同理，有意用普通泊松（DC 对让球边际影响 <1%）。"""
        n = self.MAX_GOALS
        m = [[self._poisson(i, self.lam_h) * self._poisson(j, self.lam_a) for j in range(n)] for i in range(n)]
        if point is not None and float(point).is_integer():
            win = sum(m[i][j] for i in range(n) for j in range(n) if i - j > -point)
            push = sum(m[i][j] for i in range(n) for j in range(n) if i - j == -point)
            return win, push
        win = sum(m[i][j] for i in range(n) for j in range(n) if i - j > -point)
        return win, 0.0

    def btts_prob(self):
        """双方进球概率（BTTS）。
        与 over_under 同理，有意用普通泊松（DC 对 BTTS 边际影响 <1%）。"""
        n = self.MAX_GOALS
        both = sum(self._poisson(i, self.lam_h) * self._poisson(j, self.lam_a)
                   for i in range(1, n) for j in range(1, n))
        return round(both, 4)


# ── 文字分析生成（规则模板）──────────────────────────────

class AnalysisWriter:
    """基于赔率/波胆/大小球/消息信号生成中文文字分析。"""

    @staticmethod
    def generate(home, away, odds_result, msg_result, score_model, ou=None, spreads=None, standings=None, context=None, ou_model=None):
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
        # 注意区分两个口径：ou=市场盘口隐含概率，ou_model=模型泊松判断。
        # 二者方向相反时以模型为准，并明示分歧，避免"市场看大/模型看小"被误解。
        if not ou:
            ou = score_model.over_under(2.5)
        if ou:
            price_str = ""
            if ou.get("overPrice") and ou.get("underPrice"):
                price_str = f"（赔率 大 {ou['overPrice']} / 小 {ou['underPrice']}）"
            mkt_dir = "大球" if ou["over"] >= 0.5 else "小球"
            if ou_model and ou_model.get("over") is not None:
                m_over, m_under = ou_model["over"], ou_model["under"]
                edge = ou_model.get("edge")
                edge_str = f"，edge {edge:+.0%}" if edge is not None else ""
                mod_dir = "大球" if m_over >= 0.5 else "小球"
                if mkt_dir == mod_dir:
                    lines.append(f"【大小球】{ou['line']} 球盘口：大球 {ou['over']:.0%} / 小球 {ou['under']:.0%}{price_str}；模型大球 {m_over:.0%} / 小球 {m_under:.0%}{edge_str}，市场与模型均倾向{mkt_dir}。")
                else:
                    lines.append(f"【大小球】{ou['line']} 球盘口：市场 大球 {ou['over']:.0%} / 小球 {ou['under']:.0%}{price_str}；模型 大球 {m_over:.0%} / 小球 {m_under:.0%}{edge_str}。市场倾向{mkt_dir}，模型倾向{mod_dir}（以模型为准）。")
            else:
                lines.append(f"【大小球】{ou['line']} 球盘口：大球 {ou['over']:.0%} / 小球 {ou['under']:.0%}{price_str}，倾向{mkt_dir}。")

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

def atomic_write(path, data):
    """原子写 JSON：先写临时文件再 rename，避免写入中断产生损坏文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def load_odds():
    """加载 data/odds/*.json, 返回 {league: [matches]}。
    单个文件损坏/解析失败 → 跳过该联赛（不中断整批预测）。"""
    result = {}
    if not ODDS_DIR.exists():
        return result
    for f in ODDS_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            result[data.get("league", f.stem)] = data.get("matches", [])
        except Exception as e:
            print(f"  ⚠ 跳过损坏赔率文件 {f.name}: {e}")
    return result


def load_messages():
    path = DATA_DIR / "messages.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("messages", [])
    except Exception as e:
        print(f"  ⚠ 消息数据解析失败: {e}")
        return []


def load_standings():
    """加载 data/standings.json → {league_code: [rows]}。"""
    path = DATA_DIR / "standings.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("standings", {})
    except Exception as e:
        print(f"  ⚠ 积分榜解析失败: {e}")
        return {}


def norm_team(s):
    """队名归一化（统一使用 elo.py 的 ALIASES 体系，消除双套归一化不一致）。"""
    return __import__('elo').norm_team(s)


FORM_LAST = 5  # 近 5 场状态


def load_form(min_kickoff=""):
    """每队最近 N 场场均积分（0~3），主/客场分开计算。
    数据源 data/season_2025.json。
    只使用早于预测窗口开赛日的赛果（防未来函数），并按时间排序取每队最近 FORM_LAST 场。
    返回 { team: {"home": 场均主场积分, "away": 场均客场积分, "overall": 场均总积分} }"""
    path = DATA_DIR / "season_2025.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    team_matches = {}
    for m in data.get("matches", []):
        if m.get("homeGoals") is None or m.get("awayGoals") is None:
            continue
        # 未来赛果（>= 最早预测开赛日）不参与 form 计算
        if min_kickoff and (m.get("utcDate") or "") >= min_kickoff:
            continue
        for side in ("home", "away"):
            t = norm_team(m[f"{side}Team"])
            team_matches.setdefault(t, []).append({"m": m, "side": side})
    form = {}
    for t, ms in team_matches.items():
        ms = sorted(ms, key=lambda x: x["m"].get("utcDate", ""))
        # 最近 N 场总体
        recent = ms[-FORM_LAST:]
        if not recent:
            continue
        def avg_pts(items):
            if not items:
                return None
            total = 0.0
            for it in items:
                m = it["m"]
                gf, ga = m["homeGoals"], m["awayGoals"]
                pts = 3 if gf > ga else (1 if gf == ga else 0)
                total += pts
            return round(total / len(items), 3)
        # 主场只看主队身份的最近 N 场；客场只看客队身份
        home_ms = [it for it in recent if it["side"] == "home"][-FORM_LAST:]
        away_ms = [it for it in recent if it["side"] == "away"][-FORM_LAST:]
        form[t] = {
            "home": avg_pts(home_ms),
            "away": avg_pts(away_ms),
            "overall": avg_pts(recent),
        }
    return form


def _league_avg_goals(min_kickoff=""):
    """各联赛历史场均总进球：{league: 场均总进球}。
    数据源 data/season_2025.json（与 load_form/Elo 训练同源同口径）。
    只统计早于预测窗口的赛果（防未来函数），球队数≥5 才纳入（避免单场噪声）。"""
    path = DATA_DIR / "season_2025.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    goals = {}
    for m in data.get("matches", []):
        hg, ag = m.get("homeGoals"), m.get("awayGoals")
        if hg is None or ag is None:
            continue
        if min_kickoff and (m.get("utcDate") or "") >= min_kickoff:
            continue
        goals.setdefault(m.get("league", ""), []).append(hg + ag)
    out = {}
    for lg, gs in goals.items():
        if len(gs) >= 5:
            out[lg] = round(sum(gs) / len(gs), 2)
    return out


# The Odds API 联赛代码 → Football-Data.org 联赛代码（积分榜用）
LEAGUE_ALIAS = {
    "CH": "ELC", "ED": "DED", "BDF": "BPL", "JLG": "J1", "KL1": "KLE",
}


def _train_elo(standings_by_league: dict, min_kickoff: str):
    """Elo 训练：积分榜初始化 → 上赛季 season_2025 迭代 → 本赛季已完赛（时间正序，新赛季最后覆盖）。"""
    elo = EloModel()
    elo.init_from_standings(standings_by_league)
    season_total = 0
    finished_count = 0
    try:
        # 上赛季迭代（与回测同源）：消除积分榜与赔率源球队构成不一致导致的无历史评级
        season_path = DATA_DIR / "season_2025.json"
        if season_path.exists():
            try:
                with open(season_path, encoding="utf-8") as f:
                    season = json.load(f)
                ms_by_lg = {}
                for sm in season.get("matches", []):
                    if sm.get("homeGoals") is None or sm.get("awayGoals") is None:
                        continue
                    # 防未来函数：预测窗口之后的赛果不参与训练
                    if min_kickoff and (sm.get("utcDate") or "") >= min_kickoff:
                        continue
                    ms_by_lg.setdefault(sm["league"], []).append(sm)
                for lg_ms in ms_by_lg.values():
                    # 抓取顺序 = 从最新往回 → 反转即时间正序
                    elo.update([{
                        "utcDate": sm.get("utcDate", ""),
                        "homeTeam": {"name": sm["homeTeam"]},
                        "awayTeam": {"name": sm["awayTeam"]},
                        "score": {"fullTime": {"home": sm["homeGoals"], "away": sm["awayGoals"]}},
                    } for sm in reversed(lg_ms)])
                    season_total += len(lg_ms)
            except Exception as e:
                print(f"Elo 上赛季迭代失败: {e}")
        # 本赛季已完赛（开赛后逐步积累，最后迭代使其权重最高）
        with open(DATA_DIR / "fixtures.json", encoding="utf-8") as f:
            fixtures_all = json.load(f).get("matches", [])
        finished = [m for m in fixtures_all if m.get("status") == "FINISHED"]
        elo.update(finished)
        finished_count = len(finished)
    except Exception as e:
        print(f"Elo 赛果迭代跳过: {e}")
    return elo, season_total, finished_count


def _train_xg(min_kickoff: str):
    """xG 攻防强度模型：上赛季 season_2025（单场实时 xG 已下线，不再纳入本赛季数据）。"""
    xg_model = XgModel(max_age=20)
    n_xg = 0
    try:
        season_path = DATA_DIR / "season_2025.json"
        if season_path.exists():
            with open(season_path, encoding="utf-8") as f:
                season = json.load(f)
            for sm in season.get("matches", []):
                if sm.get("xgHome") is None or sm.get("xgAway") is None:
                    continue
                # 防未来函数：预测窗口之后的比赛不参与训练
                if min_kickoff and (sm.get("utcDate") or "") >= min_kickoff:
                    continue
                xg_model.add_match(sm["homeTeam"], sm["awayTeam"], sm["xgHome"], sm["xgAway"])
                n_xg += 1
    except Exception as e:
        print(f"xG 上赛季训练跳过: {e}")
    xg_model.finalize()
    return xg_model, n_xg


def _load_lineups() -> dict:
    """赛前首发/伤停（lineups.json，来自每小时 FotMob/ESPN/Sofascore 抓取）。"""
    lineups_by_match = {}
    try:
        lu_path = DATA_DIR / "lineups.json"
        if lu_path.exists():
            with open(lu_path, encoding="utf-8") as f:
                for lm in json.load(f).get("matches", []):
                    if lm.get("homeLineup") or lm.get("awayLineup"):
                        lineups_by_match[(norm_team(lm.get("homeTeam", "")),
                                          norm_team(lm.get("awayTeam", "")))] = lm
            print(f"首发数据: {len(lineups_by_match)} 场已公布")
    except Exception:
        pass
    return lineups_by_match


def load_jingcai() -> dict:
    """已弃用：竞彩功能移除。保留空实现避免引用报错。"""
    return {}



def main():
    parser = argparse.ArgumentParser(description="比分预测引擎")
    parser.add_argument("--skip-ai", action="store_true",
                        help="跳过 AI 研判（高频刷新时省 AI 额度，仅输出统计模型）")
    args = parser.parse_args()

    with open(CONFIG_DIR / "prediction_rules.json", "r", encoding="utf-8") as f:
        rules = json.load(f)

    analyzers = [cls(rules) for cls in ANALYSIS_REGISTRY.values()]
    print(f"分析器: {[a.name for a in analyzers]}")

    odds_by_league = load_odds()
    messages = load_messages()
    standings_by_league = load_standings()

    # 预测窗口最早开赛日：form / Elo 训练都只用早于该时刻的赛果（防未来函数）
    min_kickoff = min(
        (m.get("kickoff", "") for ms in odds_by_league.values() for m in ms if m.get("kickoff")),
        default="",
    )

    form = load_form(min_kickoff)
    print(f"赔率联赛: {list(odds_by_league.keys())}, 消息: {len(messages)} 条, 积分榜联赛: {len(standings_by_league)}, 近5场form球队: {len(form)}")

    # 各联赛历史场均总进球（供 ScoreModel 入参：避免波胆/大小球被低 λ 系统性低估）
    league_avg_goals = _league_avg_goals(min_kickoff)
    if league_avg_goals:
        print(f"联赛场均总进球: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(league_avg_goals.items())))

    # 模型训练：Elo（赛果）+ xG（攻防强度）并行独立训练，互不依赖
    elo, season_total, finished_count = _train_elo(standings_by_league, min_kickoff)
    print(f"Elo: {len(elo.ratings)} 队, 已用 {season_total} 场上赛季 + {finished_count} 场本赛季赛果迭代")

    xg_model, n_xg = _train_xg(min_kickoff)
    print(f"xG 模型: {len(xg_model.attack)} 队攻防强度, 已用 {n_xg} 场上赛季 xG 训练")

    # 赛前首发/伤停（lineups.json，来自每小时 FotMob/ESPN/Sofascore 抓取）
    lineups_by_match = _load_lineups()

    predictor = ScorePredictor(rules)
    predictions = []

    def find_standing(league_code, team_name):
        """在积分榜中按队名模糊匹配排名。"""
        table = standings_by_league.get(league_code) or standings_by_league.get(LEAGUE_ALIAS.get(league_code, league_code))
        if not table:
            return None
        n = norm_team(team_name)
        for row in table:
            # 跳过季前占位行（0 场/0 分），避免生成"排名第1（0分/0场）"无意义文本
            if (row.get("playedGames") or 0) <= 0:
                continue
            if norm_team(row["team"]) == n or norm_team(row.get("shortName", "")) == n:
                return {"position": row["position"], "points": row["points"],
                        "playedGames": row["playedGames"], "goalDifference": row["goalDifference"]}
        # 全词包含匹配（防首词前缀误配："real racing santander" 不再命中 "real madrid"）
        # 要求：查询词全部出现在榜单队名中，且查询词数>=2（单词如 "real" 不降级）
        words = [w for w in n.split() if w]
        if len(words) >= 2:
            for row in table:
                if (row.get("playedGames") or 0) <= 0:
                    continue
                rn = norm_team(row["team"])
                if all(w in rn.split() for w in words):
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

            # ── Elo 独立概率（近 5 场 form 微调，主队用主场 form、客队用客场 form）──
            ep_home, ep_draw, ep_away = elo.predict(home, away)
            fh = form.get(norm_team(home)) or {}
            fa = form.get(norm_team(away)) or {}
            # 主场龙/客场虫：主队取主场积分，客队取客场积分，缺失时回退 overall
            fh_v = fh.get("home") if isinstance(fh, dict) else fh
            fa_v = fa.get("away") if isinstance(fa, dict) else fa
            if fh_v is None and isinstance(fh, dict):
                fh_v = fh.get("overall")
            if fa_v is None and isinstance(fa, dict):
                fa_v = fa.get("overall")
            if fh_v is not None and fa_v is not None:
                adj = (fh_v - fa_v) * 0.05
                ep_home += adj
                ep_away -= adj
                _s = ep_home + ep_draw + ep_away
                ep_home, ep_draw, ep_away = ep_home / _s, ep_draw / _s, ep_away / _s
            pred_elo = {"home": ep_home, "draw": ep_draw, "away": ep_away}
            market_prob = None
            if odds_result and odds_result.get("prob"):
                market_prob = odds_result["prob"]

            # ── 融合概率：市场为主锚(去水后隐含概率) + Elo 修正（供波胆/大小球/让球等全部下游模型）──
            # 消息信号不直接加减概率（量小滞后，避免噪声），只参与置信度与分歧展示。
            fusion = rules.get("fusion", {})
            w_mkt = fusion.get("marketWeight", 0.6)
            w_elo = fusion.get("eloWeight", 0.25)
            w_xg = fusion.get("xgWeight", 0.15)
            h_sig = msg_result.get("signals", {}).get(home.lower(), {}).get("score", 0) if msg_result else 0
            a_sig = msg_result.get("signals", {}).get(away.lower(), {}).get("score", 0) if msg_result else 0
            msg_diff = (h_sig - a_sig) / 2  # 仅展示与置信度用，不参与概率融合
            # xG 模型独立概率（缺数据回退均匀，不影响融合权重和）
            xg_h, xg_d, xg_a = xg_model.predict(home, away)
            if market_prob:
                f_home = w_mkt * market_prob["home"] + w_elo * ep_home + w_xg * xg_h
                f_draw = w_mkt * market_prob["draw"] + w_elo * ep_draw + w_xg * xg_d
                f_away = w_mkt * market_prob["away"] + w_elo * ep_away + w_xg * xg_a
                _s = f_home + f_draw + f_away
                if _s > 0:
                    f_home, f_draw, f_away = f_home / _s, f_draw / _s, f_away / _s
                # 平局钳制：模型平局低于市场平局的 70% 时，向市场平局抬升
                # （模型由 1X2 反推 λ 会系统性压低平局；市场平局定价 ~24-27%，更可信）
                # 抬升后重新归一化，保持三方向概率和为 1。
                draw_floor = 0.70 * market_prob["draw"]
                if f_draw < draw_floor:
                    f_draw = draw_floor
                    _s = f_home + f_draw + f_away
                    if _s > 0:
                        f_home, f_draw, f_away = f_home / _s, f_draw / _s, f_away / _s
            else:
                f_home, f_draw, f_away = ep_home, ep_draw, ep_away

            # ── 纯模型概率（Elo+xG+Form，不含市场赔率）：用于价值检测 ──
            # 比纯 Elo 更准：xG 补充攻防强度，form 补充近期状态
            m_h = w_elo * ep_home + w_xg * xg_h
            m_d = w_elo * ep_draw + w_xg * xg_d
            m_a = w_elo * ep_away + w_xg * xg_a
            _ms = m_h + m_d + m_a
            if _ms > 0:
                model_prob = {"home": m_h / _ms, "draw": m_d / _ms, "away": m_a / _ms}
            else:
                model_prob = pred_elo

            score_model = ScoreModel()
            score_model.fit(f_home, f_draw, f_away, xg_hint=xg_model.get_lambda(home, away),
                            min_total_goals=league_avg_goals.get(league))
            score_model.score_prior = rules.get("scorePrior", {}) or {}

            # ── 多模型分歧：Elo vs Dixon-Coles vs 市场（方向不一致 = 不碰）──
            # 用真正的 Dixon-Coles 修正概率（归一化、和=1），替代旧版普通泊松拟合（未归一化）
            dc_home, dc_draw, dc_away = score_model.dc_1x2()
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
            # 凯利用模型概率算（市场概率 ⇒ o×pp≈1 恒≈0/负，无法反映模型优势）
            if raw_odds and pred_elo:
                kelly = {}
                for k in ("home", "draw", "away"):
                    o = raw_odds.get(k)
                    mp = pred_elo.get(k)
                    if isinstance(o, (int, float)) and o > 1 and mp:
                        kelly[k] = round((o * mp - 1) / (o - 1), 3)
                    else:
                        kelly[k] = None
            pred["kelly"] = kelly
            pred["spreads"] = m.get("markets", {}).get("spreads")

            # ── Elo 输出 + 价值检测 ──
            pred["eloProb"] = {k: round(v, 3) for k, v in pred_elo.items()}
            pred["dcProb"] = {k: round(v, 3) for k, v in dc12.items()}
            pred["xgProb"] = {"home": round(xg_h, 3), "draw": round(xg_d, 3), "away": round(xg_a, 3)}
            # 首发/伤停（若有）：供 AI 研判与前端展示
            # lineups.json 队名来自 fixtures（可能带后缀/简称），与 odds 队名不完全一致：
            # 先精确匹配，再用"核心词包含"兜底（home/away 双方都命中才算）
            lu = lineups_by_match.get((norm_team(home), norm_team(away)))
            if not lu:
                for (lk_h, lk_a), lm in lineups_by_match.items():
                    hn, an = norm_team(home), norm_team(away)
                    h_words = [w for w in hn.split() if len(w) >= 4]
                    a_words = [w for w in an.split() if len(w) >= 4]
                    lh_words = [w for w in lk_h.split() if len(w) >= 4]
                    la_words = [w for w in lk_a.split() if len(w) >= 4]
                    if h_words and a_words and lh_words and la_words:
                        h_hit = any(w in lh_words for w in h_words)
                        a_hit = any(w in la_words for w in a_words)
                        if h_hit and a_hit:
                            lu = lm
                            break
            if lu:
                pred["lineup"] = {
                    "homeLineup": lu.get("homeLineup", []),
                    "awayLineup": lu.get("awayLineup", []),
                    "injuries": lu.get("injuries", {}),
                    "confirmed": lu.get("confirmed"),
                    "source": lu.get("source"),
                    "fetchedAt": lu.get("fetchedAt"),
                }
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
            # 市场去水概率（独立展示，供"模型 vs 市场"对比，不参与任何模型下游）
            pred["marketProbs"] = {
                k: round(market_prob[k], 3) for k in ("home", "draw", "away")
            } if market_prob else None
            # 纯模型概率（Elo+xG+Form，不含市场先验）：价值检测同源，前端双列对比用
            pred["pureModelProbs"] = {
                "home": round(model_prob["home"], 3),
                "draw": round(model_prob["draw"], 3),
                "away": round(model_prob["away"], 3),
            }
            # 统一胜率：probabilities（前端 h2h 面板/AI 输入）复用融合概率，
            # 避免 ScorePredictor 的"市场+消息"版本与下游模型不一致（Getafe 场曾出现 52/24/23 vs 39/27/34）
            pred["probabilities"] = {
                "home": round(f_home, 3), "draw": round(f_draw, 3), "away": round(f_away, 3)
            }
            # confidence 与最终概率同源：概率集中度 + 消息一致性(低权重) - 模型分歧折价
            top_p = max(f_home, f_draw, f_away)
            msg_conf = abs(h_sig - a_sig)
            conf = min(0.95, 0.4 + top_p * 0.4 + msg_conf * fusion.get("messageConfWeight", 0.1))
            if diverge:
                conf *= 0.85  # Elo/DC/市场方向分歧 → 不确定性上升，置信度折价
            disc = rules.get("confidenceDiscount") if isinstance(rules, dict) else None
            if isinstance(disc, (int, float)) and 0.5 < disc < 1.0 and conf >= 0.65:
                conf *= disc
            pred["confidence"] = round(conf, 3)
            value_picks = []
            reverse_picks = []  # 模型 vs 市场对立（负 edge）：提示不碰，非价值
            value_threshold = rules.get("valueThreshold", 0.1)
            if market_prob and not diverge:
                sorted_probs = sorted(model_prob.values(), reverse=True)
                model_conf = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
                for k, label in (("home", "主胜"), ("draw", "平局"), ("away", "客胜")):
                    diff = model_prob[k] - market_prob[k]
                    if diff >= value_threshold:  # 正 edge：模型比市场更看好 = 价值信号
                        # gold 门槛收紧到 25%（此前 20% 的假信号伤害大；25%+ 才配得上 gold 展示）
                        level = "gold" if diff >= 0.25 and model_conf >= 0.25 else "watch"
                        value_picks.append({
                            "side": k, "label": label,
                            "modelProb": round(model_prob[k], 3),
                            "oddsProb": round(market_prob[k], 3),
                            "edge": round(diff, 3),
                            "level": level,
                            "modelConf": round(model_conf, 3),
                        })
                    elif diff <= -value_threshold:  # 负 edge：模型比市场更不看好 = 反向信号（提示不碰）
                        reverse_picks.append({
                            "side": k, "label": label,
                            "modelProb": round(model_prob[k], 3),
                            "oddsProb": round(market_prob[k], 3),
                            "edge": round(diff, 3),
                        })
            pred["valuePicks"] = value_picks
            pred["reversePicks"] = reverse_picks
            pred["signalLevel"] = "gold" if any(v["level"] == "gold" for v in value_picks) else ("watch" if value_picks else "none")

            # ── 推荐投注候选（ROI 结算用）：跨盘口类型收集 ──
            # 只记有正 edge + 正凯利的候选（负期望不推荐，避免必亏场次计入结算）
            # 候选类型: h2h(胜负) / spread(让球,Odds 盘) / score(波胆,竞彩 crsOdds)。
            bet_candidates = []
            # h2h 候选：价值信号方向 + 欧赔 + 凯利注额（用模型概率重算，避免旧版误用市场概率恒≈0）
            # 止血：仅 edge≥8% 的强信号才进投注池（弱 edge 多为模型误差，此前 6 注全输的教训）
            if value_picks:
                strong_vp = [v for v in value_picks if v.get("edge", 0) >= 0.08]
                if strong_vp:
                    best_vp = max(strong_vp, key=lambda v: abs(v.get("edge", 0)))
                    o = raw_odds.get(best_vp["side"]) if raw_odds else None
                    if isinstance(o, (int, float)) and o > 1:
                        mp = best_vp.get("modelProb")
                        kl = (o * mp - 1) / (o - 1) if mp else 0
                        stake = round(max(0.0, min(0.05, kl)), 4)
                        if stake > 0:
                            bet_candidates.append({
                                "market": "h2h", "side": best_vp["side"],
                                "score": None, "odds": round(o, 2), "stake": stake,
                                "source": "value:" + best_vp["level"], "edge": best_vp.get("edge"),
                            })

            # Dixon-Coles 波胆 + 大小球 + 分布
            dc_all = score_model.dc_scores(6)
            pred["correctScores"] = dc_all[:3]  # 全站统一展示 Top3（波胆页签/结果页对比）
            # 预测比分与波胆列表同口径（DC 修正），且与 probabilities 方向一致：
            # 全局波胆 Top1 常是 1-1（联合分布众数），会与边际 1X2 方向矛盾（如概率指向客胜却预测 1-1），
            # 因此从 DC 波胆中取与 probabilities 方向相同的最高概率比分。
            prob_dir = max(pred.get("probabilities", {}), key=lambda k: pred.get("probabilities", {}).get(k, 0))
            dir_map = {"home": "H", "draw": "D", "away": "A"}

            def _outcome(score):
                h, a = map(int, score.split("-"))
                return "H" if h > a else ("D" if h == a else "A")

            same_dir = [s for s in dc_all if _outcome(s["score"]) == dir_map.get(prob_dir)]
            if same_dir:
                pred["predictedScore"] = same_dir[0]["score"]
            elif dc_all:
                pred["predictedScore"] = dc_all[0]["score"]

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

            # ── 大小球模型价值：模型大球概率 vs 盘口隐含概率 ──
            ou_model = None
            if ou and ou.get("line") is not None:
                dist = score_model.total_goals_dist()
                over_p = sum(v for k, v in dist.items() if int(k) > ou["line"])
                over_p = min(over_p, 0.999)
                under_p = round(1 - over_p, 3)
                over_p = round(over_p, 3)
                ou_model = {"line": ou["line"], "over": over_p, "under": under_p}
                ou_edge = round(over_p - ou["over"], 3) if ou.get("over") else None
                ou_kl = None
                if ou.get("overPrice") and ou["overPrice"] > 1:
                    ou_kl = round((ou["overPrice"] * over_p - 1) / (ou["overPrice"] - 1), 3)
                ou_model["edge"] = ou_edge  # 正=模型认为大球有价值
                ou_model["kelly"] = ou_kl
            pred["ouModel"] = ou_model

            # ── 让球模型价值：模型赢盘率 vs 盘口隐含概率 ──
            sp_model = None
            sp_mkt = m.get("markets", {}).get("spreads") or {}
            hp = (sp_mkt.get("home") or {}).get("point")
            if hp is not None:
                # hp 为 The Odds API 主队让球点数（负=主让），cover_prob 同口径（负=让球）
                cover, push = score_model.cover_prob(hp)
                price = (sp_mkt.get("home") or {}).get("price")
                implied = round(1 / price, 3) if price and price > 1 else None
                # 整球盘走盘退本金：赢盘按 win，走盘按 push 计 0.5（亚盘惯例折算）
                eff = cover + 0.5 * push if push else cover
                sp_model = {
                    "point": hp, "cover": round(eff, 3), "price": price,
                    "implied": implied,
                    "edge": round(eff - implied, 3) if implied else None,
                    "kelly": round((price * eff - 1) / (price - 1), 3) if price and price > 1 and eff else None,
                }
            pred["spModel"] = sp_model
            pred["btts"] = score_model.btts_prob()
            pred["expectedGoals"] = {
                "home": round(score_model.lam_h, 2), "away": round(score_model.lam_a, 2)
            }
            pred["analysis"] = AnalysisWriter.generate(home, away, odds_result, msg_result, score_model, ou, pred["spreads"], pred["standings"], {"valuePicks": value_picks}, ou_model)

            # ── 跨市场交叉确认：给 h2h bet 加入 spread 确认数 ──
            # 逻辑：主胜 pick + sp_model 主赢盘正 edge = 确认; 客胜 pick + sp_model 负 edge = 确认
            for bc in bet_candidates:
                if bc["market"] != "h2h":
                    continue
                side = bc["side"]
                confirm = 0
                if sp_model and sp_model.get("edge") is not None:
                    if side == "home" and sp_model["edge"] > 0.02:
                        confirm += 1
                    elif side == "away" and sp_model["edge"] < -0.02:
                        confirm += 1
                if ou_model and ou_model.get("edge") is not None:
                    if side == "home" and ou_model["edge"] > 0.03:
                        confirm += 1
                    elif side == "away" and ou_model["edge"] < -0.03:
                        confirm += 1
                bc["crossConfirm"] = confirm

            # ── 让球候选（模型 spModel vs The Odds API 让球盘）──
            if sp_model and sp_model.get("edge") is not None and sp_model["edge"] >= value_threshold:
                sk = sp_model.get("kelly")
                if isinstance(sk, (int, float)) and sk > 0 and sp_model.get("price"):
                    stake_sp = round(max(0.0, min(0.05, sk)), 4)
                    if stake_sp > 0:
                        bet_candidates.append({
                            "market": "spread",
                            "side": "home_cover",
                            "point": sp_model["point"],
                            "score": None,
                            "odds": round(sp_model["price"], 2), "stake": stake_sp,
                            "source": "value:spread", "edge": sp_model["edge"],
                        })

            # ── 波胆候选（模型 DC 波胆概率，竞彩价值对比已随竞彩功能移除）──

            # ── 跨盘口类型价值投注列表：命中胜负/让球各自有值即各自成注 ──
            # 每个 market 独立成 betRec(正期望,注额=凯利上限5%),供按盘口类型分别结算。
            # pred["betRec"] 保留"最优单注"供旧字段兼容; 结算/展示用 pred["betRecs"]。
            pred["betRecs"] = bet_candidates
            pred["betRec"] = bet_candidates[0] if bet_candidates else None

            # ── AI 最终研判（可选增强层）：失败返回 None，不影响统计预测 ──
            # 费用控制：仅对 72h 内开赛的比赛调用；临场期(<24h)每次刷新都重调
            # (赛前伤停/赔率/阵容变动敏感)，远期(≥24h)12h 冷却；窗口外沿用旧研判。
            if not args.skip_ai:
                try:
                    from ai_predictor import ai_judge
                    now = datetime.now(timezone.utc)
                    kick = None
                    # 注意：预测管道只写入 pred["kickoff"]（无 utcDate 字段），窗口判断必须读 kickoff
                    if pred.get("kickoff"):
                        try:
                            kick = datetime.fromisoformat(pred["kickoff"].replace("Z", "+00:00"))
                        except Exception:
                            kick = None
                    hours_to_kick = (kick - now).total_seconds() / 3600 if kick else None
                    in_window = hours_to_kick is not None and 0 <= hours_to_kick <= AI_WINDOW_HOURS
                    need_fresh = False
                    if in_window:
                        # 临场期(<24h)：每次刷新都重调，确保赛前变动进入研判
                        need_fresh = hours_to_kick < AI_LATE_HOURS
                        if not need_fresh:
                            # 远期(≥24h)：同场 12h 内不重复调用（省额度）
                            at = pred.get("aiJudgeAt")
                            if at:
                                try:
                                    age_h = (now - datetime.fromisoformat(at.replace("Z", "+00:00"))).total_seconds() / 3600
                                    need_fresh = age_h >= AI_COOLDOWN_HOURS
                                except Exception:
                                    need_fresh = True
                            else:
                                need_fresh = True
                    if need_fresh:
                        ai = ai_judge(pred)
                        if ai:
                            pred["aiJudge"] = ai
                            pred["aiJudgeAt"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception as e:
                    print(f"  [predict] AI 研判失败(降级): {e}")
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

    # ── AI 研判继承：窗口外/冷却中的比赛沿用上次已生成的研判（省额度）──
    # 窗口内且冷却已过的比赛已在上方循环中生成新研判，这里只补窗口外的。
    try:
        old_path = DATA_DIR / "predictions.json"
        if old_path.exists():
            old = json.loads(old_path.read_text(encoding="utf-8"))
            by_key = {}
            for op in old.get("predictions", []):
                if op.get("aiJudge"):
                    by_key[(op.get("league"), op.get("homeTeam", "").lower(), op.get("awayTeam", "").lower())] = op
            for p in predictions:
                k = (p.get("league"), p.get("homeTeam", "").lower(), p.get("awayTeam", "").lower())
                if k in by_key and not p.get("aiJudge"):
                    p["aiJudge"] = by_key[k]["aiJudge"]
                    p["aiJudgeAt"] = by_key[k].get("aiJudgeAt")
                    p["aiJudgeStale"] = True  # 基于上次数据，供前端提示
    except Exception:
        pass

    # ── 预测战绩：存档 + 赛果对比 + 成功率统计 ──
    stats = evaluate_predictions(predictions)
    out["stats"] = stats

    atomic_write(DATA_DIR / "predictions.json", out)

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


def _season_start_year(iso_str):
    """从 kickoff 推断赛季开始年份（与前端一致：8 月起属当年赛季，否则属上一赛季）。
    返回赛季起始年，如 2026-08 开赛 → 2026；2026-03 开赛 → 2025。"""
    try:
        y = int((iso_str or "")[:4])
        mo = int((iso_str or "")[5:7])
        if y and mo:
            return y if mo >= 8 else y - 1
    except (ValueError, IndexError):
        pass
    return 0


def _prune_history(hist: dict, keep_seasons: int = 2):
    """prediction_history 只保留最近 keep_seasons 个完整赛季的记录，防止无限增长。
    对 predictions 与 results 同时按 kickoff 推算的赛季年份裁剪。
    无效/无日期的记录保留（不丢信息）。"""
    now = datetime.now(timezone.utc)
    cur_sea = _season_start_year(now.strftime("%Y-%m-%dT00:00:00Z"))
    min_season = cur_sea - (keep_seasons - 1)

    def keep(rec):
        s = _season_start_year((rec or {}).get("kickoff") or "")
        if s == 0:
            return True  # 未知赛季不裁剪
        return s >= min_season

    for key in ("predictions", "results"):
        arr = hist.get(key)
        if isinstance(arr, list):
            hist[key] = [r for r in arr if keep(r)]
    return hist


_REBET = {"push": None, False: False, True: True}


def _bet_won(br: dict, actual_score: str):
    """按盘口类型判定一条 betRec 是否赢。
    返回 True/False/None(push 走水=退本金,不记胜负)。"""
    if not br:
        return False
    try:
        h, a = (int(x) for x in actual_score.split("-"))
    except (ValueError, AttributeError):
        return False
    market = br.get("market") or "h2h"
    if market == "score":
        return br.get("side") == actual_score
    if market == "spread":
        pt = br.get("point")
        if pt is None:
            return False
        # 主队让球点(pt 负=主让)。主队赢盘 ⇔ 净胜差 > -pt；整球盘恰等于 -pt 为走水
        diff = h - a
        cover = diff > -pt
        if float(pt).is_integer() and diff == -pt:
            return _REBET[None]
        return cover
    # h2h（默认）：胜负方向比对
    ao = "home" if h > a else ("draw" if h == a else "away")
    return br.get("side") == ao


def _norm_beta(p):
    """把预测/历史记录里的投注清单归一化为列表。
    老数据存单条 betRec(dict)；新数据存 betRecs(list)。兼容两种。"""
    if isinstance(p.get("betRecs"), list) and p["betRecs"]:
        return p["betRecs"]
    br = p.get("betRec")
    return [br] if isinstance(br, dict) else []


def _settle_bets(bets, actual_score: str):
    """给每条 bet 写入 pnl/actualOutcome。返回 (stake, pnl, won, settled)。
    走水(push)记 pnl=0 退本金，不计胜负。"""
    stake = pnl = 0.0
    won = settled = 0
    for br in bets:
        if br is None or not br.get("odds") or not br.get("stake"):
            continue
        br["market"] = br.get("market") or "h2h"   # 老数据无 market → 视为 h2h
        br["actualOutcome"] = outcome_of(actual_score)
        w = _bet_won(br, actual_score)
        if w is True:
            br["pnl"] = round(br["stake"] * (br["odds"] - 1), 4)
            won += 1
        elif w is False:
            br["pnl"] = round(-br["stake"], 4)
        else:
            br["pnl"] = 0.0  # 走水：退本金
        stake += br["stake"]
        pnl += br["pnl"]
        settled += 1
    return stake, pnl, won, settled


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
    # 只保留最近 2 个完整赛季，防止 history 无限增长
    hist = _prune_history(hist, keep_seasons=2)

    # 1. 合并当前预测（临场口径：同场已有预测时，赛前 24h 内的新预测覆盖旧预测，
    #    取离开赛最近的一次；开赛前 >24h 的预测不覆盖已存档的）
    seen = set()
    for p in hist["predictions"]:
        seen.add((norm_team(p["homeTeam"]), norm_team(p["awayTeam"]), p.get("kickoff", "")[:10]))
    for p in predictions:
        key = (norm_team(p["homeTeam"]), norm_team(p["awayTeam"]), (p.get("kickoff") or "")[:10])
        ai = p.get("aiJudge") or {}
        if key in seen:
            # 已存档：若本次预测距开赛 <24h（临场）→ 覆盖旧预测
            try:
                kickoff_ts = datetime.fromisoformat((p.get("kickoff") or "").replace("Z", "+00:00"))
                hours_to_kickoff = (kickoff_ts - datetime.now(timezone.utc)).total_seconds() / 3600
                is_late = 0 <= hours_to_kickoff < 24
            except Exception:
                is_late = False
            old = next((q for q in hist["predictions"]
                        if (norm_team(q["homeTeam"]), norm_team(q["awayTeam"]), q.get("kickoff", "")[:10]) == key), None)
            if old and is_late and not old.get("actualScore"):
                # 覆盖为新预测（临场最新）
                old["predictedScore"] = p["predictedScore"]
                old["confidence"] = p.get("confidence")
                old["league"] = p.get("league", old.get("league", ""))
                old["modelProbs"] = p.get("modelProbs") or p.get("probabilities") or old.get("modelProbs") or {}
                old["probabilities"] = p.get("probabilities") or old.get("probabilities") or {}
                old["valuePicks"] = p.get("valuePicks") or []
                old["signalLevel"] = p.get("signalLevel", "none")
                old["predictedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                old["ouModel"] = p.get("ouModel") or old.get("ouModel")
                old["spModel"] = p.get("spModel") or old.get("spModel")
                old["betRec"] = p.get("betRec") or old.get("betRec")
                old["betRecs"] = p.get("betRecs") or old.get("betRecs")
                # AI 字段：新预测有则用新的，否则保留旧的
                if ai.get("pick"):
                    old["aiPick"] = ai.get("pick")
                    old["aiScore"] = ai.get("score")
                    old["aiConfidence"] = ai.get("confidence")
                    old["aiModel"] = ai.get("model")
                    old["aiReason"] = ai.get("reason")
                    old["aiAnalysis"] = ai.get("analysis") or old.get("aiAnalysis") or {}
                    old["aiKeyFactors"] = ai.get("keyFactors") or old.get("aiKeyFactors") or []
                    old["aiRisks"] = ai.get("risks") or old.get("aiRisks") or []
                    old["aiAltScore"] = ai.get("altScore") or old.get("aiAltScore")
            elif old and not old.get("aiPick") and ai.get("pick"):
                # 非临场或已结算：仅补写 AI 研判
                old["aiPick"] = ai.get("pick")
                old["aiScore"] = ai.get("score")
                old["aiConfidence"] = ai.get("confidence")
                old["aiModel"] = ai.get("model")
                old["aiReason"] = ai.get("reason")
                old["aiAnalysis"] = ai.get("analysis") or old.get("aiAnalysis") or {}
                old["aiKeyFactors"] = ai.get("keyFactors") or old.get("aiKeyFactors") or []
                old["aiRisks"] = ai.get("risks") or old.get("aiRisks") or []
                old["aiAltScore"] = ai.get("altScore") or old.get("aiAltScore")
            if old and not old.get("league"):
                old["league"] = p.get("league", "")
            if old and not old.get("modelProbs"):
                old["modelProbs"] = p.get("modelProbs") or p.get("probabilities") or {}
            continue
        seen.add(key)
        hist["predictions"].append({
            "homeTeam": p["homeTeam"], "awayTeam": p["awayTeam"],
            "league": p.get("league", ""),
            "kickoff": p.get("kickoff", ""),
            "predictedScore": p["predictedScore"],
            "confidence": p.get("confidence"),
            "correctScores": (p.get("correctScores") or [])[:3],
            "modelProbs": p.get("modelProbs") or p.get("probabilities") or {},
            "probabilities": p.get("probabilities") or {},
            "valuePicks": p.get("valuePicks") or [],
            "signalLevel": p.get("signalLevel", "none"),
            "aiPick": ai.get("pick"),
            "aiScore": ai.get("score"),
            "aiConfidence": ai.get("confidence"),
            "aiModel": ai.get("model"),
            "aiReason": ai.get("reason"),
            "aiAnalysis": ai.get("analysis") or {},
            "aiKeyFactors": ai.get("keyFactors") or [],
            "aiRisks": ai.get("risks") or [],
            "aiAltScore": ai.get("altScore"),
            "betRec": p.get("betRec"),
            "betRecs": p.get("betRecs"),
            # 盘口方向（用于按盘口类型统计）
            "ouModel": p.get("ouModel"),
            "spModel": p.get("spModel"),
            "hitOu": p.get("hitOu"),
            "hitSp": p.get("hitSp"),
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
        # 全词包含：查询词全部在对方队名中（"real betis" 不再命中 "real madrid"）
        wa, wb = a.split(), b.split()
        if len(wa) >= 2 and all(w in wb for w in wa):
            return True
        if len(wb) >= 2 and all(w in wa for w in wb):
            return True
        return a in b or b in a

    result_keys = {}
    for r in finished:
        result_keys[(norm_team(r["homeTeam"]), norm_team(r["awayTeam"]), r["kickoff"][:10])] = r
    # 当前预测索引（结算时补充老存档缺失的 modelProbs/aiReason）
    pred_by_key = {}
    for pr in predictions:
        pred_by_key[(pr.get("league", ""), norm_team(pr["homeTeam"]), norm_team(pr["awayTeam"]))] = pr
    evaluated = 0
    exact_hit = 0
    outcome_hit = 0
    cs_total = 0   # 波胆列表命中（任一 Top6 波胆中即命中）
    cs_hit = 0
    value_total = 0
    value_hit = 0
    ai_total = 0
    ai_hit = 0
    ai_exact = 0
    # 盘口类型统计（模型方向 vs 实际）
    ou_total = 0
    ou_hit = 0
    sp_total = 0
    sp_hit = 0
    # 置信度区间统计
    conf_buckets = {"low": [0, 0, 0.0], "mid": [0, 0, 0.0], "high": [0, 0, 0.0]}  # [n, hit, sum_conf]
    bet_stake_total = 0.0   # 累计下注额
    bet_pnl = 0.0           # 累计盈亏
    bet_won = 0             # 赢的场次数
    bet_settled = 0         # 已结算推荐场次
    bet_by_market = {}      # 按盘口类型(h2h/spread/score)统计: {market: [n, won, stake, pnl]} 
    for p in hist["predictions"]:
        if p.get("actualScore"):
            evaluated += 1
            # 已结算历史记录：补算波胆列表命中（老存档无 correctScores，从当前预测补齐近似）
            if p.get("hitCs") is None:
                _cur = pred_by_key.get((p.get("league", ""), norm_team(p["homeTeam"]), norm_team(p["awayTeam"])))
                _cs = (p.get("correctScores") or (_cur or {}).get("correctScores") or [])
                p["correctScores"] = list(_cs)[:3]
                p["hitCs"] = p["actualScore"] in [c.get("score") for c in _cs]
            if p.get("correctScores"):
                cs_total += 1
                if p.get("hitCs"):
                    cs_hit += 1
            if p.get("hitExact"):
                exact_hit += 1
            if p.get("hitOutcome"):
                outcome_hit += 1
            if p.get("aiHitOutcome") is not None:
                ai_total += 1
                if p.get("aiHitOutcome"):
                    ai_hit += 1
                if p.get("aiHitExact"):
                    ai_exact += 1
            # 已结算的历史记录：补算推荐投注盈亏（老数据无 pnl 字段，含单条 betRec / betRecs 列表）
            all_bets = _norm_beta(p)
            pending = [br for br in all_bets if br and br.get("pnl") is None and br.get("odds") and br.get("stake")]
            if pending:
                stake_a, pnl_a, won_a, settled_a = _settle_bets(pending, p["actualScore"])
            # 计入总量与 per-market（含已结算过 pnl 的记录，保证 ROI 统计完整）
            for br in all_bets:
                if not br or br.get("pnl") is None or not br.get("stake"):
                    continue
                mk = br.get("market") or "h2h"
                bet_stake_total += br["stake"]
                bet_pnl += br["pnl"]
                bet_settled += 1
                if br["pnl"] > 0:
                    bet_won += 1
                b = bet_by_market.setdefault(mk, [0, 0, 0.0, 0.0])
                b[0] += 1
                if br["pnl"] > 0:
                    b[1] += 1
                b[2] += br["stake"]
                b[3] += br["pnl"]
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
            # 从当前预测补充当时胜率/AI 理由（老存档无这些字段，且已完赛场次可能即将从 predictions 淘汰）
            if not p.get("modelProbs"):
                cur = pred_by_key.get((p.get("league", ""), norm_team(p["homeTeam"]), norm_team(p["awayTeam"])))
                if cur:
                    p["modelProbs"] = cur.get("modelProbs") or cur.get("probabilities") or {}
                    p["probabilities"] = cur.get("probabilities") or {}
                    if not p.get("aiReason"):
                        p["aiReason"] = (cur.get("aiJudge") or {}).get("reason")
                    if p.get("aiConfidence") is None:
                        p["aiConfidence"] = (cur.get("aiJudge") or {}).get("confidence")
            p["actualScore"] = r["actualScore"]
            p["evaluatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            p["hitExact"] = (p["predictedScore"] == r["actualScore"])
            p["hitOutcome"] = (outcome_of(p["predictedScore"]) == outcome_of(r["actualScore"]))
            # 波胆列表命中：模型 Top6 波胆任一等于实际比分即命中（比单一主推比分更合理）
            cs_scores = [c.get("score") for c in (p.get("correctScores") or [])]
            p["hitCs"] = r["actualScore"] in cs_scores
            evaluated += 1
            cs_total += 1
            if p["hitCs"]:
                cs_hit += 1
            if p["hitExact"]:
                exact_hit += 1
            if p["hitOutcome"]:
                outcome_hit += 1
            # 盘口类型命中：模型方向 vs 实际赛果
            actual_out = outcome_of(r["actualScore"])
            hg, ag = (int(x) for x in r["actualScore"].split("-"))
            # 大小球（ouModel：over≥0.5 → 大球方向）
            om = p.get("ouModel") or {}
            if om.get("over") is not None and om.get("line") is not None:
                ou_total += 1
                model_over = om["over"] >= 0.5
                actual_over = (hg + ag) > om["line"]
                hit_ou = model_over == actual_over
                if hit_ou:
                    ou_hit += 1
                p["hitOu"] = hit_ou
            # 让球（spModel：cover≥0.5 → 主队赢盘方向）
            sm = p.get("spModel") or {}
            if sm.get("cover") is not None and sm.get("point") is not None:
                sp_total += 1
                model_home_cover = sm["cover"] >= 0.5
                actual_cover = (hg - ag) > -sm["point"]
                hit_sp = model_home_cover == actual_cover
                if hit_sp:
                    sp_hit += 1
                p["hitSp"] = hit_sp
            # 置信度区间
            conf = p.get("confidence") or 0
            if conf < 0.5:
                key_b = "low"
            elif conf < 0.65:
                key_b = "mid"
            else:
                key_b = "high"
            conf_buckets[key_b][0] += 1
            if p["hitOutcome"]:
                conf_buckets[key_b][1] += 1
            conf_buckets[key_b][2] += conf
            # AI 研判命中（aiPick 非空且非 none 才统计）
            p["aiHitOutcome"] = None
            p["aiHitExact"] = False
            if p.get("aiPick") and p["aiPick"] != "none":
                actual_out = outcome_of(r["actualScore"])
                p["aiHitOutcome"] = (p["aiPick"] == actual_out)
                p["aiHitExact"] = bool(p.get("aiScore") and p["aiScore"] == r["actualScore"])
                ai_total += 1
                if p["aiHitOutcome"]:
                    ai_hit += 1
                if p["aiHitExact"]:
                    ai_exact += 1
            # 推荐投注盈亏：老数据可能没存 betRec/betRecs → 从 valuePicks 重建
            if not _norm_beta(p) and p.get("valuePicks"):
                vp = max(p["valuePicks"], key=lambda v: abs(v.get("edge", 0)))
                old_odds = p.get("rawOdds") or {}
                o = old_odds.get(vp["side"])
                if isinstance(o, (int, float)) and o > 1:
                    p["betRec"] = {"market": "h2h", "side": vp["side"], "odds": round(o, 2),
                                   "stake": 0.02, "source": "value:" + vp.get("level", "watch"),
                                   "edge": vp.get("edge")}
            all_bets = _norm_beta(p)
            eligible = [br for br in all_bets if br and br.get("odds") and br.get("stake")]
            if eligible:
                stake_a, pnl_a, won_a, settled_a = _settle_bets(eligible, r["actualScore"])
                p["betRecs"] = all_bets   # 结算结果写回(供 results/存档展示含 pnl)
                bet_stake_total += stake_a
                bet_pnl += pnl_a
                bet_settled += settled_a
                bet_won += won_a
                for br in all_bets:
                    if not br:
                        continue
                    mk = br.get("market") or "h2h"
                    b = bet_by_market.setdefault(mk, [0, 0, 0.0, 0.0])
                    b[0] += 1
                    if br.get("pnl", 0) > 0:
                        b[1] += 1
                    b[2] += br.get("stake", 0)
                    b[3] += br.get("pnl", 0)
            # 价值标记方向命中（bestOutcome）
            vp = p.get("valuePicks") or []
            if vp:
                value_total += 1
                best = max(vp, key=lambda v: abs(v.get("edge", 0)))
                if best.get("side") == outcome_of(r["actualScore"]):
                    value_hit += 1
            hist["results"].append({
                "homeTeam": p["homeTeam"], "awayTeam": p["awayTeam"],
                "league": p.get("league", ""),
                "kickoff": p.get("kickoff", ""),
                "predictedScore": p["predictedScore"],
                "actualScore": r["actualScore"],
                "hitExact": p["hitExact"],
                "hitOutcome": p["hitOutcome"],
                "hitCs": p.get("hitCs"),
                "correctScores": (p.get("correctScores") or [])[:3],
                "modelProbs": p.get("modelProbs") or p.get("probabilities") or {},
                "probabilities": p.get("probabilities") or {},
                "aiPick": p.get("aiPick"),
                "aiScore": p.get("aiScore"),
                "aiConfidence": p.get("aiConfidence"),
                "aiReason": p.get("aiReason"),
                "aiHitOutcome": p.get("aiHitOutcome"),
                "aiHitExact": p.get("aiHitExact"),
                "betRec": p.get("betRec"),
                "betRecs": p.get("betRecs"),
            })

    # 3.5 历史 results 记录补波胆列表（老存档无 correctScores，从当前预测补齐，命中重算）
    for rec in hist.get("results", []):
        if rec.get("correctScores"):
            continue
        _cur = pred_by_key.get((rec.get("league", ""), norm_team(rec.get("homeTeam", "")), norm_team(rec.get("awayTeam", ""))))
        _cs = (_cur or {}).get("correctScores") or []
        if _cs:
            rec["correctScores"] = list(_cs)[:3]
            rec["hitCs"] = rec.get("actualScore") in [c.get("score") for c in _cs]

    # 4. 统计
    stats = {
        "total": len(hist["predictions"]),
        "finished": evaluated,
        "exactHit": exact_hit,
        "exactRate": round(exact_hit / evaluated, 4) if evaluated else 0,
        "outcomeHit": outcome_hit,
        "outcomeRate": round(outcome_hit / evaluated, 4) if evaluated else 0,
        "csTotal": cs_total,
        "csHit": cs_hit,
        "csRate": round(cs_hit / cs_total, 4) if cs_total else 0,
        "valueTotal": value_total,
        "valueHit": value_hit,
        "valueRate": round(value_hit / value_total, 4) if value_total else 0,
        "aiTotal": ai_total,
        "aiHit": ai_hit,
        "aiRate": round(ai_hit / ai_total, 4) if ai_total else 0,
        "aiExact": ai_exact,
        "aiExactRate": round(ai_exact / ai_total, 4) if ai_total else 0,
        "betSettled": bet_settled,
        "betWon": bet_won,
        "betRate": round(bet_won / bet_settled, 4) if bet_settled else 0,
        "betStakeTotal": round(bet_stake_total, 4),
        "betPnl": round(bet_pnl, 4),
        "betROI": round(bet_pnl / bet_stake_total, 4) if bet_stake_total else 0,
        # 按盘口类型(h2h/spread/score)分别结算 ROI
        "betByMarket": {
            k: {"n": v[0], "won": v[1],
                "rate": round(v[1] / v[0], 4) if v[0] else 0,
                "stake": round(v[2], 4), "pnl": round(v[3], 4),
                "roi": round(v[3] / v[2], 4) if v[2] else 0}
            for k, v in sorted(bet_by_market.items())
        },
        # 盘口类型统计
        "ouTotal": ou_total,
        "ouHit": ou_hit,
        "ouRate": round(ou_hit / ou_total, 4) if ou_total else 0,
        "spTotal": sp_total,
        "spHit": sp_hit,
        "spRate": round(sp_hit / sp_total, 4) if sp_total else 0,
        # 置信度区间统计
        "confBuckets": {
            k: {"n": v[0], "hit": v[1],
                "rate": round(v[1] / v[0], 4) if v[0] else 0,
                "avgConf": round(v[2] / v[0], 4) if v[0] else 0}
            for k, v in conf_buckets.items()
        },
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    hist["stats"] = stats
    atomic_write(history_path, hist)
    return stats


if __name__ == "__main__":
    sys.exit(main())
