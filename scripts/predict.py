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
        "manchester united": "manchester united", "man utd": "manchester united", "united": "manchester united",
        "manchester city": "manchester city", "man city": "manchester city",
        "real madrid": "real madrid", "barcelona": "barcelona", "barça": "barcelona",
        "bayern": "bayern munich", "bayern munich": "bayern munich",
        "psg": "paris saint-germain", "paris saint-germain": "paris saint-germain", "paris st-germain": "paris saint-germain",
        "inter milan": "inter", "inter": "inter", "ac milan": "ac milan", "milan": "ac milan",
        "atletico": "atletico madrid", "atlético": "atletico madrid",
        "juventus": "juventus", "juve": "juventus",
        "dortmund": "borussia dortmund", "liverpool": "liverpool",
        "arsenal": "arsenal", "chelsea": "chelsea", "tottenham": "tottenham",
        "napoli": "napoli", "roma": "roma", "lazio": "lazio",
        "marseille": "marseille", "lyon": "lyon", "monaco": "monaco",
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
            # 该消息涉及哪支球队
            involved = set()
            if any(tok in text for tok in [home, self.TEAM_ALIASES.get(home, home)]):
                involved.add(home)
            if any(tok in text for tok in [away, self.TEAM_ALIASES.get(away, away)]):
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

    def over_under(self, line=2.5):
        """大小球: 返回 {over, under} 概率。"""
        n = self.MAX_GOALS
        m = [[self._poisson(i, self.lam_h) * self._poisson(j, self.lam_a) for j in range(n)] for i in range(n)]
        p_over = sum(m[i][j] for i in range(n) for j in range(n) if i + j > line)
        return {"line": line, "over": round(p_over, 3), "under": round(1 - p_over, 3)}


# ── 文字分析生成（规则模板）──────────────────────────────

class AnalysisWriter:
    """基于赔率/波胆/大小球/消息信号生成中文文字分析。"""

    @staticmethod
    def generate(home, away, odds_result, msg_result, score_model, ou=None):
        lines = []
        prob = odds_result.get("prob") if odds_result else None

        # 1. 胜负分析
        if prob:
            ph, pd, pa = prob["home"], prob["draw"], prob["away"]
            if ph >= 0.55:
                verdict = f"市场高度看好主队{home}，主胜隐含概率 {ph:.0%}"
            elif pa >= 0.55:
                verdict = f"市场明显看好客队{away}，客胜隐含概率 {pa:.0%}"
            elif ph >= pa and ph - pa < 0.15:
                verdict = f"双方实力接近，主队{home}略占优（主胜 {ph:.0%} vs 客胜 {pa:.0%}）"
            else:
                verdict = f"比赛悬念较大，主胜 {ph:.0%} / 平 {pd:.0%} / 客胜 {pa:.0%}"
            lines.append(f"【胜负】{verdict}。")

        # 2. 波胆分析
        cs = score_model.correct_scores(3)
        if cs:
            top = cs[0]
            cs_str = "、".join(f"{c['score']}（{c['prob']:.0%}）" for c in cs)
            lines.append(f"【波胆】泊松模型推算最可能比分：{cs_str}。")

        # 3. 大小球（真实赔率优先，否则泊松）
        if not ou:
            ou = score_model.over_under(2.5)
        if ou:
            direction = "大球" if ou["over"] >= 0.5 else "小球"
            lines.append(f"【大小球】{ou['line']} 球盘口：大球 {ou['over']:.0%} / 小球 {ou['under']:.0%}，倾向{direction}。")

        # 4. 消息信号
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

        # 5. 综合结论
        if prob:
            winner = home if prob["home"] >= prob["away"] else away
            lines.append(f"【结论】综合赔率与消息，{winner}不败概率更高，关注波胆 {cs[0]['score']} 方向。")

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


def main():
    with open(CONFIG_DIR / "prediction_rules.json", "r", encoding="utf-8") as f:
        rules = json.load(f)

    analyzers = [cls(rules) for cls in ANALYSIS_REGISTRY.values()]
    print(f"分析器: {[a.name for a in analyzers]}")

    odds_by_league = load_odds()
    messages = load_messages()
    print(f"赔率联赛: {list(odds_by_league.keys())}, 消息: {len(messages)} 条")

    predictor = ScorePredictor(rules)
    predictions = []

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

            # 泊松模型: 波胆 + 大小球
            score_model = ScoreModel()
            if odds_result and odds_result.get("prob"):
                p = odds_result["prob"]
                score_model.fit(p["home"], p["draw"], p["away"])

            pred = predictor.predict(home, away, odds_result, msg_result)
            pred["league"] = league
            pred["kickoff"] = m.get("kickoff", "")
            pred["matchUrl"] = m.get("matchUrl", "")
            pred["correctScores"] = score_model.correct_scores(6)
            # 真实大小球（The Odds API totals）优先，否则泊松推导
            ou = None
            totals_mkt = m.get("markets", {}).get("totals")
            if totals_mkt and totals_mkt.get("over") and totals_mkt.get("under"):
                po, pu = 1 / totals_mkt["over"], 1 / totals_mkt["under"]
                s = po + pu
                ou = {"line": totals_mkt["line"], "over": round(po / s, 3), "under": round(pu / s, 3)}
            if not ou:
                ou = score_model.over_under(2.5)
            pred["overUnder"] = ou
            # 预测比分 = 泊松模型最可能波胆（比期望值四舍五入更有区分度）
            top_cs = score_model.correct_scores(1)
            if top_cs:
                pred["predictedScore"] = top_cs[0]["score"]
            pred["analysis"] = AnalysisWriter.generate(home, away, odds_result, msg_result, score_model, ou)
            predictions.append(pred)

    out = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(predictions),
        "rulesVersion": "1.0",
        "predictions": predictions,
        "disclaimer": "本结果仅用于个人数据分析与研究, 不构成任何投注建议",
    }
    with open(DATA_DIR / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 摘要
    notable = [p for p in predictions if p["confidence"] >= rules.get("confidenceMin", 0.4)]
    print(f"\n总计预测: {len(predictions)} 场, 高置信(≥{rules.get('confidenceMin')}): {len(notable)} 场")
    for p in notable[:5]:
        print(f"  {p['homeTeam']} vs {p['awayTeam']}: {p['predictedScore']} (置信度 {p['confidence']:.0%})")
    print(f"\n输出: data/predictions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
