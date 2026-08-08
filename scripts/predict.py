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

            pred = predictor.predict(home, away, results.get("odds"), results.get("message"))
            pred["league"] = league
            pred["kickoff"] = m.get("kickoff", "")
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
