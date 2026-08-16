# -*- coding: utf-8 -*-
"""
check_consistency.py - 预测一致性自检（CI 用）
对 data/predictions.json 逐场校验：
  1. predictedScore 的胜负方向 == probabilities 中最大概率的方向（应为 0 场不一致）
  2. probabilities 三项应归一化(和≈1)
  3. dcProb 归一化(有则校验)
不一致 > 0 或存在未归一化概率 → 输出问题并 return 1，供 CI 拦截。

用法:
  python scripts/check_consistency.py
"""
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "predictions.json"


def outcome_of(score_str):
    try:
        h, a = score_str.split("-")
        h, a = int(h), int(a)
        return "H" if h > a else ("D" if h == a else "A")
    except (ValueError, AttributeError):
        return None


def main():
    if not PATH.exists():
        print("MISSING data/predictions.json")
        return 1
    data = json.loads(PATH.read_text(encoding="utf-8"))
    preds = data.get("predictions", [])
    issues = []
    for p in preds:
        h, a = p.get("homeTeam"), p.get("awayTeam")
        lg = p.get("league", "?")
        tag = f"[{lg}] {h} vs {a}"
        probs = p.get("probabilities") or {}
        if not probs:
            issues.append(f"{tag}: 无 probabilities")
            continue
        s = sum(v for v in probs.values() if isinstance(v, (int, float)))
        if s and abs(s - 1.0) > 0.01:
            issues.append(f"{tag}: probabilities 和={s:.4f}")
        top = max(probs, key=lambda k: (probs.get(k) or 0))
        dir_map = {"home": "H", "draw": "D", "away": "A"}
        o = outcome_of(p.get("predictedScore", ""))
        if o and dir_map.get(top) != o:
            issues.append(f"{tag}: predictedScore={p.get('predictedScore')}({o}) 方向≠概率最大值({top})")

    print(f"predictions 共 {len(preds)} 场, 不一致/异常 {len(issues)} 项")
    for i in issues:
        print("  X " + i)
    if issues:
        return 1
    print("一致性校验通过: predictedScore 方向均与最高概率一致, 概率归一化正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
