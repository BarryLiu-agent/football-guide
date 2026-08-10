# -*- coding: utf-8 -*-
"""修复 AnalysisWriter.generate 签名 + 价值段落。"""
from pathlib import Path

p = Path("scripts/predict.py")
src = p.read_text(encoding="utf-8")

# 1. 签名加 extra 参数
old_sig = "    def generate(home, away, odds_result, msg_result, score_model, ou=None, spreads=None, standings=None):"
new_sig = "    def generate(home, away, odds_result, msg_result, score_model, ou=None, spreads=None, standings=None, extra=None):"
assert old_sig in src, "签名未找到"
src = src.replace(old_sig, new_sig, 1)

# 2. 在结论段前插入价值段落（找到 结论 段）
anchor = "        # 5. 综合结论"
value_block = '''        # 4.5 价值提示（模型 vs 市场分歧）
        if extra and extra.get("valuePicks"):
            label = {"home": "主胜", "draw": "平局", "away": "客胜"}
            for vp in extra["valuePicks"][:2]:
                lines.append(
                    f"【价值】独立模型（Elo）更看好{label[vp['side']]}：模型概率 {vp['modelProb'] * 100:.0f}% "
                    f"vs 市场 {vp['oddsProb'] * 100:.0f}%，差值 +{vp['edge'] * 100:.0f}%，存在价值分歧。")

'''
assert anchor in src, "结论锚点未找到"
src = src.replace(anchor, value_block + anchor, 1)

p.write_text(src, encoding="utf-8")
print("generate 已修复")
