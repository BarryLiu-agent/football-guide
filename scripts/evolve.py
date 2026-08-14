# -*- coding: utf-8 -*-
"""
evolve.py - 模型持续进化机制
每周自动运行：
1. 用最近结算的预测战绩重新评估模型校准（confBuckets 偏差）
2. 参数自适应：按历史偏差调整置信度输出（偏乐观则打折）
3. 生成进化报告 data/evolution.json：
   - 当前参数版本
   - 各置信度区间实际命中 vs 平均置信（偏差）
   - 建议：高置信打折系数、是否需要重新训练

用法:
  python scripts/evolve.py          # 生成进化报告
  python scripts/evolve.py --apply  # 生成报告并应用自适应参数（写 prediction_rules.json）
"""
import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
RULES_FILE = CONFIG_DIR / "prediction_rules.json"
EVO_FILE = DATA_DIR / "evolution.json"

# 置信度打折区间（与 predict.py confBuckets 一致）
BUCKETS = [("low", "<50%"), ("mid", "50-65%"), ("high", "≥65%")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="应用自适应参数到 prediction_rules.json")
    args = parser.parse_args()

    history_path = DATA_DIR / "prediction_history.json"
    if not history_path.exists():
        print("无预测历史，跳过进化")
        return 0
    with open(history_path, encoding="utf-8") as f:
        hist = json.load(f)

    stats = hist.get("stats") or {}
    conf_buckets = stats.get("confBuckets") or {}
    # 已结算样本量
    settled = stats.get("finished", 0)

    # 校准评估：每个区间的偏差（实际命中 - 平均置信）
    calib = []
    for key, label in BUCKETS:
        b = conf_buckets.get(key)
        if not b or not b.get("n"):
            continue
        bias = round(b["rate"] - b["avgConf"], 4)
        calib.append({
            "bucket": key, "label": label,
            "n": b["n"], "hitRate": b["rate"], "avgConf": b["avgConf"],
            "bias": bias,
            "verdict": "良好" if abs(bias) < 0.03 else ("偏乐观" if bias < 0 else "偏保守"),
        })

    # 整体偏差与建议折扣
    overall_bias = None
    discount = 1.0
    high_b = conf_buckets.get("high")
    if high_b and high_b.get("n", 0) >= 20 and high_b["rate"] < high_b["avgConf"]:
        # 高置信偏乐观：建议折扣 = 实际命中 / 平均置信
        discount = round(high_b["rate"] / max(0.001, high_b["avgConf"]), 3)
        overall_bias = round(high_b["rate"] - high_b["avgConf"], 4)

    # 盘口类型统计
    ou_rate = stats.get("ouRate")
    sp_rate = stats.get("spRate")

    report = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modelVersion": 1,
        "settledSamples": settled,
        "calibration": calib,
        "overallBias": overall_bias,
        "highConfDiscount": discount,
        "ouRate": ou_rate,
        "spRate": sp_rate,
        "recommendations": [],
        "applied": False,
    }

    # 建议生成
    if settled < 20:
        report["recommendations"].append("样本不足 20 场，暂不调参，继续积累")
    elif overall_bias is not None and overall_bias < -0.03:
        report["recommendations"].append(
            f"高置信区间偏乐观 {overall_bias:.1%}，建议置信度输出乘 {discount} 折扣")
    else:
        report["recommendations"].append("校准良好，无需调整")
    if ou_rate is not None and stats.get("ouTotal", 0) >= 20 and ou_rate < 0.45:
        report["recommendations"].append(f"大小球命中率 {ou_rate:.1%} 偏低，检查大小球模型")
    if sp_rate is not None and stats.get("spTotal", 0) >= 20 and sp_rate < 0.45:
        report["recommendations"].append(f"让球命中率 {sp_rate:.1%} 偏低，检查让球模型")

    if args.apply and overall_bias is not None and overall_bias < -0.03 and discount < 0.97:
        # 应用折扣：把 high 置信度的输出乘折扣（写入 rules 供 predict.py 使用）
        rules = json.loads(RULES_FILE.read_text(encoding="utf-8"))
        rules["confidenceDiscount"] = discount
        RULES_FILE.write_text(json.dumps(rules, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        report["applied"] = True
        report["recommendations"].append(f"已应用高置信折扣 {discount}")

    EVO_FILE.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
