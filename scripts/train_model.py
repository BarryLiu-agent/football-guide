# -*- coding: utf-8 -*-
"""
train_model.py - 机器学习模型训练与评估
用历史赛程训练逻辑回归模型，输出概率预测替代手写融合权重。

用法:
  python scripts/train_model.py              # 用现有赛季数据训练
  python scripts/train_model.py --fetch      # 先抓多赛季数据再训练

输出:
  data/ml_model.pkl    - 训练好的模型（含 StandardScaler）
  终端评估报告
"""
import io
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from features import compute_features, FEATURE_NAMES, N_FEATURES  # noqa: E402

DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "ml_model.pkl"


# ── 数据加载 ────────────────────────────────────────────

def load_all_seasons():
    """加载 data/season_*.json，按时间排序合并。"""
    all_matches = []
    for p in sorted(DATA_DIR.glob("season_*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            for m in data.get("matches", []):
                if m.get("homeGoals") is not None and m.get("awayGoals") is not None:
                    all_matches.append(m)
        except Exception as e:
            print(f"  ⚠ 跳过 {p.name}: {e}")
    all_matches.sort(key=lambda m: m.get("utcDate", ""))
    return all_matches


def load_fixtures_finished():
    """加载 fixtures.json 中已完赛的比赛（补充本赛季赛果）。"""
    path = DATA_DIR / "fixtures.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    finished = []
    for m in data.get("matches", []):
        score = m.get("score") or {}
        ft = score.get("fullTime") or {}
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None:
            continue
        comp = m.get("competition") or {}
        league_code = comp.get("code", "")
        # football-data code → 我们的 code
        FD_MAP = {"PL": "PL", "PD": "PD", "SA": "SA", "BL1": "BL1", "FL1": "FL1", "CL": "CL"}
        league = FD_MAP.get(league_code, league_code)
        home_t = m.get("homeTeam") or {}
        away_t = m.get("awayTeam") or {}
        finished.append({
            "homeTeam": home_t.get("shortName") or home_t.get("name", ""),
            "awayTeam": away_t.get("shortName") or away_t.get("name", ""),
            "homeGoals": hg, "awayGoals": ag,
            "utcDate": m.get("utcDate", ""),
            "league": league,
            "season": (m.get("season") or {}).get("currentMatchday", ""),
        })
    finished.sort(key=lambda m: m.get("utcDate", ""))
    return finished


# ── 模型评估 ────────────────────────────────────────────

def evaluate(y_true, y_pred, y_prob, label=""):
    """打印分类报告、校准桶、方向分布。"""
    from sklearn.metrics import accuracy_score, log_loss, classification_report

    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob, labels=[0, 1, 2])

    # 基线
    dist = Counter(y_true)
    baseline = max(dist.values()) / n

    print(f"\n{'='*50}")
    print(f"  {label} ({n} 场)")
    print(f"{'='*50}")
    print(f"  命中率: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  基线(最多类): {baseline:.4f} ({baseline*100:.1f}%)")
    print(f"  提升: +{(acc - baseline)*100:.1f}%")

    names = ["主胜", "平局", "客胜"]
    print(f"\n  分类报告:")
    print(classification_report(y_true, y_pred, target_names=names, digits=3))

    # 方向分布
    pred_dist = Counter(y_pred)
    print(f"  真实分布: {dict(dist)}")
    print(f"  预测分布: {dict(pred_dist)}")

    # 校准桶
    print(f"\n  校准 (预测概率 vs 实际命中):")
    for cls_idx, cls_name in enumerate(names):
        buckets = [(0, 0.35, "<35%"), (0.35, 0.45, "35-45%"), (0.45, 0.55, "45-55%"),
                   (0.55, 0.65, "55-65%"), (0.65, 1.01, "≥65%")]
        for lo, hi, bl in buckets:
            mask = (y_prob[:, cls_idx] >= lo) & (y_prob[:, cls_idx] < hi)
            if mask.sum() < 3:
                continue
            pred_avg = y_prob[mask, cls_idx].mean()
            act_rate = (y_true[mask] == cls_idx).mean()
            bias = act_rate - pred_avg
            print(f"    {cls_name} {bl}: n={mask.sum():3d}, 预测={pred_avg:.3f}, 实际={act_rate:.3f}, 偏差={bias:+.3f}")

    # 最高概率方向的整体校准
    print(f"\n  整体方向校准 (最高概率方向):")
    best_cls = y_prob.argmax(axis=1)
    best_prob = y_prob.max(axis=1)
    hit = (best_cls == y_true)
    buckets = [(0, 0.40, "<40%"), (0.40, 0.50, "40-50%"), (0.50, 0.60, "50-60%"),
               (0.60, 0.70, "60-70%"), (0.70, 1.01, "≥70%")]
    for lo, hi, bl in buckets:
        mask = (best_prob >= lo) & (best_prob < hi)
        if mask.sum() < 3:
            continue
        print(f"    {bl}: n={mask.sum():3d}, 命中={hit[mask].mean():.3f}, 平均置信={best_prob[mask].mean():.3f}")

    return acc


def show_feature_importance(model, feature_names):
    """打印特征重要性（逻辑回归系数）。"""
    coefs = model.coef_  # shape (3, n_features)
    print(f"\n  特征重要性 (平均|系数|):")
    avg_abs = np.abs(coefs).mean(axis=0)
    ranked = sorted(zip(feature_names, avg_abs), key=lambda x: -x[1])
    for name, imp in ranked[:15]:
        h, d, a = coefs[0, feature_names.index(name)], coefs[1, feature_names.index(name)], coefs[2, feature_names.index(name)]
        print(f"    {name:25s} {imp:.4f}  (H={h:+.4f} D={d:+.4f} A={a:+.4f})")


# ── 训练 ────────────────────────────────────────────────

def train_model(X_train, y_train, X_test, y_test, feature_names, C=1.0):
    """训练逻辑回归并评估。返回 (model, scaler)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        C=C, max_iter=2000,
        solver="lbfgs", random_state=42, class_weight=None,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    evaluate(y_test, y_pred, y_prob, label=f"ML 模型 (C={C})")
    show_feature_importance(model, feature_names)

    return model, scaler


def train_and_save():
    """主训练流程。"""
    print("加载数据...")
    matches = load_all_seasons()
    fixtures = load_fixtures_finished()
    if fixtures:
        matches = matches + fixtures
        print(f"  赛季数据 + 本赛季已赛: {len(matches)} 场")
    else:
        print(f"  赛季数据: {len(matches)} 场")

    if len(matches) < 100:
        print("ERROR: 数据不足 100 场，请先运行 fbref_seasons.py 抓取多赛季")
        return 1

    print("提取特征...")
    t0 = time.time()
    X, y, meta = compute_features(matches, verbose=True)
    print(f"  耗时 {time.time()-t0:.1f}s")

    # 时序切分: 前 80% 训练, 后 20% 验证
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n切分: 训练 {split} 场, 验证 {n-split} 场")
    print(f"训练集分布: {dict(Counter(y_train))}")
    print(f"验证集分布: {dict(Counter(y_test))}")

    # ── 基线：Elo 纯模型 ──
    print("\n" + "="*50)
    print("  基线: Elo 概率直接选方向")
    print("="*50)
    elo_probs = X_test[:, 1:4]  # elo_prob_h, elo_prob_d, elo_prob_a
    elo_pred = elo_probs.argmax(axis=1)
    from sklearn.metrics import accuracy_score
    elo_acc = accuracy_score(y_test, elo_pred)
    print(f"  Elo 命中率: {elo_acc:.4f} ({elo_acc*100:.1f}%)")

    # ── 训练 ML 模型 ──
    print("\n── 训练 ML 模型 ──")
    model, scaler = train_model(X_train, y_train, X_test, y_test, FEATURE_NAMES, C=1.0)

    # 交叉验证找最优 C
    print("\n── 交叉验证找正则化参数 ──")
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler as SS
    scaler_cv = SS()
    X_s = scaler_cv.fit_transform(X)
    lrcv = LogisticRegressionCV(
        Cs=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        cv=5, max_iter=2000,
        solver="lbfgs", random_state=42, scoring="accuracy",
    )
    lrcv.fit(X_s, y)
    print(f"  最优 C: {lrcv.C_[0]:.2f}")
    print(f"  CV 平均准确率: {lrcv.score(X_s, y):.4f}")

    # 用最优 C 重新训练
    best_C = float(lrcv.C_[0])
    print(f"\n── 用最优 C={best_C:.2f} 重新训练 ──")
    model, scaler = train_model(X_train, y_train, X_test, y_test, FEATURE_NAMES, C=best_C)

    # 保存模型
    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_size": split,
        "test_size": n - split,
        "best_C": best_C,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n✅ 模型已保存: {MODEL_PATH}")
    print(f"   特征数: {N_FEATURES}, 训练集: {split}, 验证集: {n-split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(train_and_save())
