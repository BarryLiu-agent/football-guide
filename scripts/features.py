# -*- coding: utf-8 -*-
"""
features.py - 特征工程模块
从历史赛程提取机器学习特征，供 train_model.py 和 predict.py 共用。
严格时序遍历：每场比赛只使用该场之前的数据（防未来泄露）。

特征列表（32维）：
  [0]  elo_diff          - Elo 主客差（含主场优势）
  [1]  elo_prob_h        - Elo 模型主胜概率
  [2]  elo_prob_d        - Elo 模型平局概率
  [3]  elo_prob_a        - Elo 模型客胜概率
  [4-8] form_home_*      - 主队近5场主场：场均积分/进球/失球/净胜球/胜率
  [9-13] form_away_*     - 客队近5场客场：同上
  [14] form_pts_diff     - 主队主场场均积分 - 客队客场场均积分
  [15] form_gd_diff      - 主队主场净胜球 - 客队客场净胜球
  [16-18] h2h_*          - 近5次交锋：主胜率/平局率/场均总进球
  [19] form_home_cs      - 主队主场零封率
  [20] form_away_cs      - 客队客场零封率
  [21-24] xg_*（可选）   - xG 差值/进攻/防守（仅当数据可用时）
  [25] season_progress   - 赛季进度 0~1（第几轮/总轮次，估算）
  [26] home_form_trend   - 主队近3场 vs 近10场状态趋势（上升/下降）
  [27] away_form_trend   - 客队趋势
  [28] elo_rating_diff   - 纯 Elo 差（不含主场优势，供模型自行学习主场权重）
  [29] home_advantage    - 常量 40（供模型自行缩放）
  [30] league_encoded    - 联赛编号 0~4（五大联赛有不同平局率基线）
  [31] total_goals_trend - 双方近5场总进球趋势（高/低进球联赛特征）

用法：
  from features import compute_features, FEATURE_NAMES
  X, y, meta = compute_features(matches)
"""

import math
from collections import defaultdict

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from elo import EloModel, XgModel, norm_team, HOME_ADV, INIT_ELO, SPREAD

# ── 常量 ─────────────────────────────────────────────────
FORM_N = 5       # 近5场状态窗口
FORM_TREND_N = 3 # 趋势：近3场 vs 近 FORM_N 场
FORM_LONG_N = 10 # 长窗口（趋势用）
H2H_N = 5        # 近5次交锋
DEFAULT_PTS = 1.0   # 缺失时场均积分默认值（≈平局率基线）
DEFAULT_GD = 0.0
DEFAULT_CS = 0.2
DEFAULT_H2H_WR = 0.33
DEFAULT_H2H_DR = 0.26
DEFAULT_H2H_GOALS = 2.5
DEFAULT_XG = 0.0

LEAGUE_MAP = {"PL": 0, "PD": 1, "SA": 2, "BL1": 3, "FL1": 4, "CL": 5}

FEATURE_NAMES = [
    "elo_diff", "elo_prob_h", "elo_prob_d", "elo_prob_a",
    "form_home_pts", "form_home_gf", "form_home_ga", "form_home_gd", "form_home_wr",
    "form_away_pts", "form_away_gf", "form_away_ga", "form_away_gd", "form_away_wr",
    "form_pts_diff", "form_gd_diff",
    "h2h_home_wr", "h2h_draw_rate", "h2h_avg_goals",
    "form_home_cs", "form_away_cs",
    "xg_home_diff", "xg_away_diff", "xg_home_for", "xg_away_for",
    "season_progress", "home_form_trend", "away_form_trend",
    "elo_rating_diff", "home_advantage",
    "league_encoded", "total_goals_trend",
]

N_FEATURES = len(FEATURE_NAMES)  # 32


def _avg(items):
    return sum(items) / len(items) if items else None


def _form_stats(matches_list):
    """从比赛列表计算 form 统计。matches_list: [{gf, ga}]"""
    n = len(matches_list)
    if n == 0:
        return {
            "pts": DEFAULT_PTS, "gf": 1.2, "ga": 1.2,
            "gd": DEFAULT_GD, "wr": 0.33, "cs": DEFAULT_CS,
        }
    pts, gfs, gas, wins, cs = 0.0, 0.0, 0.0, 0, 0
    for m in matches_list:
        gf, ga = m["gf"], m["ga"]
        gfs += gf
        gas += ga
        if gf > ga:
            pts += 3; wins += 1
        elif gf == ga:
            pts += 1
        if ga == 0:
            cs += 1
    return {
        "pts": pts / n,
        "gf": gfs / n,
        "ga": gas / n,
        "gd": (gfs - gas) / n,
        "wr": wins / n,
        "cs": cs / n,
    }


def _h2h_stats(meetings):
    """从交锋记录计算 H2H 统计。meetings: [{hg, ag}]"""
    n = len(meetings)
    if n == 0:
        return {
            "home_wr": DEFAULT_H2H_WR,
            "draw_rate": DEFAULT_H2H_DR,
            "avg_goals": DEFAULT_H2H_GOALS,
        }
    hw, dr, total_g = 0, 0, 0.0
    for m in meetings:
        hg, ag = m["hg"], m["ag"]
        total_g += hg + ag
        if hg > ag:
            hw += 1
        elif hg == ag:
            dr += 1
    return {
        "home_wr": hw / n,
        "draw_rate": dr / n,
        "avg_goals": total_g / n,
    }


def compute_features(matches, verbose=False):
    """
    从赛程列表提取特征矩阵。
    matches: 按时间排序的比赛列表，每场需有 homeTeam, awayTeam, homeGoals, awayGoals, utcDate, league。
    可选字段: xgHome, xgAway, season。
    返回: (X: np.ndarray shape(N, N_FEATURES), y: np.ndarray shape(N), meta: list[dict])
    """
    import numpy as np

    elo = EloModel()
    xg_model = XgModel(max_age=20)

    # 队内比赛历史（用于 form 计算）
    # team_history[norm_name] = list of {gf, ga, side, date}
    team_history = defaultdict(list)
    # 交锋记录（用于 H2H 计算）
    # h2h_history[(norm_home, norm_away)] = list of {hg, ag}
    h2h_history = defaultdict(list)
    # 联赛总进球趋势（最近 N 场的场均总进球）
    league_goals = defaultdict(list)

    rows = []
    labels = []
    meta_list = []

    for m in matches:
        home = m.get("homeTeam", "")
        away = m.get("awayTeam", "")
        hg = m.get("homeGoals")
        ag = m.get("awayGoals")
        league = m.get("league", "")
        date = m.get("utcDate", "")
        season = m.get("season", "")

        if not home or not away or hg is None or ag is None:
            continue

        nh, na = norm_team(home), norm_team(away)

        # ── 1. Elo 特征 ──
        elo_h = elo.get_rating(home)
        elo_a = elo.get_rating(away)
        elo_diff_raw = elo_h - elo_a  # 不含主场优势
        ep_h, ep_d, ep_a = elo.predict(home, away)  # 含主场优势

        # ── 2. Form 特征（主队主场 vs 客队客场）──
        home_home = [x for x in team_history[nh] if x["side"] == "home"][-FORM_N:]
        away_away = [x for x in team_history[na] if x["side"] == "away"][-FORM_N:]
        hf = _form_stats(home_home)
        af = _form_stats(away_away)

        # ── 3. H2H 特征 ──
        h2h_key = (nh, na)
        h2h = _h2h_stats(h2h_history[h2h_key][-H2H_N:])

        # ── 4. xG 特征（可选）──
        xg_home_hist = [x for x in team_history[nh] if x.get("xg_for") is not None][-FORM_N:]
        xg_away_hist = [x for x in team_history[na] if x.get("xg_for") is not None][-FORM_N:]
        hxg = _form_stats(xg_home_hist) if xg_home_hist else None
        axg = _form_stats(xg_away_hist) if xg_away_hist else None

        # ── 5. 赛季进度 ──
        # 粗估：用联赛已完赛场次 / (联赛队数 × (联赛队数-1)) 推算
        lg_n = len(league_goals.get(league, []))
        # 五大联赛每轮约 n/2 场，38 轮总 380 场（单循环）
        est_total = 380
        season_progress = min(1.0, lg_n / est_total) if est_total > 0 else 0.0

        # ── 6. Form 趋势 ──
        home_all = [x for x in team_history[nh]][-FORM_LONG_N:]
        away_all = [x for x in team_history[na]][-FORM_LONG_N:]
        home_short = _form_stats(home_all[-FORM_TREND_N:]) if len(home_all) >= FORM_TREND_N else None
        home_long = _form_stats(home_all) if home_all else None
        away_short = _form_stats(away_all[-FORM_TREND_N:]) if len(away_all) >= FORM_TREND_N else None
        away_long = _form_stats(away_all) if away_all else None
        home_trend = ((home_short["pts"] - home_long["pts"]) if home_short and home_long else 0.0)
        away_trend = ((away_short["pts"] - away_long["pts"]) if away_short and away_long else 0.0)

        # ── 7. 联赛进球趋势 ──
        lg_recent = league_goals.get(league, [])[-50:]
        lg_avg_goals = _avg(lg_recent) if lg_recent else 2.5

        # ── 组装特征向量 ──
        feat = [
            elo_diff_raw + HOME_ADV,   # elo_diff（含主场优势）
            ep_h, ep_d, ep_a,          # elo_prob
            hf["pts"], hf["gf"], hf["ga"], hf["gd"], hf["wr"],  # form_home_*
            af["pts"], af["gf"], af["ga"], af["gd"], af["wr"],  # form_away_*
            hf["pts"] - af["pts"],     # form_pts_diff
            hf["gd"] - af["gd"],       # form_gd_diff
            h2h["home_wr"], h2h["draw_rate"], h2h["avg_goals"],  # h2h_*
            hf["cs"], af["cs"],        # form_*_cs
            (hxg["gd"] if hxg else DEFAULT_XG),   # xg_home_diff
            (axg["gd"] if axg else DEFAULT_XG),   # xg_away_diff
            (hxg["gf"] if hxg else DEFAULT_XG),   # xg_home_for
            (axg["gf"] if axg else DEFAULT_XG),   # xg_away_for
            season_progress,           # season_progress
            home_trend, away_trend,    # form_trend
            elo_diff_raw,              # elo_rating_diff（不含主场优势）
            HOME_ADV,                  # home_advantage
            LEAGUE_MAP.get(league, 5), # league_encoded
            lg_avg_goals,              # total_goals_trend
        ]

        # ── 标签：0=主胜 1=平局 2=客胜 ──
        if hg > ag:
            label = 0
        elif hg == ag:
            label = 1
        else:
            label = 2

        rows.append(feat)
        labels.append(label)
        meta_list.append({
            "home": home, "away": away, "league": league,
            "date": date, "season": season,
            "score": f"{hg}-{ag}", "outcome": label,
        })

        # ── 更新内部状态（用本场结果，供后续场次使用）──
        elo.update([{
            "utcDate": date,
            "homeTeam": {"name": home},
            "awayTeam": {"name": away},
            "score": {"fullTime": {"home": hg, "away": ag}},
        }])
        xg_model.add_match(home, away, m.get("xgHome"), m.get("xgAway"))
        xg_model.finalize()

        team_history[nh].append({
            "gf": hg, "ga": ag, "side": "home",
            "xg_for": m.get("xgHome"), "xg_against": m.get("xgAway"),
            "date": date,
        })
        team_history[na].append({
            "gf": ag, "ga": hg, "side": "away",
            "xg_for": m.get("xgAway"), "xg_against": m.get("xgHome"),
            "date": date,
        })
        h2h_history[h2h_key].append({"hg": hg, "ag": ag})

        total_g = hg + ag
        league_goals[league].append(total_g)

    import numpy as np
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    if verbose:
        print(f"特征矩阵: {X.shape}, 标签分布: 主胜={sum(y==0)}, 平局={sum(y==1)}, 客胜={sum(y==2)}")
        has_xg = sum(1 for r in rows if r[21] != DEFAULT_XG)
        print(f"有 xG 数据的场次: {has_xg}/{len(rows)}")

    return X, y, meta_list


def compute_match_features(home, away, league, elo_model, xg_model, team_history, h2h_history, league_goals):
    """
    为单场即将进行的比赛计算特征向量（供 predict.py 实时预测用）。
    返回: np.ndarray shape(1, N_FEATURES)
    """
    import numpy as np

    nh, na = norm_team(home), norm_team(away)

    # Elo
    elo_h = elo_model.get_rating(home)
    elo_a = elo_model.get_rating(away)
    elo_diff_raw = elo_h - elo_a
    ep_h, ep_d, ep_a = elo_model.predict(home, away)

    # Form
    home_home = [x for x in team_history.get(nh, []) if x["side"] == "home"][-FORM_N:]
    away_away = [x for x in team_history.get(na, []) if x["side"] == "away"][-FORM_N:]
    hf = _form_stats(home_home)
    af = _form_stats(away_away)

    # H2H
    h2h_key = (nh, na)
    h2h = _h2h_stats(h2h_history.get(h2h_key, [])[-H2H_N:])

    # xG
    xg_home_hist = [x for x in team_history.get(nh, []) if x.get("xg_for") is not None][-FORM_N:]
    xg_away_hist = [x for x in team_history.get(na, []) if x.get("xg_for") is not None][-FORM_N:]
    hxg = _form_stats(xg_home_hist) if xg_home_hist else None
    axg = _form_stats(xg_away_hist) if xg_away_hist else None

    # 赛季进度
    lg_n = len(league_goals.get(league, []))
    est_total = 380
    season_progress = min(1.0, lg_n / est_total) if est_total > 0 else 0.0

    # 趋势
    home_all = [x for x in team_history.get(nh, [])][-FORM_LONG_N:]
    away_all = [x for x in team_history.get(na, [])][-FORM_LONG_N:]
    home_short = _form_stats(home_all[-FORM_TREND_N:]) if len(home_all) >= FORM_TREND_N else None
    home_long = _form_stats(home_all) if home_all else None
    away_short = _form_stats(away_all[-FORM_TREND_N:]) if len(away_all) >= FORM_TREND_N else None
    away_long = _form_stats(away_all) if away_all else None
    home_trend = ((home_short["pts"] - home_long["pts"]) if home_short and home_long else 0.0)
    away_trend = ((away_short["pts"] - away_long["pts"]) if away_short and away_long else 0.0)

    lg_recent = league_goals.get(league, [])[-50:]
    lg_avg_goals = _avg(lg_recent) if lg_recent else 2.5

    feat = [
        elo_diff_raw + HOME_ADV,
        ep_h, ep_d, ep_a,
        hf["pts"], hf["gf"], hf["ga"], hf["gd"], hf["wr"],
        af["pts"], af["gf"], af["ga"], af["gd"], af["wr"],
        hf["pts"] - af["pts"],
        hf["gd"] - af["gd"],
        h2h["home_wr"], h2h["draw_rate"], h2h["avg_goals"],
        hf["cs"], af["cs"],
        (hxg["gd"] if hxg else DEFAULT_XG),
        (axg["gd"] if axg else DEFAULT_XG),
        (hxg["gf"] if hxg else DEFAULT_XG),
        (axg["gf"] if axg else DEFAULT_XG),
        season_progress,
        home_trend, away_trend,
        elo_diff_raw,
        HOME_ADV,
        LEAGUE_MAP.get(league, 5),
        lg_avg_goals,
    ]

    return np.array([feat], dtype=np.float32)
