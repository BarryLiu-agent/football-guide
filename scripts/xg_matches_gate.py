# -*- coding: utf-8 -*-
"""
xg_matches_gate.py - 单场 xG 抓取的"智能门控"
在运行 xg_fetch_local.py --matches 前判断当前是否有比赛正在打或刚结束：
  - 有（进行中/今天有比赛）→ 返回 0，允许抓取（比赛日 15 分钟一次才有意义）
  - 没有（休赛期/无比赛日）→ 返回 1，跳过抓取（不弹浏览器、不空转）

判断依据：data/fixtures.json（football-data.org，云端每 30 分钟更新）
  - 有 IN_PLAY/PAUSED/LIVE 状态 → 比赛进行中，抓
  - 未来 3 小时内有比赛开赛（TIMED/SCHEDULED）→ 临近开赛，抓（覆盖开赛前 60-75 分钟首发窗口）
  - 过去 3 小时内有比赛刚结束 → 抓（完赛 xG 补录）
  - 否则 → 无比赛窗口，跳过

用法:
  python scripts/xg_matches_gate.py
  exit code 0 = 应该抓取, 1 = 跳过（供 bat 判断）
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOW = datetime.now(timezone.utc)
LOOKBACK = timedelta(hours=3)   # 刚结束的比赛窗口
LOOKAHEAD = timedelta(hours=3)  # 临近开赛窗口

LIVE_STATUS = {"IN_PLAY", "PAUSED", "LIVE", "SUSPENDED", "EXTRA_TIME", "PENALTY_SHOOTOUT"}


def has_live_or_near_matches() -> bool:
    path = ROOT / "data" / "fixtures.json"
    if not path.exists():
        # 无赛程文件（本地从未拉过）→ 保守放行（允许抓，由 xg 脚本自身容错）
        return True
    try:
        with open(path, encoding="utf-8") as f:
            fx = json.load(f)
    except Exception:
        return True  # 文件损坏 → 放行，不因门控卡死

    live = 0
    soon = 0
    recent_finish = 0
    for m in fx.get("matches", []):
        status = m.get("status", "")
        kick = m.get("utcDate") or ""
        try:
            kt = datetime.fromisoformat(kick.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if status in LIVE_STATUS:
            live += 1
        elif status in ("TIMED", "SCHEDULED"):
            if NOW - LOOKBACK <= kt <= NOW + LOOKAHEAD:
                soon += 1
        elif status == "FINISHED":
            if NOW - LOOKBACK <= kt <= NOW:
                recent_finish += 1

    if live > 0:
        print(f"有 {live} 场进行中 → 抓取")
        return True
    if soon > 0:
        print(f"未来 3 小时 {soon} 场开赛 → 抓取")
        return True
    if recent_finish > 0:
        print(f"过去 3 小时 {recent_finish} 场刚完赛 → 抓取(补录)")
        return True
    print("当前无进行中/临近比赛 → 跳过")
    return False


if __name__ == "__main__":
    ok = has_live_or_near_matches()
    sys.exit(0 if ok else 1)
