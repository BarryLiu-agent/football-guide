"""
jingcai_fetch_local.py - 竞彩 SP 数据本地抓取（需在中国大陆网络环境）
从中国竞彩官方接口抓取 SP 赔率（胜平负/让球/总进球/半全场）。

用法:
  python scripts/jingcai_fetch_local.py             # 抓取并保存
  python scripts/jingcai_fetch_local.py --push      # 抓取后自动 git push

输出:
  data/jingcai.json（含全部已开售比赛 + SP 历史变化 oddsList）

说明:
  - 竞彩接口仅限国内网络访问 → 必须本地运行（GitHub Actions 海外无法访问）
  - 建议 Windows 计划任务每小时运行 1 次（SP 会随时间变化）
  - 接口参考: https://webapi.sporttery.cn/gateway/jc/football/getMatchCalculatorV1.qry
"""

import argparse
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

API_URL = "https://webapi.sporttery.cn/gateway/jc/football/getMatchCalculatorV1.qry"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.sporttery.cn/",
}

# 本站仅覆盖五大联赛 + 欧冠；竞彩接口会同时售卖其他赛事，必须白名单过滤。
LEAGUE_WHITELIST = {
    # 五大联赛（竞彩中文全名/常见别名）
    "英格兰超级联赛", "英超",
    "西班牙甲级联赛", "西甲",
    "意大利甲级联赛", "意甲",
    "德国甲级联赛", "德甲",
    "法国甲级联赛", "法甲",
    # 欧冠
    "欧洲冠军联赛", "欧冠",
}


def fetch(pool_code: str = "hhad,had,ttg,hafu,crs") -> dict:
    """抓取竞彩全部在售比赛与 SP。"""
    params = {"poolCode": pool_code, "channel": "c"}
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    d = r.json()
    if str(d.get("errorCode")) != "0":
        raise RuntimeError(f"接口错误: {d.get('errorMessage')}")
    return d


def normalize(d: dict) -> dict:
    """归一化为统一结构，只保留有用字段。"""
    value = d.get("value", {})
    matches = []
    for day in value.get("matchInfoList", []):
        for m in day.get("subMatchList", []):
            if m.get("sellStatus") not in (1, 2):  # 在售/部分销售都保留
                continue
            league_name = m.get("leagueAllName", "")
            if league_name not in LEAGUE_WHITELIST:
                continue
            # 四种玩法 SP
            had = m.get("had") or {}
            hhad = m.get("hhad") or {}
            ttg = m.get("ttg") or {}
            hafu = m.get("hafu") or {}

            # 比分玩法（波胆）SP：键格式 s{主} s{客}，如 s00s01 = 0:1
            crs = m.get("crs") or {}
            crs_odds = {}
            for k, v in crs.items():
                if k.startswith("s") and "s" in k[1:] and v not in (None, ""):
                    try:
                        hs, as_ = k[1:].split("s")
                        crs_odds[f"{int(hs)}-{int(as_)}"] = v
                    except (ValueError, TypeError):
                        pass

            # SP 历史（用于涨跌监控）
            odds_history = []
            for o in m.get("oddsList", []):
                odds_history.append({
                    "time": o.get("updateTime", ""),
                    "had": {"h": o.get("h"), "d": o.get("d"), "a": o.get("a")},
                    "hhad": {"h": o.get("hh"), "d": o.get("hd"), "a": o.get("ha")},
                })

            matches.append({
                "matchId": m.get("matchId"),
                "matchNumStr": m.get("matchNumStr"),
                "leagueName": m.get("leagueAllName"),
                "homeTeam": m.get("homeTeamAllName"),
                "awayTeam": m.get("awayTeamAllName"),
                "homeRank": (m.get("homeRank") or [""])[0],
                "awayRank": (m.get("awayRank") or [""])[0],
                "matchDate": m.get("matchDate"),
                "matchTime": m.get("matchTime"),
                "matchStatus": m.get("matchStatus"),
                # 胜平负 SP
                "had": {"h": had.get("h"), "d": had.get("d"), "a": had.get("a")},
                # 让球胜平负（goalLine = 让球数）
                "hhad": {
                    "h": hhad.get("h"), "d": hhad.get("d"), "a": hhad.get("a"),
                    "goalLine": hhad.get("goalLineValue"),
                },
                # 总进球 SP（0-7+）。接口键为 s0~s7（s7=7+球），s{i}f 为涨跌标记
                "ttg": {str(i): ttg.get(f"s{i}") for i in range(8)},
                # 半全场 SP（胜胜/胜平/胜负/平胜/平平/平负/负胜/负平/负负）
                "hafu": {
                    "胜胜": hafu.get("hh"), "胜平": hafu.get("hd"), "胜负": hafu.get("ha"),
                    "平胜": hafu.get("dh"), "平平": hafu.get("dd"), "平负": hafu.get("da"),
                    "负胜": hafu.get("ah"), "负平": hafu.get("ad"), "负负": hafu.get("aa"),
                },
                "crsOdds": crs_odds,
                "oddsHistory": odds_history,
            })

    return {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lastUpdateTime": value.get("lastUpdateTime", ""),
        "total": len(matches),
        "matches": matches,
    }


def git_push(message: str):
    """add → commit → pull --rebase（防云端 Actions 并发冲突）→ push，失败重试 3 次。"""
    git = "git"
    for attempt in range(3):
        for cmd in [["add", "-A"], ["commit", "-m", message]]:
            subprocess.run([git, *cmd], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        subprocess.run([git, "pull", "--rebase", "--autostash", "origin", "main"],
                       cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        r = subprocess.run([git, "push", "origin", "main"], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode == 0:
            print("  git push: OK")
            return
        print(f"  git push 失败，重试 {attempt + 1}/3: {r.stderr.strip()[:120]}")
    print("  git push: 失败（请检查网络/凭据）")


def main():
    parser = argparse.ArgumentParser(description="竞彩 SP 本地抓取")
    parser.add_argument("--push", action="store_true", help="抓取后自动 git push")
    args = parser.parse_args()

    print("抓取竞彩 SP 数据...")
    try:
        raw = fetch()
        out = normalize(raw)
    except Exception as e:
        print(f"✗ 抓取失败: {e}")
        return 1

    out_path = ROOT / "data" / "jingcai.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"✓ 在售比赛 {out['total']} 场 -> {out_path}")
    for m in out["matches"][:5]:
        print(f"  {m['matchNumStr']} {m['leagueName']}: {m['homeTeam']} vs {m['awayTeam']} | 胜平负 {m['had']['h']}/{m['had']['d']}/{m['had']['a']}")

    if args.push:
        print("\n提交推送...")
        git_push("chore: 本地更新竞彩 SP 数据")

    return 0


if __name__ == "__main__":
    sys.exit(main())
