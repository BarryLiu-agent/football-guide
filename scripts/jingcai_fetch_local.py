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


def fetch(pool_code: str = "hhad,had,ttg,hafu") -> dict:
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
            if m.get("sellStatus") != 1:  # 只保留在售
                continue
            # 四种玩法 SP
            had = m.get("had") or {}
            hhad = m.get("hhad") or {}
            ttg = m.get("ttg") or {}
            hafu = m.get("hafu") or {}

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
                # 总进球 SP（0-7+）
                "ttg": {str(i): ttg.get(str(i)) for i in range(8)},
                # 半全场 SP（胜胜/胜平/胜负/平胜/平平/平负/负胜/负平/负负）
                "hafu": {
                    "胜胜": hafu.get("hh"), "胜平": hafu.get("hd"), "胜负": hafu.get("ha"),
                    "平胜": hafu.get("dh"), "平平": hafu.get("dd"), "平负": hafu.get("da"),
                    "负胜": hafu.get("ah"), "负平": hafu.get("ad"), "负负": hafu.get("aa"),
                },
                "oddsHistory": odds_history,
            })

    return {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lastUpdateTime": value.get("lastUpdateTime", ""),
        "total": len(matches),
        "matches": matches,
    }


def git_push(message: str):
    git = "git"
    for cmd in [["add", "-A"], ["commit", "-m", message], ["push", "origin", "main"]]:
        r = subprocess.run([git, *cmd], cwd=ROOT, capture_output=True, text=True)
        print(f"  git {' '.join(cmd[:1])}: {r.returncode == 0}" + (f" | {r.stderr.strip()[:120]}" if r.returncode != 0 else ""))


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
