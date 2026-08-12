"""
fetch_data.py - GitHub Actions 数据抓取脚本
从 Football-Data.org 抓取五大联赛+欧冠比赛数据，应用筛选规则，生成静态 JSON。

用法:
  python scripts/fetch_data.py                    # 常规抓取
  python scripts/fetch_data.py --no-filter         # 不过滤（保存全部比赛）
  python scripts/fetch_data.py --output data/test.json

环境变量:
  FOOTBALL_API_KEY: Football-Data.org API Key（必需）
"""

import json
import os
import sys
import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Fix Windows encoding for GitHub Actions
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Configuration ──────────────────────────────────────────

API_KEY = os.environ.get("FOOTBALL_API_KEY", "")
if not API_KEY:
    print("ERROR: FOOTBALL_API_KEY 环境变量未设置！")
    sys.exit(1)

API_BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# Competition codes (Football-Data.org free tier 实际支持列表)
COMPETITIONS = {
    "PL": "英超",
    "PD": "西甲",
    "SA": "意甲",
    "BL1": "德甲",
    "FL1": "法甲",
    "CL": "欧冠",
}

# Project root (this script lives in scripts/, project is parent)
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "data" / "fixtures.json"
CONFIG_DIR = ROOT / "config"

# ── Helpers ────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_date_range():
    """返回过去7天到未来35天的日期范围（覆盖需求中的±30天）"""
    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    date_to = (now + timedelta(days=35)).strftime("%Y-%m-%d")
    return date_from, date_to

def fetch_competition(code, date_from, date_to, retry=2):
    """抓取单个联赛的比赛数据（带异常保护与限流重试）。
    返回 {"matches": [...], "error": str_or_None}，用于区分无比赛和真正失败。"""
    url = f"{API_BASE}/competitions/{code}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    except Exception as e:
        print(f"  ✗ {code}: 网络错误 {e}")
        if retry > 0:
            time.sleep(10)
            return fetch_competition(code, date_from, date_to, retry - 1)
        return {"matches": [], "error": f"网络错误: {e}"}
    if r.status_code == 200:
        try:
            return {"matches": r.json().get("matches", []), "error": None}
        except Exception as e:
            return {"matches": [], "error": f"JSON 解析失败: {e}"}
    elif r.status_code == 429:
        if retry > 0:
            print(f"  ⚠ {code}: 请求太频繁(429)，等待60秒后重试...")
            time.sleep(60)
            return fetch_competition(code, date_from, date_to, retry - 1)
        print(f"  ✗ {code}: 429 重试耗尽")
        return {"matches": [], "error": "请求太频繁(429)"}
    else:
        err_text = r.text[:200]
        print(f"  ✗ {code}: HTTP {r.status_code} - {err_text}")
        return {"matches": [], "error": f"HTTP {r.status_code}: {err_text}"}


def fetch_standings():
    """抓取各联赛积分榜 → data/standings.json（供排名对比分析）"""
    standings = {}
    for code in COMPETITIONS:
        try:
            r = requests.get(f"{API_BASE}/competitions/{code}/standings", headers=HEADERS, timeout=20)
            if r.status_code == 429:
                # Football-Data 免费 10次/分钟：等待后重试一次
                print(f"  - {code} 积分榜限流(429)，等待12秒重试...")
                time.sleep(12)
                r = requests.get(f"{API_BASE}/competitions/{code}/standings", headers=HEADERS, timeout=20)
            if r.status_code != 200:
                print(f"  - {code} 积分榜: HTTP {r.status_code}")
                continue
            data = r.json()
            table = None
            for s in data.get("standings", []):
                if s.get("type") == "TOTAL":
                    table = s.get("table", [])
                    break
            if not table:
                continue
            # 季前占位积分榜（全部球队 0 场/0 分，position 全 1）无分析价值，跳过
            if table and all((t.get("playedGames") or 0) == 0 for t in table):
                print(f"  - {code} 积分榜为季前占位（0 场），跳过")
                continue
            standings[code] = [{
                "position": t.get("position"),
                "team": (t.get("team") or {}).get("name", ""),
                "shortName": (t.get("team") or {}).get("shortName", ""),
                "playedGames": t.get("playedGames"),
                "points": t.get("points"),
                "goalDifference": t.get("goalDifference"),
                "won": t.get("won"), "draw": t.get("draw"), "lost": t.get("lost"),
                "goalsFor": t.get("goalsFor"), "goalsAgainst": t.get("goalsAgainst"),
            } for t in table]
        except Exception as e:
            print(f"  - {code} 积分榜失败: {e}")
    out = {"generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "standings": standings}
    (ROOT / "data" / "standings.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ 积分榜: {len(standings)} 个联赛 -> data/standings.json")

def is_big_team(team_name, flat_list):
    """判断球队是否在豪门列表中"""
    return team_name in flat_list

def is_rivalry_match(home, away, pairs):
    """判断是否为德比/宿敌对阵"""
    for pair in pairs:
        t1, t2 = pair["team1"], pair["team2"]
        if (home == t1 and away == t2) or (home == t2 and away == t1):
            return pair["name"]
    return None

def is_star_match(home, away, star_teams):
    """判断是否有关注球星出场"""
    return home in star_teams or away in star_teams

# ── Main ───────────────────────────────────────────────────

def main(no_filter=False, no_standings=False):
    print(f"╔══════════════════════════════════════════╗")
    print(f"║  足球比赛观看指南 - 数据抓取脚本         ║")
    print(f"║  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}             ║")
    print(f"╚══════════════════════════════════════════╝")

    # Load configs
    big_teams = load_json(CONFIG_DIR / "big_teams.json")
    rivalry_pairs = load_json(CONFIG_DIR / "rivalry_pairs.json")
    star_players = load_json(CONFIG_DIR / "star_players.json")

    flat_big_teams = big_teams["flatList"]
    star_teams_set = set(p["team"] for p in star_players["players"])

    date_from, date_to = get_date_range()
    print(f"\n 日期范围: {date_from} ~ {date_to}")

    # Fetch all competitions
    all_matches = []
    errors = []

    for code, name in COMPETITIONS.items():
        print(f"   抓取 {name}({code})...", end=" ", flush=True)
        result = fetch_competition(code, date_from, date_to)
        matches = result["matches"]
        error = result.get("error")
        if error:
            errors.append({"code": code, "name": name, "reason": error})
            print(f"✗ {error}")
        elif matches:
            print(f"✓ {len(matches)} 场比赛")
        else:
            # 空数据≠错误；欧冠等赛事在休赛期/资格赛阶段会自然无比赛
            print("○ 当前窗口无比赛")
        all_matches.extend(matches)
        # Football-Data 免费 10 req/min：11 联赛需间隔，防 429
        time.sleep(4)

    print(f"\n 总计抓取: {len(all_matches)} 场比赛")

    # Apply filtering rules
    notable_count = 0
    rivalry_count = 0
    star_count = 0
    big_team_count = 0

    for match in all_matches:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]

        if no_filter:
            match["notable"] = True
            match["notableReasons"] = ["--no-filter"]
            notable_count += 1
            continue

        reasons = []

        # Rule 1: Big team
        if is_big_team(home, flat_big_teams) or is_big_team(away, flat_big_teams):
            big_teams_in_match = [t for t in [home, away] if is_big_team(t, flat_big_teams)]
            reasons.append({"type": "豪门", "teams": big_teams_in_match})
            big_team_count += 1

        # Rule 2: Rivalry
        rivalry_name = is_rivalry_match(home, away, rivalry_pairs["pairs"])
        if rivalry_name:
            reasons.append({"type": "德比", "name": rivalry_name})
            rivalry_count += 1

        # Rule 3: Star player
        if is_star_match(home, away, star_teams_set):
            stars_in_match = [p["name"] for p in star_players["players"]
                              if p["team"] in [home, away]]
            reasons.append({"type": "球星", "players": stars_in_match})
            star_count += 1

        match["notable"] = len(reasons) > 0
        if reasons:
            match["notableReasons"] = reasons
            notable_count += 1

    print(f"  知名比赛: {notable_count} 场")
    print(f"    - 豪门参与: {big_team_count} 场")
    print(f"    - 德比/宿敌: {rivalry_count} 场")
    print(f"    - 球星出场: {star_count} 场")
    if errors:
        print(f"  ⚠ 抓取失败联赛: {[e['name'] for e in errors]}")

    # Build output
    output = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dateRange": {"from": date_from, "to": date_to},
        "competitions": {code: name for code, name in COMPETITIONS.items()},
        "totalMatches": len(all_matches),
        "notableMatches": notable_count,
        "errors": errors,
        "matches": all_matches,
    }

    # Write
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已保存到: {OUTPUT_FILE}")
    print(f"   文件大小: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    # Also write a minimal stats file
    stats_file = ROOT / "data" / "stats.json"
    stats = {
        "generatedAt": output["generatedAt"],
        "totalMatches": output["totalMatches"],
        "notableMatches": output["notableMatches"],
        "competitions": output["competitions"],
        "byCompetition": {},
    }
    for match in all_matches:
        code = match["competition"]["code"]
        if code not in stats["byCompetition"]:
            stats["byCompetition"][code] = 0
        stats["byCompetition"][code] += 1

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"   统计文件: {stats_file}")

    # 积分榜（供排名对比分析；高频赛果刷新时跳过以节省请求）
    if not no_standings:
        fetch_standings()

    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="足球数据抓取")
    parser.add_argument("--no-filter", action="store_true",
                        help="不过滤比赛（保存全部）")
    parser.add_argument("--no-standings", action="store_true",
                        help="跳过积分榜抓取（高频赛果刷新用）")
    parser.add_argument("--output", type=str,
                        help="自定义输出路径")
    args = parser.parse_args()

    if args.output:
        OUTPUT_FILE = Path(args.output)

    sys.exit(main(no_filter=args.no_filter, no_standings=args.no_standings))
