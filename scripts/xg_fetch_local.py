# -*- coding: utf-8 -*-
"""
xg_fetch_local.py - Understat xG 数据本地抓取（真实浏览器渲染）
Understat 使用 Cloudflare 反爬：必须用真实浏览器（Playwright）加载。

用法:
  python scripts/xg_fetch_local.py                 # 赛季球队/球员 xG 榜（每天 1 次）
  python scripts/xg_fetch_local.py --matches       # 单场比赛实时 xG（比赛日每 10-15 分钟）
  python scripts/xg_fetch_local.py --push          # 抓取后自动 git push
  python scripts/xg_fetch_local.py --all --push    # 全部 + 推送

输出:
  data/xg/{PL,PD,BL1,SA,FL1}.json   赛季球队榜 + 射手榜
  data/xg/matches.json              单场实时 xG（含比分、两队 xG）

依赖:
  pip install playwright beautifulsoup4
  python -m playwright install chromium
  （国内网络: PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright）
"""

import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

LEAGUES = {
    "PL": "EPL",
    "PD": "La_liga",
    "BL1": "Bundesliga",
    "SA": "Serie_A",
    "FL1": "Ligue_1",
}
# Understat 联赛代码 -> 我们用的代码（用于匹配前端）
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "Chrome/120.0.0.0 Safari/537.36")


def _num(val):
    try:
        return float(str(val).strip().split("+")[0])
    except (ValueError, AttributeError):
        return 0.0


def _browser():
    from playwright.sync_api import sync_playwright
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
    ctx = browser.new_context(user_agent=UA, viewport={"width": 1400, "height": 900})
    return p, browser, ctx


def _wait_data(page, min_tables=1, min_size=40000, marker=None):
    """等待页面数据渲染完成，返回 HTML。"""
    html = ""
    for _ in range(40):
        page.wait_for_timeout(3000)
        html = page.content()
        ok = len(html) > min_size
        if marker:
            ok = ok and marker in html
        if ok:
            return html
    return html


def fetch_league(league_code: str, understat_path: str) -> dict:
    """赛季球队 xG 榜 + 射手榜。"""
    from bs4 import BeautifulSoup

    url = f"https://understat.com/league/{understat_path}"
    p, browser, ctx = _browser()
    page = ctx.new_page()
    page.goto(url, timeout=60000, wait_until="domcontentloaded")
    html = _wait_data(page, marker="<tbody>")
    browser.close(); p.stop()

    soup = BeautifulSoup(html, "lxml")
    teams, players = [], []
    for tb in soup.find_all("table"):
        heads = [th.get_text(strip=True) for th in tb.find_all("th")]
        if not heads:
            continue
        if "xG" in heads and "xPTS" in heads:
            for tr in tb.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 12:
                    teams.append({
                        "position": int(tds[0] or 0),
                        "title": tds[1],
                        "played": int(tds[2] or 0),
                        "wins": int(tds[3] or 0),
                        "draws": int(tds[4] or 0),
                        "losses": int(tds[5] or 0),
                        "goalsFor": int(tds[6] or 0),
                        "goalsAgainst": int(tds[7] or 0),
                        "pts": int(tds[8] or 0),
                        "xG": _num(tds[9]),
                        "xGA": _num(tds[10]),
                        "xPTS": _num(tds[11]),
                    })
        elif "Player" in heads and "xG" in heads:
            for tr in tb.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 9:
                    players.append({
                        "rank": tds[0],
                        "player_name": tds[1],
                        "team_title": tds[2],
                        "apps": tds[3],
                        "minutes": int(tds[4] or 0),
                        "goals": int(tds[5] or 0),
                        "assists": int(tds[6] or 0),
                        "xG": _num(tds[7]),
                        "xA": _num(tds[8]),
                    })
    return {"teams": teams, "players": players}


def fetch_matches() -> list:
    """单场比赛实时 xG：遍历五大联赛页拿比赛链接 → 比赛页提取 xG。"""
    results = []
    for code, path in LEAGUES.items():
        try:
            links = _get_match_links(path)
            if not links:
                print(f"  {code}: 无比赛链接（休赛期或未开赛）")
                continue
            for mid in links:
                try:
                    info = _get_match_info(mid)
                    if info:
                        info["league"] = code
                        results.append(info)
                except Exception as e:
                    print(f"  {code} match {mid}: {e}")
        except Exception as e:
            print(f"  {code}: {e}")
    return results


def _get_match_links(understat_path: str) -> list:
    """从联赛页提取比赛链接（历史赛季页有最近比赛）。"""
    url = f"https://understat.com/league/{understat_path}/2025"
    p, browser, ctx = _browser()
    page = ctx.new_page()
    page.goto(url, timeout=60000, wait_until="domcontentloaded")
    html = _wait_data(page, min_size=100000)
    browser.close(); p.stop()
    links = set(re.findall(r'href="(/match/\d+)"', html))
    return [int(l.split("/")[-1]) for l in links]


def _get_match_info(match_id: int) -> dict or None:
    """比赛页提取 match_info JSON（队名、比分、两队 xG）。"""
    p, browser, ctx = _browser()
    page = ctx.new_page()
    page.goto(f"https://understat.com/match/{match_id}", timeout=60000, wait_until="domcontentloaded")
    html = _wait_data(page, min_size=50000, marker="match_info")
    browser.close(); p.stop()

    m = re.search(r"match_info\s*=\s*JSON\.parse\('(.+?)'\)", html, re.S)
    if not m:
        return None
    raw = m.group(1).encode().decode("unicode_escape")
    data = json.loads(raw)
    return {
        "matchId": data.get("id"),
        "homeTeam": data.get("team_h") or data.get("h_title"),
        "awayTeam": data.get("team_a") or data.get("a_title"),
        "homeGoals": data.get("h_goals"),
        "awayGoals": data.get("a_goals"),
        "xgHome": data.get("h_xg"),
        "xgAway": data.get("a_xg"),
        "date": data.get("date"),
        "season": data.get("season"),
    }


def git_push(message: str):
    for cmd in [["add", "-A"], ["commit", "-m", message], ["push", "origin", "main"]]:
        r = subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True)
        print(f"  git {cmd[0]}: {'OK' if r.returncode == 0 else r.stderr.strip()[:100]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Understat xG 本地抓取")
    parser.add_argument("--matches", action="store_true", help="抓取单场比赛实时 xG")
    parser.add_argument("--leagues", nargs="*", default=list(LEAGUES.keys()))
    parser.add_argument("--push", action="store_true", help="抓取后自动 git push")
    args = parser.parse_args()

    out_dir = ROOT / "data" / "xg"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.matches:
        print("抓取单场比赛 xG...")
        matches = fetch_matches()
        out = {
            "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total": len(matches),
            "matches": matches,
        }
        (out_dir / "matches.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"✓ {len(matches)} 场比赛 -> data/xg/matches.json")
        for m in matches[:5]:
            print(f"  {m['homeTeam']} {m['homeGoals']}-{m['awayGoals']} {m['awayTeam']} | xG {m['xgHome']} vs {m['xgAway']}")
    else:
        ok = 0
        for code in args.leagues:
            path = LEAGUES.get(code)
            if not path:
                continue
            print(f"抓取 {code} ({path})...")
            try:
                data = fetch_league(code, path)
                if not data["teams"] and not data["players"]:
                    print(f"  ✗ {code}: 表格为空")
                    continue
                out = {
                    "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "league": code,
                    "data": data,
                }
                (out_dir / f"{code}.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  ✓ {code}: {len(data['teams'])} 队 / {len(data['players'])} 球员")
                ok += 1
            except Exception as e:
                print(f"  ✗ {code}: {e}")
        print(f"完成: {ok}/{len(args.leagues)} 联赛")

    if args.push:
        print("提交推送...")
        git_push("chore: 本地更新 xG 数据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
