# -*- coding: utf-8 -*-
"""
xg_fetch_local.py - Understat xG 数据本地抓取（真实浏览器渲染）
Understat 使用 Cloudflare 反爬：必须用真实浏览器（Playwright）加载。

用法:
  python scripts/xg_fetch_local.py                 # 赛季球队/球员 xG 榜（每天 1 次）
  python scripts/xg_fetch_local.py --push          # 抓取后自动 git push

输出:
  data/xg/{PL,PD,BL1,SA,FL1,CL}.json   赛季球队榜 + 射手榜

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


# 共享 Playwright 实例：重复 start/stop 在新版驱动下会报
# "Sync API inside the asyncio loop"，改为整个进程复用一次，最后统一关闭。
_PW = {"p": None, "browser": None}


def _browser():
    from playwright.sync_api import sync_playwright
    if _PW["p"] is None:
        p = sync_playwright().start()
        browser = p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
        _PW["p"], _PW["browser"] = p, browser
    ctx = _PW["browser"].new_context(user_agent=UA, viewport={"width": 1400, "height": 900})
    return _PW["p"], _PW["browser"], ctx


def _close_browser():
    """进程结束时统一关闭 Playwright。"""
    if _PW["browser"] is not None:
        try:
            _PW["browser"].close()
        except Exception:
            pass
        try:
            _PW["p"].stop()
        except Exception:
            pass
        _PW["p"] = _PW["browser"] = None


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
    ctx.close()

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


def git_push(message: str):
    """add → commit → pull --rebase（防云端 Actions 并发冲突）→ push，失败重试 3 次。"""
    for attempt in range(3):
        for cmd in [["add", "-A"], ["commit", "-m", message]]:
            subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"],
                       cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        r = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode == 0:
            print("  git push: OK")
            return
        print(f"  git push 失败，重试 {attempt + 1}/3: {r.stderr.strip()[:120]}")
    print("  git push: 失败（请检查网络/凭据）")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Understat xG 本地抓取（球队榜/球员榜）")
    parser.add_argument("--season", type=int, default=None, help="Understat 赛季标签（如 2026 = 2026/27 赛季）")
    parser.add_argument("--leagues", nargs="*", default=list(LEAGUES.keys()))
    parser.add_argument("--push", action="store_true", help="抓取后自动 git push")
    args = parser.parse_args()

    out_dir = ROOT / "data" / "xg"
    out_dir.mkdir(parents=True, exist_ok=True)

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
                json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print(f"  ✓ {code}: {len(data['teams'])} 队 / {len(data['players'])} 球员")
            ok += 1
        except Exception as e:
            print(f"  ✗ {code}: {e}")
    print(f"完成: {ok}/{len(args.leagues)} 联赛")

    _close_browser()

    if args.push:
        print("提交推送...")
        git_push("chore: 本地更新 xG 数据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
