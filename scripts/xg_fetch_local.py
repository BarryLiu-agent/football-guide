# -*- coding: utf-8 -*-
"""
xg_fetch_local.py - Understat xG 数据本地抓取（真实浏览器渲染）
Understat 使用 Cloudflare 反爬：普通 requests 只能拿到空壳页面，
必须用真实浏览器（Playwright）加载后才会有数据（服务端渲染的 HTML 表格）。

用法:
  python scripts/xg_fetch_local.py             # 抓取并保存
  python scripts/xg_fetch_local.py --push      # 抓取后自动 git push

输出:
  data/xg/{PL,PD,BL1,SA,FL1}.json
  结构: { "generatedAt": ..., "league": "PL", "data": { "teams": [...], "players": [...] } }

依赖:
  pip install playwright beautifulsoup4
  python -m playwright install chromium
  （国内网络: 设置 PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright 后安装）

说明:
  - 仅五大联赛（Understat 无欧冠数据）
  - 建议每天运行 1 次（Windows 计划任务）
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

# Understat 联赛路径（无欧冠）
LEAGUES = {
    "PL": "EPL",
    "PD": "La_liga",
    "BL1": "Bundesliga",
    "SA": "Serie_A",
    "FL1": "Ligue_1",
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "Chrome/120.0.0.0 Safari/537.36")


def _num(val):
    """清洗数字：'77.49+6.49' -> 77.49, '12' -> 12.0"""
    try:
        return float(str(val).strip().split("+")[0])
    except (ValueError, AttributeError):
        return 0.0


def fetch_league(league_code: str, understat_path: str) -> dict:
    """Playwright 真实浏览器加载页面 → 解析 HTML 表格。"""
    from bs4 import BeautifulSoup
    from playwright.sync_api import sync_playwright

    url = f"https://understat.com/league/{understat_path}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        # 等待数据表格出现（Cloudflare 校验 + JS 异步渲染球队表，约 15-20 秒）
        html = ""
        for _ in range(40):
            page.wait_for_timeout(3000)
            html = page.content()
            tables = re.findall(r"<table.*?</table>", html, re.S)
            if tables and "<tbody>" in tables[0] and tables[0].count("<tr") > 1:
                break
        browser.close()

    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError(f"{league_code}: 页面无数据表格（可能被 Cloudflare 拦截）")

    teams, players = [], []
    for tb in tables:
        heads = [th.get_text(strip=True) for th in tb.find_all("th")]
        if not heads:
            continue
        if "xG" in heads and "xPTS" in heads:
            # 球队积分榜: № Team M W D L G GA PTS xG xGA xPTS
            for tr in tb.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 12:
                    teams.append({
                        "position": int(tds[0]),
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
            # 球员射手榜: № Player Team Apps Min G A xG xA xG90 xA90
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
    for cmd in [["add", "-A"], ["commit", "-m", message], ["push", "origin", "main"]]:
        r = subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True)
        print(f"  git {cmd[0]}: {'OK' if r.returncode == 0 else r.stderr.strip()[:100]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Understat xG 本地抓取")
    parser.add_argument("--push", action="store_true", help="抓取后自动 git push")
    parser.add_argument("--leagues", nargs="*", default=list(LEAGUES.keys()), help="联赛代码")
    args = parser.parse_args()

    out_dir = ROOT / "data" / "xg"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for code in args.leagues:
        path = LEAGUES.get(code)
        if not path:
            print(f"  {code}: 不支持的联赛")
            continue
        print(f"抓取 {code} ({path})...")
        try:
            data = fetch_league(code, path)
            if not data["teams"] and not data["players"]:
                print(f"  ✗ {code}: 表格为空（休赛期或解析失败）")
                continue
            out = {
                "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "league": code,
                "data": data,
            }
            (out_dir / f"{code}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  ✓ {code}: {len(data['teams'])} 队 / {len(data['players'])} 球员 -> data/xg/{code}.json")
            ok += 1
        except Exception as e:
            print(f"  ✗ {code}: {e}")

    print(f"\n完成: {ok}/{len(args.leagues)} 联赛")
    if args.push and ok:
        print("提交推送...")
        git_push("chore: 本地更新 xG 数据")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
