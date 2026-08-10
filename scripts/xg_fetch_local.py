"""
xg_fetch_local.py - 本地 xG 数据抓取（需在你电脑上运行）
从 Understat 抓取五大联赛 xG 数据（Understat 反爬较强，需真实浏览器渲染）。

安装依赖（一次）:
  pip install playwright
  python -m playwright install chromium

用法:
  python scripts/xg_fetch_local.py            # 抓全部 5 大联赛
  python scripts/xg_fetch_local.py --league EPL  # 只抓英超
  python scripts/xg_fetch_local.py --push     # 抓取后自动 git push 到仓库

输出:
  data/xg/EPL.json, La_liga.json, Bundesliga.json, Serie_A.json, Ligue_1.json

说明:
  - Understat 无欧冠数据 → 欧冠 xG 不抓取
  - 建议 Windows 计划任务每天 1 次自动运行（见 README 注释）
"""

import argparse
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
XG_DIR = ROOT / "data" / "xg"

LEAGUES = {
    "EPL": ("EPL", "英超"),
    "La_liga": ("La_liga", "西甲"),
    "Bundesliga": ("Bundesliga", "德甲"),
    "Serie_A": ("Serie_A", "意甲"),
    "Ligue_1": ("Ligue_1", "法甲"),
}


def fetch_league(league_key: str, headless: bool = True) -> dict:
    """用 Playwright 渲染 Understat 联赛页，提取页面内嵌 JSON。"""
    from playwright.sync_api import sync_playwright

    url = f"https://understat.com/league/{league_key}"
    data = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.new_page()
        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            # 等待数据 JSON 注入（Understat 页面会执行 var xxxData = JSON.parse('...')）
            page.wait_for_function(
                "() => document.body.innerHTML.includes('JSON.parse')", timeout=45000
            )
            html = page.content()

            # 提取所有 JSON 数据变量
            for var_name in ["teamsData", "datesData", "playersData", "shotsData", "matchesData", "fixturesData"]:
                pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\);"
                m = re.search(pattern, html, re.S)
                if m:
                    try:
                        raw = m.group(1)
                        # 解码 JS 转义（\\' -> ' 等）
                        raw = raw.replace("\\'", "'").replace('\\\\"', '"').replace('\\\\/', '/')
                        data[var_name] = json.loads(raw)
                        print(f"  ✓ {var_name}: {len(data[var_name]) if hasattr(data[var_name], '__len__') else '?'} 条")
                    except Exception as e:
                        print(f"  - {var_name} 解析失败: {e}")

            # 页面标题确认赛季
            title = page.title()
            print(f"  页面: {title[:60]}")
        except Exception as e:
            print(f"  ✗ 抓取失败: {e}")
        finally:
            browser.close()

    return data


def save_league(league_key: str, data: dict, cn_name: str):
    XG_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "league": league_key,
        "name": cn_name,
        "data": data,
    }
    path = XG_DIR / f"{league_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  → {path} ({path.stat().st_size / 1024:.0f} KB)")


def git_push(message: str):
    """抓取完成后自动提交推送。"""
    git = "git"
    for cmd in [
        ["add", "-A"],
        ["commit", "-m", message],
        ["push", "origin", "main"],
    ]:
        r = subprocess.run([git, *cmd], cwd=ROOT, capture_output=True, text=True)
        print(f"  git {' '.join(cmd[:1])}: {r.returncode == 0}" + (f" | {r.stderr.strip()[:120]}" if r.returncode != 0 else ""))


def main():
    parser = argparse.ArgumentParser(description="Understat xG 本地抓取")
    parser.add_argument("--league", choices=list(LEAGUES.keys()), default=None, help="指定联赛，默认全部")
    parser.add_argument("--headful", action="store_true", help="显示浏览器窗口（调试用）")
    parser.add_argument("--push", action="store_true", help="抓取后自动 git push")
    args = parser.parse_args()

    leagues = [args.league] if args.league else list(LEAGUES.keys())
    print(f"开始抓取 {len(leagues)} 个联赛的 xG 数据（Understat）...")

    for key in leagues:
        code, cn = LEAGUES[key]
        print(f"\n=== {cn} ({key}) ===")
        data = fetch_league(code, headless=not args.headful)
        if data:
            save_league(key, data, cn)
        else:
            print("  ✗ 未获取到数据（可能被反爬拦截，稍后重试或换 --headful 模式）")

    if args.push:
        print("\n提交推送...")
        git_push("chore: 本地更新 xG 数据（Understat）")

    print("\n完成。建议每天运行一次（Windows 计划任务）")


if __name__ == "__main__":
    main()
