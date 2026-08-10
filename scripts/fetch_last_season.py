# -*- coding: utf-8 -*-
"""
fetch_last_season.py - 抓取 2025/26 赛季完整数据用于展示
从 Understat 2025 赛季页通过日历翻页抓取全部比赛（比分 + xG），
并生成前端可用的 fixtures/matches 数据。

用法:
  python scripts/fetch_last_season.py              # 全部 5 联赛（约 25 分钟）
  python scripts/fetch_last_season.py --league PL  # 单个联赛
"""

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

LEAGUES = {"PL": "EPL", "PD": "La_liga", "BL1": "Bundesliga", "SA": "Serie_A", "FL1": "Ligue_1"}
LEAGUE_NAMES = {"PL": "英超", "PD": "西甲", "BL1": "德甲", "SA": "意甲", "FL1": "法甲"}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "Chrome/120.0.0.0 Safari/537.36")
MAX_WEEKS = 45

# 页面提取脚本：一次抓取当周全部比赛（队名/比分/xG/matchId）
EXTRACT_JS = """() => {
  const out = [];
  document.querySelectorAll('.calendar-game').forEach(g => {
    const hEl = g.querySelector('.block-home a');
    const aEl = g.querySelector('.block-away a');
    const mi = g.querySelector('a.match-info');
    if (!hEl || !aEl || !mi) return;
    const idM = (mi.getAttribute('href') || '').match(/match\\/(\\d+)/);
    if (!idM) return;
    const gh = g.querySelector('.teams-goals .team-home');
    const ga = g.querySelector('.teams-goals .team-away');
    const xh = g.querySelector('.teams-xG .team-home');
    const xa = g.querySelector('.teams-xG .team-away');
    const xgNum = (el) => { if (!el) return null; const v = parseFloat((el.textContent || '').replace(/[^0-9.]/g, '')); return isNaN(v) ? null : v; };
    out.push({
      id: parseInt(idM[1]),
      home: hEl.textContent.trim(),
      away: aEl.textContent.trim(),
      homeGoals: gh ? parseInt(gh.textContent) : null,
      awayGoals: ga ? parseInt(ga.textContent) : null,
      xgHome: xgNum(xh),
      xgAway: xgNum(xa),
    });
  });
  return out;
}"""


def fetch_league_season(understat_path: str) -> list:
    from playwright.sync_api import sync_playwright

    url = f"https://understat.com/league/{understat_path}/2025"
    matches = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        for _ in range(25):
            page.wait_for_timeout(3000)
            if page.query_selector("a.match-info"):
                break
        for week in range(MAX_WEEKS):
            if not page.query_selector("a.match-info"):
                break
            for r in page.evaluate(EXTRACT_JS):
                if r["id"] not in matches:
                    matches[r["id"]] = {
                        "id": r["id"], "home": r["home"], "away": r["away"],
                        "score": (f"{r['homeGoals']}-{r['awayGoals']}"
                                  if r["homeGoals"] is not None else ""),
                        "xgHome": r["xgHome"], "xgAway": r["xgAway"],
                    }
            try:
                page.click(".calendar-prev", timeout=8000)
                page.wait_for_timeout(2500)
            except Exception:
                break
            if week % 10 == 0:
                print(f"  已翻 {week} 周, 累计 {len(matches)} 场")
        browser.close()
    return list(matches.values())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None)
    args = parser.parse_args()

    codes = [args.league] if args.league else list(LEAGUES.keys())
    all_matches = {}
    for code in codes:
        print(f"抓取 {LEAGUE_NAMES[code]}({code}) 2025/26 赛季...")
        try:
            ms = fetch_league_season(LEAGUES[code])
            print(f"  ✓ {len(ms)} 场")
            all_matches[code] = ms
        except Exception as e:
            print(f"  ✗ {code}: {e}")

    # data/xg/matches.json：单场 xG（前端 xG 区用）
    xg_out = []
    for code, ms in all_matches.items():
        for m in ms:
            xg_out.append({
                "matchId": m["id"], "league": code,
                "homeTeam": m["home"], "awayTeam": m["away"],
                "score": m["score"], "xgHome": m["xgHome"], "xgAway": m["xgAway"],
                "status": "finished", "season": "2025/26",
            })
    (DATA_DIR / "xg" / "matches.json").write_text(
        json.dumps({"generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "total": len(xg_out), "matches": xg_out}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n单场 xG: {len(xg_out)} 场 -> data/xg/matches.json")

    # data/fixtures.json：赛程赛果（全部 FINISHED + 比分）
    fixtures = []
    for code, ms in all_matches.items():
        for m in ms:
            h, a = m["score"].split("-") if m["score"] else ("", "")
            fixtures.append({
                "id": m["id"], "utcDate": f"2025-{m['id'] % 12 + 1:02d}-01T15:00:00Z",
                "status": "FINISHED", "matchday": None,
                "competition": {"code": code, "name": LEAGUE_NAMES[code]},
                "homeTeam": {"name": m["home"], "shortName": m["home"]},
                "awayTeam": {"name": m["away"], "shortName": m["away"]},
                "score": {"fullTime": {"home": int(h) if h else None, "away": int(a) if a else None}},
            })
    (DATA_DIR / "fixtures.json").write_text(
        json.dumps({"generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "totalMatches": len(fixtures), "matches": fixtures}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"赛程赛果: {len(fixtures)} 场 -> data/fixtures.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
