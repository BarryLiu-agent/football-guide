"""
odds_fetcher.py - 博彩赔率抓取
从可配置数据源抓取五大联赛+欧冠赔率，归一化后生成 data/odds/*.json。

用法:
  python scripts/odds_fetcher.py

可扩展接口:
  OddsSource (抽象基类) - 实现 fetch(league_code) -> list[dict]
    已有实现: BetExplorerSource (默认, 公开页抓取), TheOddsApiSource (可选)
  新数据源: 继承 OddsSource, 在 config/odds_sources.json 注册即可
"""

import argparse
import io
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Windows/CI 编码修复
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
ODDS_DIR = DATA_DIR / "odds"


# ── 抽象接口 ─────────────────────────────────────────────

class OddsSource:
    """赔率数据源抽象基类。新数据源继承并实现 fetch() 即可。"""

    def __init__(self, config: dict):
        self.config = config

    def fetch(self, league_code: str) -> list:
        """抓取单个联赛赔率，返回归一化比赛列表。"""
        raise NotImplementedError

    def normalize(self, raw: dict) -> dict:
        """归一化单场比赛结构。"""
        raise NotImplementedError


# ── BetExplorer 实现（默认）──────────────────────────────

class BetExplorerSource(OddsSource):
    """从 BetExplorer 公开页面抓取赔率（无需注册）。"""

    LEAGUE_MAP = {
        "PL": "football/england/premier-league",
        "PD": "football/spain/laliga",
        "SA": "football/italy/serie-a",
        "BL1": "football/germany/bundesliga",
        "FL1": "football/france/ligue-1",
        "CL": "football/europe/champions-league",
    }
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def fetch(self, league_code: str) -> list:
        if league_code not in self.LEAGUE_MAP:
            print(f"    - {league_code}: 未配置联赛路径, 跳过")
            return []

        url = f"{self.config['baseUrl']}/{self.LEAGUE_MAP[league_code]}/fixtures/"
        try:
            r = requests.get(url, headers=self.HEADERS, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print(f"    - {league_code}: 请求失败 {e}")
            return []

        return self._parse(r.text, league_code)

    def _parse(self, html: str, league_code: str) -> list:
        """解析 BetExplorer 比赛表格。尽力而为：结构变化时返回空列表而非崩溃。"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("    - 需要 beautifulsoup4: pip install beautifulsoup4")
            return []

        matches = []
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="table-main--leaguefixtures")
        if not table:
            table = soup.find("table", class_="table-main")
        if not table:
            return []

        for row in table.find_all("tr"):
            # 只处理含赔率的行
            odds_cells = row.find_all("td", class_="table-main__odds")
            if not odds_cells:
                continue
            try:
                # 队名: <a class="in-match"><span>Home</span> - <span>Away</span></a>
                link = row.find("a", class_="in-match")
                if not link:
                    continue
                spans = link.find_all("span")
                if len(spans) < 2:
                    continue
                home = spans[0].get_text(strip=True)
                away = spans[1].get_text(strip=True)
                match_url = link.get("href", "")

                # h2h 赔率: 前 3 个 odds cell 的 data-odd
                h2h = []
                for cell in odds_cells[:3]:
                    btn = cell.find("button", attrs={"data-odd": True})
                    if btn:
                        try:
                            h2h.append(float(btn["data-odd"]))
                        except (ValueError, KeyError):
                            pass
                if len(h2h) < 3:
                    continue

                matches.append(self.normalize({
                    "league": league_code,
                    "home": home, "away": away,
                    "matchUrl": match_url,
                    "h2h": {"home": h2h[0], "draw": h2h[1], "away": h2h[2]},
                }))
            except Exception:
                continue
        return matches

    def normalize(self, raw: dict) -> dict:
        return {
            "league": raw["league"],
            "homeTeam": raw["home"],
            "awayTeam": raw["away"],
            "kickoff": raw.get("kickoff", ""),
            "matchUrl": raw.get("matchUrl", ""),
            "markets": {
                "h2h": raw["h2h"],
                "totals": raw.get("totals"),
                "spreads": raw.get("spreads"),
            },
            "bookmakers": 1,
            "source": "betexplorer",
        }


# ── The Odds API 实现（可选降级）─────────────────────────

class TheOddsApiSource(OddsSource):
    """The Odds API (https://the-odds-api.com) - 需要 ODDS_API_KEY 环境变量。"""

    LEAGUE_MAP = {
        "PL": "soccer_epl",
        "PD": "soccer_spain_la_liga",
        "SA": "soccer_italy_serie_a",
        "BL1": "soccer_germany_bundesliga",
        "FL1": "soccer_france_ligue_one",
        # 欧冠正赛抽签后改为 soccer_uefa_champs_league
        "CL": "soccer_uefa_champs_league_qualification",
        # 扩展赛事（The Odds API 覆盖的其他可博彩联赛）
        "CH": "soccer_efl_champ",
        "EL1": "soccer_england_league1",
        "ED": "soccer_netherlands_eredivisie",
        "PPL": "soccer_portugal_primeira_liga",
        "TSL": "soccer_turkey_super_league",
        "SPL": "soccer_spl",
        "BDF": "soccer_belgium_first_div",
        "BSA": "soccer_brazil_campeonato",
        "MLS": "soccer_usa_mls",
        "JLG": "soccer_japan_j_league",
        "KL1": "soccer_korea_kleague1",
    }

    def fetch(self, league_code: str) -> list:
        api_key = os.environ.get(self.config.get("apiKeyEnv", "ODDS_API_KEY"), "")
        if not api_key:
            return []
        sport = self.LEAGUE_MAP.get(league_code)
        if not sport:
            return []

        params = {
            "apiKey": api_key,
            "regions": ",".join(self.config.get("regions", ["eu"])),
            "markets": ",".join(self.config.get("markets", ["h2h"])),
        }
        url = f"{self.config['baseUrl']}/sports/{sport}/odds"
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return [self.normalize(x) for x in r.json()]
        except Exception as e:
            print(f"    - {league_code}: TheOddsApi 失败 {e}")
            return []

    def normalize(self, raw: dict) -> dict:
        home_name = raw.get("home_team", "")
        away_name = raw.get("away_team", "")
        h2h = {"home": [], "draw": [], "away": []}
        totals = {"over": [], "under": []}
        for bm in raw.get("bookmakers", []):
            for market in bm.get("markets", []):
                key = market["key"]
                if key == "h2h":
                    for o in market["outcomes"]:
                        if o["name"] == home_name:
                            h2h["home"].append(o["price"])
                        elif o["name"] == away_name:
                            h2h["away"].append(o["price"])
                        elif o["name"].lower() == "draw":
                            h2h["draw"].append(o["price"])
                elif key == "totals":
                    for o in market["outcomes"]:
                        if o["name"].lower() == "over":
                            totals["over"].append(o["price"])
                        elif o["name"].lower() == "under":
                            totals["under"].append(o["price"])
        med = lambda xs: statistics.median(xs) if xs else None
        return {
            "league": raw.get("sport_key", ""),
            "homeTeam": raw.get("home_team", ""),
            "awayTeam": raw.get("away_team", ""),
            "kickoff": raw.get("commence_time", ""),
            "matchUrl": "",
            "markets": {
                "h2h": {"home": med(h2h["home"]), "draw": med(h2h["draw"]), "away": med(h2h["away"])},
                "totals": {"over": med(totals["over"]), "under": med(totals["under"])},
                "spreads": None,
            },
            "bookmakers": len(raw.get("bookmakers", [])),
            "source": "theoddsapi",
        }


# ── 源工厂（配置驱动）────────────────────────────────────

SOURCE_REGISTRY = {
    "betexplorer": BetExplorerSource,
    "theoddsapi": TheOddsApiSource,
}


def build_sources(config: dict) -> list:
    """根据 config/odds_sources.json 实例化启用的数据源。"""
    sources = []
    for name, cfg in config.get("sources", {}).items():
        if not cfg.get("enabled", False):
            continue
        cls = SOURCE_REGISTRY.get(name)
        if cls:
            sources.append(cls(cfg))
    return sources


# ── 主流程 ───────────────────────────────────────────────

def median_or_none(values):
    vals = [v for v in values if v]
    return statistics.median(vals) if vals else None


def parse_totals(totals: dict) -> dict or None:
    """把大小球结构解析为 {line, over, under}。
    兼容两种格式: {Over 2.5: 1.8, Under 2.5: 2.0} 或 {over: 1.94, under: 1.88}(隐含2.5盘)。"""
    if not totals:
        return None
    # 已归一化格式
    if "over" in totals and "under" in totals:
        return {"line": 2.5, "over": totals["over"], "under": totals["under"]}
    import re
    over_prices, under_prices, lines = {}, {}, set()
    for name, price in totals.items():
        m = re.search(r"([0-9.]+)", name)
        if not m:
            continue
        line = float(m.group(1))
        lines.add(line)
        if name.lower().startswith("over"):
            over_prices[line] = price
        elif name.lower().startswith("under"):
            under_prices[line] = price
    if not lines:
        return None
    line = 2.5 if 2.5 in lines else sorted(lines)[0]
    return {"line": line, "over": over_prices.get(line), "under": under_prices.get(line)}


def aggregate(matches_by_source):
    """多源聚合：按 (主队,客队) 合并，赔率取中位数。"""
    grouped = {}
    for source_matches in matches_by_source:
        for m in source_matches:
            key = (m["homeTeam"].lower(), m["awayTeam"].lower())
            grouped.setdefault(key, []).append(m)

    results = []
    for (home, away), ms in grouped.items():
        def h2h_val(k):
            vals = []
            for m in ms:
                h = m.get("markets", {}).get("h2h") or {}
                v = h.get(k) or h.get(k.capitalize())
                if v:
                    vals.append(v)
            return median_or_none(vals)

        totals_parsed = [parse_totals(m.get("markets", {}).get("totals")) for m in ms]
        totals_parsed = [t for t in totals_parsed if t]
        totals_out = None
        if totals_parsed:
            totals_out = {
                "line": totals_parsed[0]["line"],
                "over": median_or_none([t["over"] for t in totals_parsed]),
                "under": median_or_none([t["under"] for t in totals_parsed]),
            }

        results.append({
            "homeTeam": ms[0]["homeTeam"],
            "awayTeam": ms[0]["awayTeam"],
            "kickoff": ms[0].get("kickoff", ""),
            "matchUrl": ms[0].get("matchUrl", ""),
            "markets": {
                "h2h": {"home": h2h_val("home"), "draw": h2h_val("draw"), "away": h2h_val("away")},
                "totals": totals_out,
            },
            "sources": [m["source"] for m in ms],
            "bookmakers": sum(m.get("bookmakers", 0) for m in ms),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="博彩赔率抓取")
    parser.add_argument("--leagues", nargs="*",
                        default=["PL", "PD", "SA", "BL1", "FL1", "CL", "CH", "ED", "PPL", "TSL", "SPL", "BDF", "BSA", "MLS", "JLG", "KL1"],
                        help="要抓取的联赛代码")
    args = parser.parse_args()

    with open(CONFIG_DIR / "odds_sources.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sources = build_sources(config)
    if not sources:
        print("没有启用的赔率数据源")
        return 1

    print(f"数据源: {[type(s).__name__ for s in sources]}")

    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    all_odds = []
    for league in args.leagues:
        print(f"  {league}...")
        per_source = []
        for src in sources:
            per_source.append(src.fetch(league))
            time.sleep(0.5)  # 礼貌限速
        merged = aggregate(per_source)
        if merged:
            out = {
                "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "league": league,
                "matches": merged,
            }
            with open(ODDS_DIR / f"{league}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"    ✓ {len(merged)} 场赔率 -> data/odds/{league}.json")
            all_odds.extend(merged)
        else:
            print(f"    - 无数据")

    print(f"\n总计: {len(all_odds)} 场比赛赔率")
    return 0


if __name__ == "__main__":
    sys.exit(main())
