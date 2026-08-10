"""
odds_fetcher.py - 博彩赔率抓取
从可配置数据源抓取五大联赛+欧冠赔率，归一化后生成 data/odds/*.json。

用法:
  python scripts/odds_fetcher.py

可扩展接口:
  OddsSource (抽象基类) - 实现 fetch(league_code) -> list[dict]
    已有实现: TheOddsApiSource
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

# 加载 .env（不覆盖已存在的环境变量），避免每次会话手动 export
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() and _v.strip() and not os.environ.get(_k.strip()):
                os.environ[_k.strip()] = _v.strip()


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


# ── The Odds API 实现（多 Key 轮换）─────────────────────

class TheOddsApiSource(OddsSource):
    """The Odds API (https://the-odds-api.com)
    支持多 API Key 轮换：ODDS_API_KEY(主) → ODDS_API_KEY_2(备) → ...
    配额保护：剩余 < MIN_REMAINING 的 Key 自动跳过；401/403 时自动切换下一个。
    """

    MIN_REMAINING = 1  # 剩余≥1 即可尝试（401 不扣配额，失败自动切换）
    _key_quota = {}  # key -> remaining（本次运行内缓存）

    LEAGUE_MAP = {
        "PL": "soccer_epl",
        "PD": "soccer_spain_la_liga",
        "SA": "soccer_italy_serie_a",
        "BL1": "soccer_germany_bundesliga",
        "FL1": "soccer_france_ligue_one",
        # 欧冠仅正赛（资格赛不抓；正赛小组赛8月底开赛）
        "CL": "soccer_uefa_champs_league",
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

    def _get_keys(self):
        """按优先级收集所有可用 Key（去空/去重）。"""
        keys = []
        for env_name in ["ODDS_API_KEY", "ODDS_API_KEY_2", "ODDS_API_KEY_3"]:
            k = os.environ.get(env_name, "").strip()
            if k and k not in keys:
                keys.append(k)
        return keys

    def fetch(self, league_code: str) -> list:
        sport = self.LEAGUE_MAP.get(league_code)
        if not sport:
            return []
        keys = self._get_keys()
        if not keys:
            print(f"    - {league_code}: 未配置 ODDS_API_KEY")
            return []

        params_base = {
            "regions": ",".join(self.config.get("regions", ["eu"])),
            "markets": ",".join(self.config.get("markets", ["h2h"])),
        }
        url = f"{self.config['baseUrl']}/sports/{sport}/odds"

        for key in keys:
            # 配额保护：本次运行内剩余不足则跳过
            if self._key_quota.get(key, 999) < self.MIN_REMAINING:
                continue
            params = dict(params_base, apiKey=key)
            try:
                r = requests.get(url, params=params, timeout=20)
                # 记录剩余配额
                try:
                    self._key_quota[key] = int(r.headers.get("x-requests-remaining", 999))
                except (TypeError, ValueError):
                    pass
                if r.status_code == 200:
                    tag = key[-4:]
                    print(f"      (key ...{tag}, 剩 {self._key_quota.get(key, '?')})")
                    return [self.normalize(x) for x in r.json()]
                if r.status_code in (401, 403):
                    # Key 无效/配额耗尽：跳过换下一个
                    self._key_quota[key] = 0
                    continue
                if r.status_code == 429:
                    self._key_quota[key] = 0
                    continue
                print(f"    - {league_code}: HTTP {r.status_code} (key ...{key[-4:]})")
                return []
            except Exception as e:
                print(f"    - {league_code}: 请求失败 {e} (key ...{key[-4:]})")
                return []
        print(f"    - {league_code}: 所有 Key 配额不足或不可用")
        return []

    def normalize(self, raw: dict) -> dict:
        home_name = raw.get("home_team", "")
        away_name = raw.get("away_team", "")
        h2h = {"home": [], "draw": [], "away": []}
        totals = {"over": [], "under": []}
        spreads = {"home": [], "away": []}
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
                elif key == "spreads":
                    for o in market["outcomes"]:
                        point = o.get("point")
                        if o["name"] == home_name:
                            spreads["home"].append({"price": o["price"], "point": point})
                        elif o["name"] == away_name:
                            spreads["away"].append({"price": o["price"], "point": point})
        med = lambda xs: statistics.median(xs) if xs else None
        # 让球盘: 取最常见点差(mode)，赔率取中位数
        def spread_agg(items):
            if not items:
                return None
            from collections import Counter
            points = [i["point"] for i in items if i.get("point") is not None]
            if not points:
                return None
            point = Counter(points).most_common(1)[0][0]
            prices = [i["price"] for i in items if i.get("point") == point]
            return {"point": point, "price": med(prices)}
        return {
            "league": raw.get("sport_key", ""),
            "homeTeam": raw.get("home_team", ""),
            "awayTeam": raw.get("away_team", ""),
            "kickoff": raw.get("commence_time", ""),
            "matchUrl": "",
            "markets": {
                "h2h": {"home": med(h2h["home"]), "draw": med(h2h["draw"]), "away": med(h2h["away"])},
                "totals": {"over": med(totals["over"]), "under": med(totals["under"])},
                "spreads": {"home": spread_agg(spreads["home"]), "away": spread_agg(spreads["away"])},
            },
            "bookmakers": len(raw.get("bookmakers", [])),
            "source": "theoddsapi",
        }


# ── 源工厂（配置驱动）────────────────────────────────────

SOURCE_REGISTRY = {
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

        # 让球聚合：取最常见盘口点差，赔率中位数
        def spread_val(side):
            items = []
            for m in ms:
                s = (m.get("markets", {}).get("spreads") or {}).get(side)
                if s and s.get("point") is not None:
                    items.append(s)
            if not items:
                return None
            from collections import Counter
            point = Counter(i["point"] for i in items).most_common(1)[0][0]
            prices = [i["price"] for i in items if i["point"] == point]
            return {"point": point, "price": median_or_none(prices)}

        results.append({
            "homeTeam": ms[0]["homeTeam"],
            "awayTeam": ms[0]["awayTeam"],
            "kickoff": ms[0].get("kickoff", ""),
            "matchUrl": ms[0].get("matchUrl", ""),
            "markets": {
                "h2h": {"home": h2h_val("home"), "draw": h2h_val("draw"), "away": h2h_val("away")},
                "totals": totals_out,
                "spreads": {"home": spread_val("home"), "away": spread_val("away")},
            },
            "sources": [m["source"] for m in ms],
            "bookmakers": sum(m.get("bookmakers", 0) for m in ms),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="博彩赔率抓取")
    parser.add_argument("--leagues", nargs="*",
                        default=["PL", "PD", "SA", "BL1", "FL1", "CL"],
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
    fetch_failed = False
    for league in args.leagues:
        print(f"  {league}...")
        per_source = []
        for src in sources:
            per_source.append(src.fetch(league))
            time.sleep(0.5)  # 礼貌限速
        merged = aggregate(per_source)
        if merged:
            # 赔率快照：读取旧文件作为"上次快照"（初盘→即时盘对比）+ 累积历史曲线
            prev = {}
            history = {}
            old_path = ODDS_DIR / f"{league}.json"
            if old_path.exists():
                try:
                    with open(old_path, encoding="utf-8") as f:
                        old_data = json.load(f)
                    for om in old_data.get("matches", []):
                        k = (om["homeTeam"].lower(), om["awayTeam"].lower())
                        prev[k] = om.get("markets", {}).get("h2h")
                    history = old_data.get("history", {}) or {}
                except Exception:
                    pass
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for m in merged:
                k = (m["homeTeam"].lower(), m["awayTeam"].lower())
                if k in prev:
                    m["prevH2h"] = prev[k]
                # 累积走势快照（初盘=第一条，临场=最后一条；保留最近 30 个）
                hk = f"{k[0]}|{k[1]}"
                snap = {"ts": ts, "h2h": m.get("markets", {}).get("h2h"),
                        "totals": m.get("markets", {}).get("totals"),
                        "spreads": m.get("markets", {}).get("spreads")}
                hist = history.get(hk, [])
                # 同一时间戳去重（重复抓取不叠加）
                if not hist or hist[-1].get("ts") != ts:
                    hist.append(snap)
                history[hk] = hist[-30:]
            out = {
                "generatedAt": ts,
                "league": league,
                "history": history,
                "matches": merged,
            }
            with open(ODDS_DIR / f"{league}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"    ✓ {len(merged)} 场赔率 -> data/odds/{league}.json")
            all_odds.extend(merged)
        else:
            # 抓取失败/无数据时保留旧文件，避免清空已有赔率
            old = ODDS_DIR / f"{league}.json"
            if old.exists():
                with open(old, encoding="utf-8") as f:
                    old_data = json.load(f)
                n_old = len(old_data.get("matches", []))
                if n_old:
                    print(f"    - 本次无新数据, 保留旧数据 {n_old} 场")
                    all_odds.extend(old_data["matches"])
                    continue
            print(f"    - 无数据")
            fetch_failed = True

    if fetch_failed:
        print("⚠ 部分联赛抓取失败(可能配额耗尽或限流), 已保留旧数据")
    print(f"\n总计: {len(all_odds)} 场比赛赔率")
    return 0


if __name__ == "__main__":
    sys.exit(main())
