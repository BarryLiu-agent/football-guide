"""
亚盘数据抓取：football-data.co.uk（免费，无需 key）

数据源: https://www.football-data.co.uk/mmz4281/{season}/{code}.csv
  - AHh: 亚盘让球数（主队视角，负=让球，正=受让）
  - MaxAHH/MaxAHA: 最高亚盘水位（主/客）
  - AvgAHH/AvgAHA: 平均亚盘水位（主/客）
  - 更新周期: 每周一更新上周数据（收盘盘口）

用法:
  python scripts/asian_fetch.py [season]   # season 默认 2526 (2025/26)
输出:
  data/asian/{PL,PD,BL1,SA,FL1}.json
  每场: {homeTeam, awayTeam, kickoff, line, homeOdds, awayOdds, homeGoals, awayGoals, result, settle}
"""

import csv
import io
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ASIAN_DIR = DATA_DIR / "asian"

# football-data.co.uk 联赛代码 → 本站联赛代码
LEAGUE_MAP = {
    "E0": "PL",   # 英超
    "SP1": "PD",  # 西甲
    "D1": "BL1",  # 德甲
    "I1": "SA",   # 意甲
    "F1": "FL1",  # 法甲
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_csv(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return raw.decode("utf-8-sig", errors="replace")


def parse_row(row: dict) -> dict | None:
    """解析一行 CSV → 亚盘记录；无亚盘/无赛果返回 None。"""
    line_raw = (row.get("AHh") or "").strip()
    if not line_raw:
        return None
    try:
        line = float(line_raw)
    except ValueError:
        return None
    # 水位：优先平均，缺则取最高
    home_odds = _num(row.get("AvgAHH")) or _num(row.get("MaxAHH"))
    away_odds = _num(row.get("AvgAHA")) or _num(row.get("MaxAHA"))
    if not home_odds or not away_odds:
        return None
    hg, ag = _num(row.get("FTHG")), _num(row.get("FTAG"))
    # 日期
    kickoff = ""
    try:
        dt = datetime.strptime(row["Date"], "%d/%m/%Y")
        kickoff = dt.strftime("%Y-%m-%d")
    except (KeyError, ValueError):
        pass
    # 结算：主队视角让球（line 负=主让）。让球后主队净胜 = 实际净胜 + line
    settle = ""
    if hg is not None and ag is not None:
        diff = (hg - ag) + line
        if abs(diff) < 1e-9:
            settle = "push"        # 走水
        elif diff > 0.25 + 1e-9:
            settle = "home"        # 主队赢盘
        elif diff < -0.25 - 1e-9:
            settle = "away"        # 客队赢盘
        elif diff > 0:
            settle = "home_half"   # 主队赢半（1/4 盘口）
        else:
            settle = "away_half"   # 客队赢半
    return {
        "homeTeam": row["HomeTeam"].strip(),
        "awayTeam": row["AwayTeam"].strip(),
        "kickoff": kickoff,
        "line": line,
        "homeOdds": home_odds,
        "awayOdds": away_odds,
        "homeGoals": hg,
        "awayGoals": ag,
        "result": f"{hg}-{ag}" if hg is not None else "",
        "settle": settle,
    }


def _num(v) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2526"
    ASIAN_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for fd_code, league in LEAGUE_MAP.items():
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{fd_code}.csv"
        try:
            text = fetch_csv(url)
        except Exception as e:
            print(f"  [{league}] 抓取失败: {e}")
            continue
        rows = list(csv.DictReader(io.StringIO(text)))
        records = []
        for r in rows:
            parsed = parse_row(r)
            if parsed:
                records.append(parsed)
        out = {
            "season": season,
            "league": league,
            "source": "football-data.co.uk",
            "generatedAt": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "matches": records,
        }
        path = ASIAN_DIR / f"{league}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        n_ah = sum(1 for m in records if m.get("line"))
        n_settled = sum(1 for m in records if m.get("settle"))
        print(f"  [{league}] {len(records)} 场（含亚盘 {n_ah}，已结算 {n_settled}）-> data/asian/{league}.json")
        total += len(records)
    print(f"总计 {total} 场亚盘记录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
