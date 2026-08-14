# -*- coding: utf-8 -*-
"""
lineup_fetcher.py - 赛前首发/伤停抓取（多源自动回退）
抓取即将开赛比赛的确认首发名单与伤停信息，供预测引擎与 AI 研判使用。

数据源（按顺序回退，成功即停）:
  1. FotMob    https://www.fotmob.com/api/...         (首发+伤停新闻, 无 key)
  2. ESPN      https://site.api.espn.com/...          (rosters+injuries, 无 key)
  3. Sofascore https://api.sofascore.com/api/v1/...   (lineups, 无 key)

抓取时机设计（关键）:
  - 五大联赛首发名单通常于开赛前 60~75 分钟公布
  - 本脚本只抓"未来 3 小时内开赛"的比赛，每小时由 GitHub Actions 触发一次
    (lineups.yml, cron 每小时 15 分)
  - 同一场比赛每小时最多抓一次，首发公布后 1 小时内必被抓到
  - 已开赛/无首发的比赛跳过，绝不阻塞主流程

用法:
  python scripts/lineup_fetcher.py [--hours 3] [--dry-run]

输出:
  data/lineups.json
  {generatedAt, matches: [{homeTeam, awayTeam, kickoff, homeLineup[], awayLineup[],
                           injuries: {home:[], away:[]}, source}]}
"""

import argparse
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# 各源联赛标识（五大联赛+欧冠）
FOTMOB_LEAGUES = {47, 87, 55, 54, 53, 42}   # PL/PD/SA/BL1/FL1/CL
ESPN_LEAGUES = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions"]
SOFASCORE_LEAGUES = {17, 8, 23, 35, 34, 7}  # PL/LaLiga/SerieA/BL/FL1/UCL


def _get(url, timeout=15):
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "application/json",
                                   "Accept-Language": "en-US,en;q=0.9"}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()


def _norm(name: str) -> str:
    """归一化队名：小写 + 去常见俱乐部后缀（FC/AFC/CF/SC/UD/CD，含不带空格与带点变体）。"""
    import re
    t = (name or "").lower().strip()
    t = re.sub(r"\b(fc|afc|cf|sc|ud|cd|ac|sv|fk)\b", "", t)
    t = re.sub(r"[\-\.']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── 源 1: FotMob ──────────────────────────────────────────
def _fotmob_parse_injuries(team_block: dict) -> list:
    """从 FotMob lineup 球员块解析伤停：球员对象 status != normal 视为异常。
    返回 [{name, status, reason}]。"""
    inj = []
    for row in (team_block.get("players") or []):
        items = row if isinstance(row, list) else [row]
        for p in items:
            if not isinstance(p, dict):
                continue
            name = (p.get("name") or {}).get("fullName") or p.get("name")
            status = (p.get("status") or "normal").lower()
            if name and status and status not in ("normal", "starter", "substitute", ""):
                inj.append({
                    "name": name,
                    "status": "out" if status in ("injured", "suspended", "absent") else "doubtful",
                    "reason": status,
                })
    return inj


def fetch_fotmob(target: dict) -> dict or None:
    """target = {homeTeam, awayTeam, kickoff}。按日期+队名匹配 matchId 后取首发。"""
    date = datetime.fromisoformat(target["kickoff"].replace("Z", "+00:00"))
    day = date.strftime("%Y%m%d")
    data = _get(f"https://www.fotmob.com/api/matches?date={day}")
    leagues = data.get("leagues", [])
    want = {_norm(target["homeTeam"]), _norm(target["awayTeam"])}
    match_id = None
    for lg in leagues:
        if lg.get("id") not in FOTMOB_LEAGUES and lg.get("primaryId") not in FOTMOB_LEAGUES:
            continue
        for m in lg.get("matches", []):
            names = {_norm((m.get("home") or {}).get("name")),
                     _norm((m.get("away") or {}).get("name"))}
            if names == want and m.get("id"):
                match_id = m["id"]
                break
        if match_id:
            break
    if not match_id:
        return None
    detail = _get(f"https://www.fotmob.com/api/matchDetails?matchId={match_id}")
    content = detail.get("content", {})
    lu = (content.get("lineup") or {}).get("lineup") or []
    out = {"homeLineup": [], "awayLineup": [], "injuries": {"home": [], "away": []},
           "confirmed": True}
    any_confirmed = False
    for team_block in lu:
        tname = _norm((team_block.get("team") or {}).get("name") or "")
        players = []
        for row in (team_block.get("players") or []):
            if isinstance(row, dict):
                players.append((row.get("name") or {}).get("fullName") or row.get("name"))
            elif isinstance(row, list):
                for p in row:
                    if isinstance(p, dict):
                        players.append((p.get("name") or {}).get("fullName") or p.get("name"))
        players = [p for p in players if p]
        # 伤停：同块解析（异常状态球员）
        inj = _fotmob_parse_injuries(team_block)
        if tname == _norm(target["homeTeam"]):
            out["homeLineup"] = players
            out["injuries"]["home"] = inj
        elif tname == _norm(target["awayTeam"]):
            out["awayLineup"] = players
            out["injuries"]["away"] = inj
        if team_block.get("confirmed") is True:
            any_confirmed = True
    if not out["homeLineup"] and not out["awayLineup"]:
        return None
    # FotMob lineup 无 confirmed 标记时：以球员数量接近 11 为"已确认"
    out["confirmed"] = any_confirmed or max(len(out["homeLineup"]), len(out["awayLineup"])) >= 11
    out["source"] = "fotmob"
    return out


# ── 源 2: ESPN ────────────────────────────────────────────
def _espn_parse_injuries(s: dict) -> dict:
    """ESPN summary 顶层 injuries 数组 → {homeTeam_norm: [inj], awayTeam_norm: [inj]}。
    结构: [{team:{displayName}, athlete:{displayName}, type:{description}, status}]。"""
    out = {}
    for it in s.get("injuries", []) or []:
        team = ((it.get("team") or {}).get("displayName") or "")
        name = ((it.get("athlete") or {}).get("displayName") or "")
        reason = ((it.get("type") or {}).get("description") or "") or (it.get("status") or "")
        st = str(it.get("status") or "").lower()
        status = "out" if st in ("out", "ruled out") else ("doubtful" if st in ("questionable", "day-to-day", "doubtful") else "out")
        if team and name:
            out.setdefault(_norm(team), []).append({
                "name": name, "status": status, "reason": reason or "injury",
            })
    return out


def fetch_espn(target: dict) -> dict or None:
    date = datetime.fromisoformat(target["kickoff"].replace("Z", "+00:00"))
    day = date.strftime("%Y%m%d")
    want = {_norm(target["homeTeam"]), _norm(target["awayTeam"])}
    for league in ESPN_LEAGUES:
        try:
            data = _get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={day}")
        except Exception:
            continue
        event_id = None
        for e in data.get("events", []):
            names = set()
            for comp in e.get("competitions", []):
                for c in comp.get("competitors", []):
                    names.add(_norm((c.get("team") or {}).get("displayName")))
            if names == want:
                event_id = e.get("id")
                break
        if not event_id:
            continue
        try:
            s = _get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary?event={event_id}")
        except Exception:
            return None
        out = {"homeLineup": [], "awayLineup": [], "injuries": {"home": [], "away": []},
               "confirmed": True}
        inj_map = _espn_parse_injuries(s)
        for roster in s.get("rosters", []):
            tname = _norm((roster.get("team") or {}).get("displayName") or "")
            starters = [((r.get("athlete") or {}).get("displayName") or "")
                        for r in roster.get("roster", []) if r.get("starter")]
            starters = [p for p in starters if p]
            if tname == _norm(target["homeTeam"]):
                out["homeLineup"] = starters
                out["injuries"]["home"] = inj_map.get(tname, [])
            elif tname == _norm(target["awayTeam"]):
                out["awayLineup"] = starters
                out["injuries"]["away"] = inj_map.get(tname, [])
        if not out["homeLineup"] and not out["awayLineup"]:
            continue
        out["source"] = "espn"
        return out
    return None


# ── 源 3: Sofascore ───────────────────────────────────────
def fetch_sofascore(target: dict) -> dict or None:
    date = datetime.fromisoformat(target["kickoff"].replace("Z", "+00:00"))
    day = date.strftime("%Y-%m-%d")
    want = {_norm(target["homeTeam"]), _norm(target["awayTeam"])}
    data = _get(f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{day}")
    event_id = None
    for e in data.get("events", []):
        t = e.get("tournament") or {}
        uid = t.get("uniqueTournament", {}).get("id")
        if uid not in SOFASCORE_LEAGUES:
            continue
        names = {_norm((e.get("homeTeam") or {}).get("name")),
                 _norm((e.get("awayTeam") or {}).get("name"))}
        if names == want:
            event_id = e.get("id")
            break
    if not event_id:
        return None
    lu = _get(f"https://api.sofascore.com/api/v1/event/{event_id}/lineups")
    out = {"homeLineup": [], "awayLineup": [], "injuries": {"home": [], "away": []},
           "confirmed": False}

    def starters(block):
        res = []
        for p in block.get("players", []):
            if p.get("starter"):
                pl = p.get("player") or {}
                res.append(pl.get("name") or "")
        return [x for x in res if x]

    def injuries(block):
        """Sofascore lineup 中 confirmed=false 的球员视为存疑（非首发未确认 ≠ 伤停 out）。
        仅当球员明确不在首发且确认缺席时才标出，避免把替补当伤停。"""
        res = []
        for p in block.get("players", []):
            if not p.get("starter") and p.get("confirmed") is False:
                pl = p.get("player") or {}
                nm = pl.get("name") or ""
                if nm:
                    res.append({"name": nm, "status": "doubtful", "reason": "not confirmed in lineup"})
        return res

    if lu.get("home") and lu["home"].get("confirmed"):
        out["homeLineup"] = starters(lu["home"])
        out["confirmed"] = True
    if lu.get("away") and lu["away"].get("confirmed"):
        out["awayLineup"] = starters(lu["away"])
        out["confirmed"] = True
    # 伤停（无论首发是否确认，只要 lineups 接口有数据就解析）
    if lu.get("home"):
        out["injuries"]["home"] = injuries(lu["home"])
    if lu.get("away"):
        out["injuries"]["away"] = injuries(lu["away"])
    if not out["homeLineup"] and not out["awayLineup"]:
        return None
    out["source"] = "sofascore"
    return out


FETCHERS = [fetch_fotmob, fetch_espn, fetch_sofascore]


def load_upcoming(hours: int) -> list:
    """从 fixtures.json 取未来 hours 小时内开赛且未开赛的比赛。"""
    path = DATA_DIR / "fixtures.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        fx = json.load(f)
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)
    out = []
    for m in fx.get("matches", []):
        if m.get("status") in ("FINISHED", "IN_PLAY", "PAUSED", "POSTPONED", "CANCELLED", "SUSPENDED"):
            continue
        kick = m.get("utcDate") or ""
        if not kick:
            continue
        try:
            kt = datetime.fromisoformat(kick.replace("Z", "+00:00"))
        except ValueError:
            continue
        if kt <= now or kt > cutoff:
            continue
        out.append({
            "homeTeam": (m.get("homeTeam") or {}).get("name") or (m.get("homeTeam") or {}).get("shortName") or "",
            "awayTeam": (m.get("awayTeam") or {}).get("name") or (m.get("awayTeam") or {}).get("shortName") or "",
            "kickoff": kick,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="赛前首发/伤停抓取")
    parser.add_argument("--hours", type=int, default=3, help="只抓未来 N 小时内开赛的比赛")
    parser.add_argument("--dry-run", action="store_true", help="只打印将抓取的比赛，不发请求")
    args = parser.parse_args()

    upcoming = load_upcoming(args.hours)
    print(f"未来 {args.hours} 小时内开赛的比赛: {len(upcoming)} 场")
    if args.dry_run:
        for m in upcoming:
            print(f"  - {m['homeTeam']} vs {m['awayTeam']} ({m['kickoff']})")
        return 0

    results = []
    for m in upcoming:
        got = None
        for fn in FETCHERS:
            try:
                got = fn(m)
            except Exception as e:
                print(f"  {fn.__name__} 失败({m['homeTeam']}): {e}")
                continue
            if got:
                break
        if got:
            got.update(m)
            got["fetchedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            results.append(got)
            print(f"  ✓ {m['homeTeam']} vs {m['awayTeam']}: {got['source']} 主{len(got['homeLineup'])}人/客{len(got['awayLineup'])}人")
        else:
            print(f"  - {m['homeTeam']} vs {m['awayTeam']}: 首发未公布或源不可用")
            results.append({**m, "homeLineup": [], "awayLineup": [],
                            "injuries": {"home": [], "away": []},
                            "source": None, "fetchedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})

    out = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(results),
        "matches": results,
    }
    (DATA_DIR / "lineups.json").write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"输出: data/lineups.json ({len(results)} 场)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
