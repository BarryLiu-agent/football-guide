# -*- coding: utf-8 -*-
"""
build_status.py - 数据更新状态清单生成（data/status.json）
聚合所有 data/*.json 的 generatedAt，生成前端"数据状态抽屉"所需的状态清单。

原则:
  - 只报事实（文件存在性 + generatedAt + 可选配额信息），
    阈值判断/文案/颜色由前端 STATUS_MODULES 描述表负责（key 对齐）。
  - 任何单模块失败不阻塞整体：文件缺失/损坏 → ok=false, updatedAt=null。
  - 欧赔/亚盘/xG 等按联赛分文件，取"最旧"文件的 generatedAt（保守），
    或文件全缺时 updatedAt=null。

用法:
  python scripts/build_status.py            # 生成 data/status.json
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

# 单文件模块: key -> 文件路径
SINGLE_FILE = {
    "fixtures": "fixtures.json",
    "predictions": "predictions.json",
    "standings": "standings.json",
    "jingcai": "jingcai.json",
    "messages": "messages.json",
    "lineups": "lineups.json",
    "oddsQuota": "odds_quota.json",
    "xgMatches": "xg/matches.json",
    "calibration": "calibration.json",
    "calibrationOu": "calibration_ou.json",
}

# 多文件模块: key -> (glob 模式, 期望文件数)
MULTI_FILE = {
    "odds": ("odds/*.json", 6),
    "asian": ("asian/*.json", 5),
    "xgTeams": ("xg/*.json", 6),  # xg/PL.json 等联赛榜（含 CL stub，不含 matches.json）
}


def _read_generated(path: Path):
    """返回 (ok, generatedAt)。文件不存在/损坏 → (False, None)。"""
    if not path.exists():
        return False, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return True, data.get("generatedAt")
    except Exception:
        return False, None


def _is_empty_stub(path: Path) -> bool:
    """判断是否为空壳数据文件（如休赛期 CL.json 无比赛 / xg 抓取失败只剩空榜）。
    空壳不参与新鲜度计算。"""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        ms = d.get("matches")
        if isinstance(ms, list):
            return len(ms) == 0
        # asian/xg 榜单类：看 data.teams 或 top 级 total
        if d.get("total") == 0:
            return True
        # xg 榜单：teams 为空 或 players 全为空行（抓取失败遗留 stub）
        data = d.get("data")
        if isinstance(data, dict):
            teams = data.get("teams")
            players = data.get("players")
            if isinstance(teams, list) and isinstance(players, list):
                if len(teams) == 0:
                    return True
                if players and all(not (p or {}).get("player_name") for p in players):
                    return True
        return False
    except Exception:
        return False


def _collect_multi(pattern: str, expect: int):
    """多文件模块：返回 (ok, 最新非空 generatedAt, 存在文件数/期望数, 缺失文件清单)。
    空壳文件（如欧冠休赛期无比赛）不参与新鲜度计算；时间取非空文件中最新（真实最近抓取）。
    缺失 = 文件不存在 / 读取失败 / 空壳且同批其他文件有数据（如某联赛抓取失败只剩 stub）。"""
    files = sorted(DATA_DIR.glob(pattern))
    # 排除 xg/matches.json（它单独一个模块）
    files = [f for f in files if not (f.parent.name == "xg" and f.name == "matches.json")]
    gens = []
    ok_files = 0
    empty_stubs = []
    missing = []
    for f in files:
        ok, g = _read_generated(f)
        if not ok:
            missing.append(f.name)
            continue
        if _is_empty_stub(f):
            # 空壳文件（欧冠休赛期无比赛 / 该联赛抓取失败只剩 stub）：不参与新鲜度与 ok 计数
            empty_stubs.append(f.name)
            continue
        ok_files += 1
        if not g:
            missing.append(f.name)
            continue
        gens.append(g)
    if not gens:
        # 全部为空壳（如整个赛季未开）：报 ok 但无更新时间
        ok = ok_files >= max(1, expect - 1)
        return ok, None, f"{ok_files}/{expect}", (missing + empty_stubs)
    newest = max(gens)  # 非空文件中取最新（真实最近一次抓取）
    ok = ok_files >= max(1, expect - 1)  # 允许缺 1 个（如欧冠空文件仍算正常）
    return ok, newest, f"{ok_files}/{expect}", (missing + empty_stubs)


def build() -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    modules = []

    for key, rel in SINGLE_FILE.items():
        ok, gen = _read_generated(DATA_DIR / rel)
        modules.append({"key": key, "updatedAt": gen, "ok": ok})

    for key, (pattern, expect) in MULTI_FILE.items():
        ok, gen, count, missing = _collect_multi(pattern, expect)
        entry = {"key": key, "updatedAt": gen, "ok": ok, "files": count}
        if missing:
            entry["missing"] = sorted(set(missing))
        modules.append(entry)

    # 欧赔配额附加信息（供前端展示剩余量）
    try:
        q = json.loads((DATA_DIR / "odds_quota.json").read_text(encoding="utf-8"))
        quota_extra = {"totalRemaining": q.get("totalRemaining")}
    except Exception:
        quota_extra = {}

    payload = {
        "generatedAt": now,
        "modules": modules,
        "oddsQuota": quota_extra,
    }
    return payload


def main():
    payload = build()
    (DATA_DIR / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok_count = sum(1 for m in payload["modules"] if m["ok"])
    print(f"status.json: {ok_count}/{len(payload['modules'])} 模块正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
