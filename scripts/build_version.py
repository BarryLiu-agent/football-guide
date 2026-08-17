# -*- coding: utf-8 -*-
"""
build_version.py - 数据版本指纹生成（data/version.json）
对前端 loadAll() 加载的全部数据文件计算内容 MD5 汇总 → 版本号。
前端用 `?v=<version>` 做缓存破坏：数据变化 → 版本变化 → URL 变化 → 拿到新数据；
数据未变 → 版本相同 → URL 相同 → 命中浏览器/GitHub Pages CDN 缓存（关键性能优化）。

替代旧的 `?t=Date.now()` 全量击穿缓存策略：
  - version.json 本身极小（~90B），允许它每次冷拉（恒带时间戳）；
  - 其余 32 个数据文件只有在版本变化时才重新下载。

用法:
  python scripts/build_version.py
"""

import hashlib
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

# 与 index.html loadAll() 一致的数据文件清单
NAMED = [
    "fixtures.json", "predictions.json", "standings.json",
    "prediction_history.json", "calibration.json",
    "calibration_ou.json", "team_names.json",
]
GLOBS = ["odds/*.json", "xg/*.json", "advanced/*.json"]


def collect_files() -> list:
    files = [DATA_DIR / n for n in NAMED]
    for g in GLOBS:
        files.extend(sorted(DATA_DIR.glob(g)))
    # xg/*.json 仅含球队榜（单场 matches.json 已下线）
    seen, out = set(), []
    for f in files:
        rp = f.relative_to(DATA_DIR).as_posix()
        if rp not in seen and f.exists():
            seen.add(rp)
            out.append(f)
    return sorted(out, key=lambda f: f.relative_to(DATA_DIR).as_posix())


def main():
    files = collect_files()
    h = hashlib.md5()
    for f in files:
        h.update(f.read_bytes())
        h.update(b"\0")
    version = h.hexdigest()[:12]
    payload = {
        "version": version,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": len(files),
    }
    (DATA_DIR / "version.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8")
    print(f"version.json: {version} ({len(files)} 个数据文件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
