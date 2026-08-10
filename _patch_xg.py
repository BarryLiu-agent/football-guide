# -*- coding: utf-8 -*-
"""index.html: xG 渲染适配新数据结构 (data.teams / data.players)。"""
import re
from pathlib import Path

p = Path("index.html")
src = p.read_text(encoding="utf-8")
changed = []

# 1. renderXg 读取新结构
OLD1 = """function renderXg() {
  const xg = state.xg[state.league];
  if (!xg || !xg.teamsData) return `<div class="empty-state">"""
NEW1 = """function renderXg() {
  const xg = state.xg[state.league];
  const teams = xg?.data?.teams || xg?.teamsData || null;
  if (!teams || !teams.length) return `<div class="empty-state">"""
if OLD1 in src:
    src = src.replace(OLD1, NEW1)
    # 后续引用 xg.teamsData 的地方改为 teams
    src = src.replace("const teams = Object.values(xg.teamsData);", "const teams = Object.values(teams);")
    changed.append("renderXg 入口")
else:
    print("renderXg 入口未匹配")
    m = re.search(r"function renderXg\(\).{0,300}", src, re.S)
    print("实际:", m.group(0)[:250] if m else "无")

# 2. renderXg 字段：旧字段名(title/xG/xGA/pts/xpts) → 新(title/xG/xGA/pts/xPTS)
OLD2 = "      const pts = t.pts || 0, xPts = t.xpts || 0, diff = pts - xPts;"
NEW2 = "      const pts = t.pts || 0, xPts = t.xPTS || t.xpts || 0, diff = pts - xPts;"
if OLD2 in src:
    src = src.replace(OLD2, NEW2); changed.append("xPTS 字段")

# 3. renderScorers 读取新结构
OLD3 = """function renderScorers() {
  const xg = state.xg[state.league];
  const players = xg?.playersData;"""
NEW3 = """function renderScorers() {
  const xg = state.xg[state.league];
  const players = xg?.data?.players || xg?.playersData || null;"""
if OLD3 in src:
    src = src.replace(OLD3, NEW3); changed.append("renderScorers 入口")
else:
    print("renderScorers 入口未匹配")

p.write_text(src, encoding="utf-8")
print("应用:", changed if changed else "无匹配!")
