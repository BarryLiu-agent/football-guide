# 本地 Agent 操作指南

> 本文件面向**接管本地电脑工作的 AI Agent**，说明这台机器上的足球数据项目如何运行、如何交接、能做什么不能做什么。
> 配套阅读: `TECH_DOC.md`（系统架构全貌）

---

## 1. 这台电脑上的项目位置

| 项 | 路径 |
|---|---|
| 项目根目录 | `C:/Users/QZ/Desktop/FOOTBALL` |
| 前端 | `index.html`（单文件 SPA） |
| 抓取/预测脚本 | `scripts/*.py` |
| 配置 | `config/*.json` |
| 数据产物 | `data/*.json`（git 跟踪） |
| 本地密钥 | `.env`（**绝不要提交/发送**） |
| 计划任务 | `FootballLocalJC` / `FootballLocalXG` / `FootballLocalFBref` |

---

## 2. 本地跑什么（3 个计划任务）

| 任务名 | 频率 | 命令 | 产出 |
|---|---|---|---|
| FootballLocalJC | 每小时 | `python scripts/jingcai_fetch_local.py --push` | `data/jingcai.json`（竞彩 SP） |
| FootballLocalXG | 每天 9:00 | `python scripts/xg_fetch_local.py --push` | `data/xg/{联赛}.json`（球队 xG 榜） |
| FootballLocalFBref | 每天 10:30 | `python scripts/fbref_advanced.py` + git push | `data/advanced/{联赛}.json` |

**共同特征**：跑完自动 `git push` 到 GitHub，网页随之更新。

---

## 3. Agent 能做什么（授权范围）

### ✅ 可以做的日常操作
1. **手动触发抓取**（验证/补数据）
   ```bash
   cd C:/Users/QZ/Desktop/FOOTBALL
   python scripts/jingcai_fetch_local.py --push      # 竞彩
   python scripts/xg_fetch_local.py --push           # 球队xG榜
   python scripts/fbref_advanced.py                  # FBref高级数据
   ```
2. **本地跑预测/校准**（不消耗配额）
   ```bash
   python scripts/predict.py --skip-ai               # 纯统计预测
   python scripts/backtest.py                        # 模型校准回测
   python scripts/build_status.py && python scripts/build_version.py
   ```
3. **改代码/配置 → 提交推送**（走正常 git 流程）
   ```bash
   git add -A && git commit -m "..." && git pull --rebase origin main && git push origin main
   ```
4. **检查数据健康**：看 `data/status.json`、`data/odds_quota.json`（配额）、各文件 generatedAt
5. **操作计划任务**（创建/禁用/查看）
   ```bash
   schtasks /query /tn "任务名" /v /fo list
   schtasks /change /tn "任务名" /disable   # 或 /enable
   ```
6. **浏览器验证网页**：用户 Chrome 打开网站，或本地 `python -m http.server 8899` + 浏览器

### ❌ 不可以做的
1. **绝不提交 `.env`**（含 4 个 ODDS Key + AI Key）——已 gitignore，但 push 前仍要确认 `git status` 无 .env
2. **绝不把 Key/Token 写进任何代码或文档**
3. **绝不删除/清空 data/ 下数据文件**（前端依赖，删了网页白屏）
4. **不在本地大改 CI 工作流**（`.github/workflows/` 属于云端，改动需在 push 后手动触发验证）
5. **不擅自抓 Understat 单场 xG**（`--matches` 功能已停用，用户明确不需要）

---

## 4. 交接检查清单（新 agent 接手时先做）

```bash
# 1. 确认项目状态
cd C:/Users/QZ/Desktop/FOOTBALL
git status -sb                 # 应在 main 分支，与 origin/main 同步
git log --oneline -5           # 最近提交

# 2. 确认数据健康
python -c "import json; d=json.load(open('data/status.json',encoding='utf-8')); print([(m['key'],m['ok'],m.get('updatedAt')) for m in d['modules']])"
python -c "import json; print(json.load(open('data/odds_quota.json',encoding='utf-8'))['totalRemaining'])"

# 3. 确认计划任务在跑
schtasks /query /fo csv | findstr Football

# 4. 确认 Python 环境可用
python --version && pip list 2>/dev/null | grep -E "requests|beautifulsoup4|playwright|soccerdata"

# 5. 确认网页在线（可选）
#   用浏览器打开 https://barryliu-agent.github.io/football-guide/
```

---

## 5. 常见用户请求 → 对应操作

| 用户说 | Agent 做 |
|---|---|
| "竞彩怎么没更新" | 查 `data/jingcai.json` generatedAt → 手动跑 `jingcai_fetch_local.py --push` → 查 FootballLocalJC 任务 |
| "赔率不对/太旧" | 查 `odds_quota.json` 配额 → 配额够就手动 `odds_fetcher.py`（云端职责，本地也可跑，消耗配额） |
| "网页数据旧" | 看 🕐 状态抽屉哪个模块红 → 对应脚本手动跑 + push |
| "改前端显示" | 改 `index.html`（注意 JS 语法检查：反引号/模板字符串闭合）→ 提交推送 |
| "改预测模型" | 改 `scripts/predict.py` / `config/prediction_rules.json` → 本地 `--skip-ai` 验证 → 推送 |
| "配额快没了" | 查 `odds_quota.json`，提醒用户 9/1 重置，或建议加 Key |
| "网站打不开" | 先本地 `curl -I https://barryliu-agent.github.io/football-guide/` 或浏览器试，区分网络问题 vs 部署问题 |
| "技术细节" | 读 `TECH_DOC.md` |

---

## 6. 安全与边界红线

1. **.env 保密**：内容含 `ODDS_API_KEY_1~4`、`AI_API_KEY`。任何输出/文档/commit 不得出现
2. **GitHub Token**：用户浏览器 localStorage 存有 `gh_pat`（云端刷新用），agent 不要读取/导出
3. **只动 FOOTBALL 项目**：不要碰桌面其他文件夹（有用户工作文件）
4. **git 操作规范**：push 前先 `git pull --rebase`；遇冲突优先保留"更新的数据"，代码冲突问用户
5. **不要删除计划任务**：三个任务是基础设施；如需禁用先问用户
6. **网络不稳定**：本机连 GitHub 常间歇性失败，push 失败重试 2-3 次，间隔 30-60 秒

---

## 7. 快速验证方法

```bash
# 语法检查（改动后必做）
python -m py_compile scripts/*.py

# 前端 JS 语法（改动 index.html 后必做）
python - <<'PY'
import re, subprocess, tempfile, os
html = open('index.html', encoding='utf-8').read()
js = re.search(r'<script>(.*?)</script>', html, re.S).group(1)
open('_check.js','w',encoding='utf-8').write(js)
r = subprocess.run(['node','--check','_check.js'])
os.remove('_check.js')
print('JS OK' if r.returncode==0 else 'JS FAIL')
PY

# 预测端到端
python scripts/predict.py --skip-ai 2>&1 | tail -3

# 检查数据一致性（预测比分 vs 概率方向）
python -c "
import json
d=json.load(open('data/predictions.json',encoding='utf-8'))
def out(s):
    h,a=map(int,s.split('-')); return 'H' if h>a else ('D' if h==a else 'A')
mis=sum(1 for p in d['predictions'] if out(p['predictedScore'])!= {'home':'H','draw':'D','away':'A'}[max(p['probabilities'],key=lambda k:p['probabilities'][k])])
print(f'{mis} 场不一致 (应为0)')
"
```

---

## 8. 交接给下一个 agent 时

```text
项目: C:/Users/QZ/Desktop/FOOTBALL（足球数据面板）
先读: TECH_DOC.md（架构）+ LOCAL_AGENT_GUIDE.md（本文件）
当前状态: 竞彩/球队xG/FBref 三个本地任务正常；单场xG已停用
数据源: 云端 Actions 全自动 + 本地 3 任务
密钥: .env（勿提交）
验证: git status 干净 + status.json 全 ok + 网页可访问
```
