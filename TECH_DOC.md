# 足球数据面板 - 技术说明文档

> 五大联赛 + 欧冠的实时比分、赔率、模型预测与 AI 研判个人数据面板
> 部署地址: https://barryliu-agent.github.io/football-guide/
> 更新时间: 2026-08-16

---

## 1. 系统架构总览

```
┌─────────────────────────── 云端（GitHub Actions，全自动） ───────────────────────────┐
│  daily.yml   每天 10:00(北京时间)  完整抓取+预测+部署                                  │
│  results.yml 每 30 分钟           赛果刷新+预测+部署                                  │
│  lineups.yml 每小时               赛前首发抓取                                         │
└──────────────┬────────────────────────────────────────────────────────────────────┘
               │ git commit / push (数据写回 main)
┌──────────────▼────────────────────────────────────────────────────────────────────┐
│  数据层 data/*.json（GitHub 仓库）→ GitHub Pages 静态托管                            │
└──────────────┬────────────────────────────────────────────────────────────────────┘
               │ fetch (带 ?v= 版本指纹缓存)
┌──────────────▼────────────────────────────────────────────────────────────────────┐
│  前端 index.html（单文件 SPA，无框架，中文移动端优先）                                 │
│  赛程/赔率/预测/竞彩/亚盘/xG/战绩/AI研判 全部功能                                    │
└───────────────────────────────────────────────────────────────────────────────────┘
               ▲
┌──────────────┴──────────────────────── 本地电脑（计划任务，需开机） ────────────────┐
│  FootballLocalJC    每小时     竞彩 SP 抓取（接口仅限国内 IP）→ push                 │
│  FootballLocalXG    每天 9:00  球队赛季 xG 榜 → push                                │
│  FootballLocalFBref 每天 10:30 FBref 高级数据 → push                                │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**核心设计思想**：
- 纯静态托管（GitHub Pages），零后端、零数据库、零成本
- 数据由 CI 与本地脚本抓取后**写回 git 仓库**，前端只做静态读取
- 所有抓取失败都有降级：保留旧数据、单模块失败不阻塞整体

---

## 2. 数据源清单

| 数据源 | 内容 | 频率 | 运行位置 | Key |
|---|---|---|---|---|
| football-data.org | 赛程/赛果/积分榜（五大联赛+欧冠） | 每 30 分钟 | GitHub Actions | FOOTBALL_API_KEY |
| The Odds API | 欧赔（h2h/大小球/让球，21 家博彩中位数） | 每天 10:00 | GitHub Actions | ODDS_API_KEY×4 |
| football-data.co.uk | 亚盘历史收盘盘口 | 每周 | GitHub Actions | 免费 |
| ESPN + Sky RSS | 新闻/伤病/转会消息 | 每天 | GitHub Actions | 免费 |
| FotMob→ESPN→Sofascore | 赛前首发名单+伤停 | 赛前每小时 | GitHub Actions | 免费 |
| DeepSeek | AI 研判（吃模型输出+首发生成建议） | 每 4 小时 | GitHub Actions | AI_API_KEY |
| Understat | 球队赛季 xG 榜 + 射手榜 | 每天 | GitHub Actions | 免费(Playwright) |
| FBref | 高级数据（射门/门将/纪律） | 每天 | GitHub Actions | 免费 |

**配额说明**：The Odds API 每 Key 500 次/月，4 个 Key 轮换；一次请求=1 次配额（与 markets 数量无关）；每月 1 日重置。

---

## 3. 目录结构

```
FOOTBALL/
├── index.html                 # 前端单文件 SPA（CSS+JS 全内联）
├── scripts/                   # 所有抓取/预测脚本
│   ├── fetch_data.py          # football-data.org 赛程/积分榜
│   ├── odds_fetcher.py        # The Odds API 多Key轮换赔率
│   ├── messages_fetcher.py    # ESPN/Sky RSS 新闻消息
│   ├── lineup_fetcher.py      # 赛前首发多源回退
│   ├── jingcai_fetch_local.py # 竞彩 SP（本地）
│   ├── xg_fetch_local.py      # Understat xG（本地）
│   ├── fbref_advanced.py      # FBref 高级数据（本地）
│   ├── fbref_seasons.py       # FBref 历史赛季赛果+xG（season_2025.json）
│   ├── predict.py             # 预测引擎主流程
│   ├── ai_predictor.py        # DeepSeek AI 研判
│   ├── elo.py                 # Elo 模型 + Dixon-Coles + XgModel
│   ├── backtest.py            # 模型校准回测
│   ├── backtest_ou.py         # 盘口(大小球/让球)校准
│   ├── evolve.py              # 模型进化(自适应折扣)
│   ├── build_status.py        # 生成 status.json（数据状态清单）
│   ├── build_version.py       # 生成 version.json（缓存指纹）
│   └── gen_team_names.py      # 生成 team_names.json（中英队名映射）
├── config/                    # 配置
│   ├── odds_sources.json      # 赔率源配置（markets/regions）
│   ├── prediction_rules.json  # 预测规则（融合权重/关键词权重/阈值）
│   ├── big_teams.json         # 豪门名单
│   ├── star_players.json      # 球星名单
│   ├── rivalry_pairs.json     # 德比宿敌对
│   └── news_sources.json      # RSS 源
├── data/                      # 数据产物（git 跟踪，随部署上线）
│   ├── fixtures.json          # 赛程/赛果
│   ├── predictions.json       # 预测输出（58 场/每 30 分钟）
│   ├── odds/{PL,PD,SA,BL1,FL1,CL}.json  # 欧赔+history走势
│   ├── jingcai.json           # 竞彩 SP
│   ├── xg/{PL..FL1}.json      # xG 球队榜
│   ├── advanced/{PL..FL1}.json # FBref 高级数据
│   ├── season_2025.json       # 上赛季 485 场赛果+xG（模型训练）
│   ├── standings.json         # 积分榜
│   ├── lineups.json           # 赛前首发
│   ├── messages.json          # 新闻消息
│   ├── calibration.json       # 模型校准报告
│   ├── calibration_ou.json    # 盘口校准报告
│   ├── prediction_history.json # 预测战绩存档
│   ├── odds_quota.json        # 配额状态（4 Key 剩余量）
│   ├── status.json            # 数据新鲜度清单（🕐抽屉）
│   ├── version.json           # 缓存指纹（?v=）
│   └── team_names.json        # 中英队名映射（480+）
├── .github/workflows/
│   ├── daily.yml              # 每天 10:00 完整流水线
│   ├── results.yml            # 每 30 分钟赛果刷新
│   └── lineups.yml            # 每小时首发
├── local_sync_*.bat           # 本地计划任务脚本
└── .env                       # 本地密钥（已 gitignore）
```

---

## 4. 预测模型

### 4.1 三模型融合

```
最终概率 = 0.6×市场(去水隐含) + 0.25×Elo(赛果) + 0.15×xG(攻防强度)
```

| 模型 | 来源 | 说明 |
|---|---|---|
| 市场 | The Odds API 去水 | 庄家定价的最强先验，主锚 |
| Elo | 积分榜初始化+485场上赛季迭代+本赛季已完赛 | 看赛果，K=32，主场+100 |
| xG | 上赛季 485 场 xgHome/xgAway 训练，每队最近 20 场 | 看过程，λ=攻×防/均值 |

消息信号（RSS 关键词）**不直接改概率**，只参与置信度（权重 0.1）。

### 4.2 输出内容（每场比赛）

- `probabilities`：融合后胜平负概率（和=1）
- `predictedScore`：与 probabilities 同方向的最高概率波胆
- `correctScores`：Dixon-Coles 修正波胆 Top6
- `overUnder` / `spModel`：大小球/让球模型 vs 盘口对比（普通泊松，有意不用 DC）
- `kelly`：凯利注额（模型概率 vs 欧赔）
- `valuePicks`：模型与市场差 ≥10% 的价值信号（gold/watch）
- `diverge`：Elo/DC/市场三方方向不一致标记
- `confidence`：0.4+概率集中度×0.4+消息×0.1，分歧时 ×0.85
- `aiJudge`：DeepSeek 研判（pick/比分/置信度/理由，置信度钳制在模型±15%）

### 4.3 防未来函数

所有训练数据（Elo/xG/form）只取**早于预测窗口最早开赛日**的赛果，避免数据泄露。

---

## 5. CI/CD 工作流

### daily.yml（每天 10:00 北京时间 + push 触发 + 手动）
```
fetch_data(赛程/积分榜) → odds_fetcher(欧赔,仅定时) → asian → messages
→ predict(定时含AI / push仅统计) → backtest ×2 → evolve → build_status
→ build_version → 写回 main[skip ci] → 部署 gh-pages
```

### results.yml（每 30 分钟 + 每 4 小时带 AI）
```
fetch_data --no-standings → messages → predict(--skip-ai 高频/含AI 每4h)
→ build_status → build_version → 部署
```

### lineups.yml（每小时）
```
lineup_fetcher(多源回退) → 写回 lineups.json[skip ci]（rebase+重试）
```

### 关键机制
- **push 不抓赔率**：`if: github.event_name != 'push'`，省配额
- **CI 数据写回**：抓完 commit 回 main，避免被本地 push 覆盖回旧版
- **[skip ci] 防循环**：写回提交带 skip ci
- **写回防冲突**：`git pull --rebase --autostash` + 3 次重试

---

## 6. 前端（index.html 单文件 SPA）

### 结构
- ~121KB / 1830 行，CSS 内联 <style>，JS 内联 <script>
- 无框架、无外部依赖（纯原生 JS + fetch）
- 移动端优先（viewport + 4 组媒体查询）

### 页面组织
```
顶部: logo(h1) + 🕐数据状态 + 🔄刷新 + 更新时间
导航: 联赛页签(PL/PD/SA/BL1/FL1/CL) + 全局页签(🎯串关/📊战绩)
主区: 按联赛/子页签渲染
  - 赛程赛果(卡片列表: 预测比分/置信度/价值标签/竞彩标记/火花图)
  - 积分榜 / xG榜 / 球队数据(FBref) / 射手榜
详情弹窗: 6页签(胜平负/竞彩SP/大小球/让球/亚盘/波胆) + AI研判 + 首发 + 交锋 + 实时xG
全局: 🎯串关推荐(价值信号组合) / 📊战绩(校准报告/ROI/结算记录)
底部: 页脚数据状态简版
```

### 数据加载
- `loadAll()`：Promise.all 并行拉 32 个 JSON（9 固定 + 6 xg + 6 odds + 5 advanced + 5 asian + status）
- **缓存策略**：`version.json` 指纹（MD5 汇总）→ `?v=<hash>`；数据变才重下，未变命中缓存；version.json 本身恒新鲜
- 失败降级：单文件失败显示红色横幅不白屏

### 数据状态抽屉（🕐）
- CI 生成 `status.json`（13 模块 updatedAt/ok）
- 前端按统一阈值着色：**≤12h 绿 / 12-24h 黄 / >24h 红**（灰=正常无数据如欧冠休赛期）
- 相对时间每 1 分钟刷新，状态每 5 分钟轮询
- 欧赔模块显示剩余配额

---

## 7. 本地电脑任务

| 任务 | 频率 | 脚本 | 输出 | 备注 |
|---|---|---|---|---|
| FootballLocalXG | 每天 9:00 | `local_sync_xg.bat` → xg_fetch_local.py --push | xg/{联赛}.json | 赛季榜(已停用,上云) |
| FootballLocalFBref | 每天 10:30 | `local_sync_fbref.bat` → fbref_advanced.py | advanced/{联赛}.json | 空数据保护(已停用,上云) |

**依赖**：Python 3.11 + requests + beautifulsoup4 + lxml + playwright（xg）+ soccerdata（fbref）

**注意事项**：
- 竞彩是唯一依赖本地的数据源，电脑关机则网页显示"竞彩数据停更"横幅
- xg/fbref 任务在休赛期空转属正常（新赛季开赛后自动有数据）
- 任务间可能撞车（0x800710E0），若遇失败重跑即可

---

## 8. 配置与密钥

### GitHub Secrets（Settings → Secrets and variables → Actions）
| Secret | 用途 |
|---|---|
| FOOTBALL_API_KEY | football-data.org |
| ODDS_API_KEY / _2 / _3 / _4 | The Odds API 4 Key |
| AI_API_KEY / AI_BASE_URL / AI_MODEL | DeepSeek |

### 本地 .env（已 gitignore）
```
FOOTBALL_API_KEY=...
ODDS_API_KEY=...
AI_API_KEY=...
```

### 关键配置（config/prediction_rules.json）
```json
"fusion": { "marketWeight": 0.6, "eloWeight": 0.25, "xgWeight": 0.15, "messageConfWeight": 0.1 },
"confidenceMin": 0.4,
"valueThreshold": 0.1
```

---

## 9. 维护操作手册

### 常用操作
```bash
# 本地跑预测（--skip-ai 省额度）
python scripts/predict.py --skip-ai

# 手动抓欧赔（6 联赛，约 6 次配额）
python scripts/odds_fetcher.py --leagues PL PD SA BL1 FL1 CL

# 抓竞彩并推送
python scripts/jingcai_fetch_local.py --push

# 查配额
cat data/odds_quota.json

# 重建状态/版本指纹
python scripts/build_status.py && python scripts/build_version.py
```

### 更新流程（普通改动）
```bash
git add -A && git commit -m "..." && git pull --rebase origin main && git push origin main
```
（push 会触发部署，不消耗赔率配额；若改的是抓取脚本需手动 workflow_dispatch 验证）

### 故障排查
| 现象 | 排查 |
|---|---|
| 网页数据旧 | 🕐 抽屉看哪个模块红 → 查对应任务/工作流日志 |
| 欧赔不更新 | 查配额（odds_quota.json）是否耗尽 → 等 9/1 重置或加 Key |
| 竞彩不更新 | 电脑关机/计划任务停 → 检查 FootballLocalJC |
| 部署失败 | GitHub Actions → daily.yml 运行日志 |
| push 被拒 | 网络问题重试；或 `git pull --rebase` 后重推 |

### 新人接手清单
1. 配好 GitHub Secrets（8 个）
2. 本地克隆 + 装依赖 + 配 .env
3. 创建 3 个计划任务（bat 已就绪）
4. 手动跑一次 `python scripts/predict.py` 验证
5. 确认 🕐 抽屉全绿即上线

---

## 10. 已知限制与设计取舍

- **竞彩依赖本地电脑**（接口限国内 IP），云端无法替代
- **欧赔每天只抓 1 次**（配额限制），临场赔率变动捕捉有限；可手动 workflow_dispatch 补抓
- **xG 实时单场功能已停用**（原 15 分钟任务已禁用），如需恢复：启用计划任务 + 恢复门控脚本
- 大小球/让球用普通泊松（非 DC），有意为之（边际影响 <1%，避免过度拟合）
- AI 置信度钳制在模型概率 ±15%，避免 AI 离谱
- 状态灯阈值 12h/24h 为统一简化，个别模块（如亚盘每周更新）会偏黄属预期
