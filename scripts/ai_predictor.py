"""
AI 最终研判模块（可选增强层，不影响纯统计预测）。

设计原则:
  - AI 只做"综合研判": 基于统计模型的输出 + 盘口 + 消息信号, 给出最终比分/方向/
    中文分析。核心概率仍由统计模型产生, AI 输出与模型融合展示。
  - 任何失败(超时/限流/格式错) → 返回 None, 调用方降级为纯统计结果。
  - 密钥从环境变量 AI_API_KEY 读取 (GitHub Actions Secrets / 本地 .env),
    绝不硬编码。

用法:
    python scripts/ai_predictor.py            # 自检: 用一条示例数据调用一次
    python scripts/ai_predictor.py --dry      # 只检查配置与模型连通性

在 predict.py 中:
    from ai_predictor import ai_judge
    ai = ai_judge(pred)   # 成功返回 dict, 失败返回 None
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

# ── 配置 ──────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"   # DeepSeek (OpenAI 兼容)
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 45          # 秒/场
MAX_RETRY = 2                 # 失败重试次数（含首次共 MAX_RETRY+1 次）

PROMPT_SYSTEM = (
    "你是一名资深足球数据分析师。根据提供的统计数据、盘口和消息信号，"
    "对比赛做全面深度研判。规则：\n"
    "1. 只基于提供的数据推理，禁止臆测伤病、停赛等未提供信息；\n"
    "2. 输出必须是合法 JSON，包含以下字段：\n"
    '   {\n'
    '     "pick": "home|draw|away",\n'
    '     "score": "X-Y",\n'
    '     "confidence": 0到1之间数字,\n'
    '     "analysis": {\n'
    '       "h2h": "胜平负分析(结合欧赔与模型概率,指出市场态度)",\n'
    '       "ou": "大小球分析(盘口线与两队进攻防守特点)",\n'
    '       "spread": "让球分析(盘口是否合理,有无诱盘迹象)",\n'
    '       "btts": "双方进球概率判断",\n'
    '       "score": "最可能比分及理由"\n'
    '     },\n'
    '     "keyFactors": ["关键因素1(数据依据)", "关键因素2", "关键因素3"],\n'
    '     "risks": ["风险点1", "风险点2"],\n'
    '     "altScore": "备选比分 X-Y",\n'
    '     "reason": "一句话总结(中文)"\n'
    '   }\n'
    "3. confidence 应在统计模型 confidence 上下 0.15 以内微调，不要偏离过大；\n"
    "4. 如果认为没有明显倾向，pick 用 'none'，confidence 用 0.5；\n"
    "5. analysis 各子项要具体引用数据（赔率/概率/盘口数值），不要空泛；\n"
    "6. keyFactors 至少 2 条、risks 至少 1 条，都要有数据支撑。\n"
)


def _load_env() -> None:
    """加载 .env (仅当环境变量未设置时)。"""
    if os.environ.get("AI_API_KEY"):
        return
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(os.path.dirname(base), ".env")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
    except Exception:
        pass


def is_configured() -> bool:
    _load_env()
    return bool(os.environ.get("AI_API_KEY"))


def _call_llm(messages: list[dict], timeout: int = DEFAULT_TIMEOUT) -> str | None:
    """调用 OpenAI 兼容接口，返回 assistant 消息文本；失败返回 None。"""
    _load_env()
    api_key = os.environ.get("AI_API_KEY")
    if not api_key:
        return None
    base = (os.environ.get("AI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("AI_MODEL") or DEFAULT_MODEL
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.2,          # 低随机性保证可复现
        "response_format": {"type": "json_object"},
        "max_tokens": 400,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_err = None
    for attempt in range(MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}"
            if e.code in (401, 403):
                break  # 密钥问题，重试无意义
        except Exception as e:
            last_err = str(e)
        if attempt < MAX_RETRY:
            time.sleep(1.5 * (attempt + 1))
    print(f"[ai_predictor] 调用失败: {last_err}", file=sys.stderr)
    return None


def _build_context(pred: dict) -> str:
    """把统计模型输出压缩为 LLM 上下文（控制 token）。"""
    home = pred.get("homeTeam", "?")
    away = pred.get("awayTeam", "?")
    league = pred.get("league", "")
    kickoff = pred.get("kickoff", "")
    probs = pred.get("probabilities") or {}
    raw = pred.get("rawOdds") or {}
    cs = pred.get("correctScores") or []
    top3 = cs[:3]
    ou = pred.get("overUnder") or {}
    elo = pred.get("eloProb") or {}
    xg = pred.get("xgProb") or {}
    lu = pred.get("lineup") or {}
    st = pred.get("standings") or {}
    msg = pred.get("messageEvidence") or []
    f = pred.get("form") or {}
    kelly = pred.get("kelly") or {}
    vp = pred.get("valuePicks") or []
    rp = pred.get("reversePicks") or []
    sp = pred.get("spModel") or {}
    ou_m = pred.get("ouModel") or {}
    div = pred.get("diverge")
    dirs = pred.get("dirs") or {}
    lines = [
        f"比赛: {home} vs {away} ({league}) 开赛 {kickoff}",
        f"统计模型比分预测: {pred.get('predictedScore', '?')}",
        f"统计模型胜负概率: 主{probs.get('home')} 平{probs.get('draw')} 客{probs.get('away')}",
        f"欧赔: 主{raw.get('home')} 平{raw.get('draw')} 客{raw.get('away')}",
        f"Elo概率: 主{elo.get('home')} 平{elo.get('draw')} 客{elo.get('away')}",
        f"xG攻防模型概率: 主{xg.get('home')} 平{xg.get('draw')} 客{xg.get('away')}",
        f"大小球盘口: {ou.get('line')} 大{ou.get('over')} 小{ou.get('under')}" + (f"(赔率大{ou.get('overPrice')}/小{ou.get('underPrice')})" if ou.get('overPrice') else ""),
        f"积分榜: 主{st.get('home')} 客{st.get('away')}",
        f"近期状态(场均积分): 主{f.get('home')} 客{f.get('away')}",
    ]
    # 欧赔变化（初盘→临场）
    oc = pred.get("oddsChange") or {}
    if any(v is not None for v in oc.values()):
        lines.append(f"欧赔变化(较上次): 主{oc.get('home')} 平{oc.get('draw')} 客{oc.get('away')}")
    # 让球盘口 + 模型价值
    if sp.get("point") is not None:
        lines.append(f"让球盘: 主队让{sp.get('point')}(赔率{sp.get('price')}) 模型赢盘率{sp.get('cover')} vs 盘口隐含{sp.get('implied')} edge{sp.get('edge')}")
    # 大小球模型价值
    if ou_m.get("edge") is not None:
        lines.append(f"大小球模型: 模型大球{ou_m.get('over')} vs 盘口大球{ou.get('over')} edge{ou_m.get('edge')}")
    # 凯利（模型 vs 欧赔）
    ktxt = " ".join(f"{k}:{kelly.get(k)}" for k in ("home", "draw", "away") if kelly.get(k) is not None)
    if ktxt:
        lines.append(f"凯利指数: {ktxt}")
    # 价值信号 / 反向信号 / 模型分歧
    if vp:
        lines.append("价值信号: " + "; ".join(f"{v.get('label')}(模型{v.get('modelProb')} vs 盘口{v.get('oddsProb')} edge{v.get('edge')})" for v in vp[:3]))
    if rp:
        lines.append("反向信号(模型不看好): " + "; ".join(f"{v.get('label')}(edge{v.get('edge')})" for v in rp[:3]))
    if div is not None:
        lines.append(f"模型分歧: {'是(Elo/DC/市场方向不一致,谨慎)' if div else '否(各模型方向一致)'} 方向={dirs}")
    # 首发名单（赛前 1 小时左右公布，最重要的一次性赛前信息）
    if lu.get("homeLineup") or lu.get("awayLineup"):
        lu_tag = "正式" if lu.get("confirmed") else "预测"
        lines.append(f"首发名单[{lu_tag}/{lu.get('source')}] 主队: {', '.join(lu.get('homeLineup', [])[:18])}")
        lines.append(f"首发名单[{lu_tag}/{lu.get('source')}] 客队: {', '.join(lu.get('awayLineup', [])[:18])}")
    inj = lu.get("injuries") or {}
    if inj.get("home") or inj.get("away"):
        def _inj_txt(lst):
            return "; ".join(f"{i.get('name')}({i.get('status')})" for i in lst)
        lines.append(f"伤停: 主队 {_inj_txt(inj.get('home', []))} 客队 {_inj_txt(inj.get('away', []))}")
    if top3:
        lines.append("模型波胆Top3: " + ", ".join(f"{c.get('score', '?')}({c.get('prob')})" for c in top3))
    # 总进球分布 top5
    tgd = pred.get("totalGoalsDist") or {}
    if tgd:
        top_tgd = sorted(tgd.items(), key=lambda x: -x[1])[:5]
        lines.append("总进球分布: " + " ".join(f"{k}球{v}" for k, v in top_tgd))
    # 双方进球概率
    if pred.get("btts") is not None:
        lines.append(f"双方进球概率: {pred.get('btts')}")
    # 消息信号
    if msg:
        lines.append("消息信号: " + "; ".join(str(m)[:120] for m in msg[:3]))
    return "\n".join(lines)


def ai_judge(pred: dict, timeout: int = DEFAULT_TIMEOUT) -> dict | None:
    """对单场预测做 AI 最终研判。

    成功: 返回 {"pick","score","confidence","reason","model","latency"}
    失败(未配置/网络/格式): 返回 None。
    """
    if not is_configured():
        return None
    t0 = time.time()
    user = _build_context(pred)
    content = _call_llm([
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": user + "\n请输出 JSON。"},
    ], timeout=timeout)
    if not content:
        return None
    try:
        # 兼容 response_format 返回 JSON 或带代码块包装
        s = content.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.startswith("json"):
                s = s[4:]
        data = json.loads(s)
        pick = str(data.get("pick", "none")).strip().lower()
        if pick not in ("home", "draw", "away", "none"):
            pick = "none"
        score = str(data.get("score", "")).strip()
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        # 强制 ±0.15 约束（提示词只是建议，模型可能偏离；防止 AI 置信度与统计概率脱节）
        base = (pred.get("modelProbs") or {}).get(pick, 0.5) if pick in ("home", "draw", "away") else 0.5
        lo, hi = max(0.0, base - 0.15), min(1.0, base + 0.15)
        conf = max(lo, min(hi, conf))
        # 深度分析字段（新）：逐项信息面 + 关键因素 + 风险 + 备选比分
        analysis = data.get("analysis") or {}
        if not isinstance(analysis, dict):
            analysis = {}
        key_factors = data.get("keyFactors") or []
        if not isinstance(key_factors, list):
            key_factors = []
        risks = data.get("risks") or []
        if not isinstance(risks, list):
            risks = []
        return {
            "pick": pick,
            "score": score,
            "confidence": round(conf, 3),
            "reason": str(data.get("reason", "")).strip(),
            "analysis": {
                "h2h": str(analysis.get("h2h", "")).strip(),
                "ou": str(analysis.get("ou", "")).strip(),
                "spread": str(analysis.get("spread", "")).strip(),
                "btts": str(analysis.get("btts", "")).strip(),
                "score": str(analysis.get("score", "")).strip(),
            },
            "keyFactors": [str(x).strip() for x in key_factors if str(x).strip()][:6],
            "risks": [str(x).strip() for x in risks if str(x).strip()][:4],
            "altScore": str(data.get("altScore", "")).strip(),
            "model": os.environ.get("AI_MODEL") or DEFAULT_MODEL,
            "latency": round(time.time() - t0, 1),
        }
    except Exception as e:
        print(f"[ai_predictor] 输出解析失败: {e} | raw={content[:200]}", file=sys.stderr)
        return None


def main() -> int:
    _load_env()
    if not is_configured():
        print("未配置 AI_API_KEY（可在 .env 或环境变量设置）。AI 层跳过，不影响统计预测。")
        return 0 if "--dry" in sys.argv else 1
    print(f"AI 配置: model={os.environ.get('AI_MODEL', DEFAULT_MODEL)} base={os.environ.get('AI_BASE_URL', DEFAULT_BASE_URL)}")
    if "--dry" in sys.argv:
        print("连通性检查通过（仅配置检查，未发起调用）。")
        return 0
    # 自检：用一条示例预测调用
    sample = {
        "homeTeam": "Arsenal", "awayTeam": "Chelsea", "league": "PL",
        "kickoff": "2026-08-21T19:00:00Z",
        "predictedScore": "2-1",
        "probabilities": {"home": 0.45, "draw": 0.28, "away": 0.27},
        "rawOdds": {"home": 2.1, "draw": 3.6, "away": 3.8},
        "eloProb": {"home": 0.48, "draw": 0.26, "away": 0.26},
        "overUnder": {"line": 2.5, "over": 0.55, "under": 0.45},
        "standings": {"home": 2, "away": 5},
        "messageEvidence": [],
    }
    r = ai_judge(sample)
    if r:
        print("自检通过，示例输出:")
        print(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
        return 0
    print("自检失败（未配置或调用出错）。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
