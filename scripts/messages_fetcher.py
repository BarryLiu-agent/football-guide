"""
messages_fetcher.py - 消息数据收集
从可配置消息源（RSS/API/文件）抓取新闻、聊天记录、通知等，统一处理后生成 data/messages.json。

用法:
  python scripts/messages_fetcher.py

可扩展接口:
  MessageSource (抽象基类) - 实现 fetch() -> list[Message]
    已有实现: RssNewsSource (默认, 公开 RSS)
  新消息源(聊天导出/邮件/webhook): 继承 MessageSource, 在 config/news_sources.json 注册即可

统一消息模型:
  { source, type, text, timestamp, metadata }
"""

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ── 抽象接口 ─────────────────────────────────────────────

class MessageSource:
    """消息源抽象基类。新消息源继承并实现 fetch() 即可。"""

    def __init__(self, config: dict):
        self.config = config

    def fetch(self) -> list:
        """返回消息列表: [{source, type, text, timestamp, metadata}]"""
        raise NotImplementedError


# ── RSS 新闻源（默认）────────────────────────────────────

class RssNewsSource(MessageSource):
    """从公开 RSS 抓取新闻。"""

    def fetch(self) -> list:
        url = self.config.get("url", "")
        if not url:
            return []
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print(f"    - {self.config.get('id', url)}: 请求失败 {e}")
            return []

        items = self._parse(r.text)
        messages = []
        for it in items:
            messages.append({
                "source": self.config.get("id", "rss"),
                "type": "news",
                "text": it.get("title", ""),
                "timestamp": it.get("published", "") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metadata": {
                    "link": it.get("link", ""),
                    "lang": self.config.get("lang", "en"),
                    "summary": it.get("summary", "")[:500],
                },
            })
        return messages

    def _parse(self, xml_text: str) -> list:
        """解析 RSS/Atom XML。"""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            return []
        items = []
        try:
            root = ET.fromstring(xml_text)
            # RSS 2.0: /rss/channel/item
            for item in root.findall(".//item"):
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                pub = item.findtext("pubDate") or ""
                desc = item.findtext("description") or ""
                if title:
                    items.append({"title": title, "link": link, "published": pub, "summary": desc})
            # Atom: /feed/entry
            if not items:
                for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                    title = entry.findtext("{http://www.w3.org/2005/Atom}title") or ""
                    link_el = entry.find("{http://www.w3.org/2005/Atom}link")
                    link = link_el.get("href", "") if link_el is not None else ""
                    pub = entry.findtext("{http://www.w3.org/2005/Atom}updated") or ""
                    summary = entry.findtext("{http://www.w3.org/2005/Atom}summary") or ""
                    if title:
                        items.append({"title": title, "link": link, "published": pub, "summary": summary})
        except ET.ParseError as e:
            print(f"    - XML 解析失败: {e}")
        return items


# ── 源工厂 ───────────────────────────────────────────────

SOURCE_REGISTRY = {
    "rss": RssNewsSource,
}


def build_sources(config: dict) -> list:
    sources = []
    for item in config.get("sources", []):
        if not item.get("enabled", True):
            continue
        cls = SOURCE_REGISTRY.get(item.get("type", "rss"))
        if cls:
            sources.append(cls(item))
    return sources


# ── 主流程 ───────────────────────────────────────────────

def dedupe(messages: list) -> list:
    """按内容哈希去重。"""
    seen = set()
    result = []
    for m in messages:
        h = hashlib.md5(m["text"].encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(m)
    return result


def denoise(messages: list, min_len: int = 12) -> list:
    """去除过短/明显广告消息。"""
    result = []
    for m in messages:
        text = m["text"].strip()
        if len(text) < min_len:
            continue
        low = text.lower()
        if any(k in low for k in ["subscribe", "click here", "advertisement", "sponsored", "优惠", "点击购买"]):
            continue
        m["text"] = text
        result.append(m)
    return result


# 非足球体育项目的强特征词（板球/高尔夫/网球/橄榄球/拳击/赛马/棒球等）
# 命中任一即判定该消息与足球无关，丢弃（综合体育源如 skysports 12040 会混入这些）
_NON_FOOTBALL_HINTS = [
    # 板球
    "cricket", "wicket", "innings", "duck", "county select", "the hundred", "bowled",
    "runs", "overs", "test match", "ashes", "batter", "all-rounder", "sri lanka", "pakistan",
    "india", "australia", "england vs", "t20", "odi",
    # 高尔夫
    "golf", "mcilroy", "spieth", "fedexcup", "playoffs", "birdie", "eagle", "bogey",
    "par ", "masters", "ryder cup", "pga", "lpg", "greenway", "golf championship",
    "mory", "danish golf", "doyle", "hungerford", "fedex cup", "worcestershire",
    # 网球
    "tennis", "wimbledon", "grand slam", "set point", "break point", "cincinnati",
    # 橄榄球联赛/联盟
    "super league", "nrl", "rugby", "rfl", "challenge cup", "leeds rhinos", "hull fc",
    "catalans", "wigan", "salford", "st helens", "warrington",
    # 拳击/格斗/赛马/棒球/篮球
    "boxing", "bellator", "ufc", "knockout", "horse racing", "grand national",
    "nba", "nfl", "mlb", "nhl", "formula 1", "f1 ", "moto",
    "kalajdzic", "showdown", "face-off", "faceoff", "fisher bill", "fight night", "coppull",
    # 台球/斯诺克/其他
    "snooker", "darts", "world championship",
]

_FOOTBALL_HINTS = [
    # 强特征足球词：命中这些才视为"确实与足球相关"（豁免非足球词误杀）
    "football", "soccer", "premier league", "championship match", "la liga", "bundesliga",
    "serie a", "ligue 1", "champions league", "europa league", "efl",
    "wolves", "arsenal", "chelsea", "liverpool", "manchester", "tottenham", "celtic",
    "rangers", "leicester", "brighton", "everton", "newcastle", "west ham", "aston villa",
    "transfer", "signing", "loan deal", "contract", "striker", "midfielder",
    "defender", "keeper", "manager", "coach", "klopp", "guardiola", "mikel",
    "goal", "scored", "score", "penalty", "red card", "yellow card", "own goal",
    "world cup", "euro 2028", "qualifier", "friendly", "derby match", "stadium",
]


def keep_football(messages: list) -> list:
    """过滤非足球消息（综合体育源会混入板球/高尔夫等）。"""
    result = []
    for m in messages:
        low = (m.get("text", "") or "").lower()
        if not low:
            continue
        # 命中非足球强特征词 → 除非同时有明显足球词（如 "england" 可能两者）
        if any(h in low for h in _NON_FOOTBALL_HINTS):
            # 有明确足球词才保留（如 "Celtic sign" 同时含 "sign"）
            if not any(f in low for f in _FOOTBALL_HINTS):
                continue
        result.append(m)
    return result


def main():
    parser = argparse.ArgumentParser(description="消息数据收集")
    parser.add_argument("--limit", type=int, default=200, help="每源最多保留消息数")
    args = parser.parse_args()

    with open(CONFIG_DIR / "news_sources.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sources = build_sources(config)
    if not sources:
        print("没有启用的消息源")
        return 1

    print(f"消息源: {[type(s).__name__ for s in sources]}")
    all_messages = []
    for src in sources:
        print(f"  {src.config.get('id', '?')}...")
        msgs = src.fetch()
        print(f"    ✓ {len(msgs)} 条")
        all_messages.extend(msgs)
        time.sleep(0.5)

    all_messages = dedupe(all_messages)
    all_messages = denoise(all_messages)
    # 综合体育源会混入板球/高尔夫等非足球消息 → 只保留足球相关
    all_messages = keep_football(all_messages)
    if args.limit:
        all_messages = all_messages[: args.limit]

    # 抓取失败（所有源返回空）时保留旧文件，避免清空已有消息
    if not all_messages:
        old = DATA_DIR / "messages.json"
        if old.exists():
            try:
                old_data = json.loads(old.read_text(encoding="utf-8"))
                n_old = len(old_data.get("messages", []))
                if n_old:
                    print(f"    本次无新消息，保留旧数据 {n_old} 条")
                    return 0
            except Exception:
                pass

    out = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(all_messages),
        "messages": all_messages,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "messages.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    print(f"\n总计: {len(all_messages)} 条消息 -> data/messages.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
