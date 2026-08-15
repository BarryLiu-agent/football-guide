"""
FBref 高级数据抓取(通过 soccerdata)
抓取 5 大联赛的:射门、门将、纪律(Misc) 三维度赛季统计
输出 data/advanced/{league}.json 供前端展示
"""
import json, sys, os, warnings, time, random
from datetime import datetime, timezone
from soccerdata import FBref

warnings.filterwarnings('ignore')

# 联赛映射:项目缩写 → soccerdata 联赛名
LEAGUE_MAP = {
    'PL':  'ENG-Premier League',
    'PD':  'ESP-La Liga',
    'BL1': 'GER-Bundesliga',
    'SA':  'ITA-Serie A',
    'FL1': 'FRA-Ligue 1',
}

# stat_type → 输出 key
STAT_KEYS = {
    'shooting': 'shooting',
    'keeper':   'goalkeeper',
    'misc':     'discipline',
}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'advanced')
os.makedirs(OUT_DIR, exist_ok=True)


def safe_float(v):
    """安全转浮点数,NaN→None"""
    try:
        f = float(v)
        return None if (f != f) else round(f, 2)  # NaN check
    except (ValueError, TypeError):
        return None


def fetch_league_stats(league_code: str, season: int) -> dict:
    """
    抓取单联赛三个维度的球队赛季统计
    返回 { shooting: [...], goalkeeper: [...], discipline: [...] }
    """
    league_name = LEAGUE_MAP[league_code]
    print(f'  抓取 {league_name} ({season})...', flush=True)

    result = {'shooting': [], 'goalkeeper': [], 'discipline': []}

    for stat_type, out_key in STAT_KEYS.items():
        try:
            fb = FBref(leagues=league_name, seasons=season)
            df = fb.read_team_season_stats(stat_type=stat_type)

            for team_name, row in df.iterrows():
                # row index: (league, season, team)
                team = team_name[2] if isinstance(team_name, tuple) else str(team_name)
                entry = {'team': team}

                if stat_type == 'shooting':
                    # 射门数据
                    entry['goals']       = safe_float(row.get(('Standard', 'Gls'), 0))
                    entry['shots']       = safe_float(row.get(('Standard', 'Sh'), 0))
                    entry['shotsOnTarget'] = safe_float(row.get(('Standard', 'SoT'), 0))
                    entry['sotPct']      = safe_float(row.get(('Standard', 'SoT%'), 0))
                    entry['shotsPer90']  = safe_float(row.get(('Standard', 'Sh/90'), 0))
                    entry['sotPer90']    = safe_float(row.get(('Standard', 'SoT/90'), 0))
                    entry['goalPerShot'] = safe_float(row.get(('Standard', 'G/Sh'), 0))
                    entry['goalPerSot']  = safe_float(row.get(('Standard', 'G/SoT'), 0))
                    entry['pkGoals']     = safe_float(row.get(('Standard', 'PK'), 0))
                    entry['pkAttempts']  = safe_float(row.get(('Standard', 'PKatt'), 0))

                elif stat_type == 'keeper':
                    # 门将数据
                    entry['goalsAgainst']   = safe_float(row.get(('Performance', 'GA'), 0))
                    entry['gaPer90']         = safe_float(row.get(('Performance', 'GA90'), 0))
                    entry['shotsOnTargetAgainst'] = safe_float(row.get(('Performance', 'SoTA'), 0))
                    entry['saves']           = safe_float(row.get(('Performance', 'Saves'), 0))
                    entry['savePct']         = safe_float(row.get(('Performance', 'Save%'), 0))
                    entry['cleanSheets']     = safe_float(row.get(('Performance', 'CS'), 0))
                    entry['csPct']           = safe_float(row.get(('Performance', 'CS%'), 0))
                    entry['pkSaved']         = safe_float(row.get(('Penalty Kicks', 'PKsv'), 0))
                    entry['pkAgainst']       = safe_float(row.get(('Penalty Kicks', 'PKA'), 0))

                elif stat_type == 'misc':
                    # 纪律/杂项
                    entry['yellowCards']  = safe_float(row.get(('Performance', 'CrdY'), 0))
                    entry['redCards']     = safe_float(row.get(('Performance', 'CrdR'), 0))
                    entry['secondYellow'] = safe_float(row.get(('Performance', '2CrdY'), 0))
                    entry['fouls']        = safe_float(row.get(('Performance', 'Fls'), 0))
                    entry['fouled']       = safe_float(row.get(('Performance', 'Fld'), 0))
                    entry['offsides']     = safe_float(row.get(('Performance', 'Off'), 0))
                    entry['crosses']      = safe_float(row.get(('Performance', 'Crs'), 0))
                    entry['interceptions'] = safe_float(row.get(('Performance', 'Int'), 0))
                    entry['tacklesWon']   = safe_float(row.get(('Performance', 'TklW'), 0))
                    entry['pkWon']        = safe_float(row.get(('Performance', 'PKwon'), 0))
                    entry['pkConceded']   = safe_float(row.get(('Performance', 'PKcon'), 0))
                    entry['ownGoals']     = safe_float(row.get(('Performance', 'OG'), 0))

                result[out_key].append(entry)

            print(f'    {stat_type}: {len(result[out_key])} 队', flush=True)

        except Exception as e:
            print(f'    {stat_type} 失败: {e}', flush=True)

    return result


def fetch_all(season: int = None):
    """抓取全部 5 联赛"""
    if season is None:
        # 默认当前年(2026 赛季用 2026)
        season = datetime.now().year

    for i, code in enumerate(LEAGUE_MAP):
        out_path = os.path.join(OUT_DIR, f'{code}.json')
        if i > 0:
            delay = random.randint(30, 60)
            print(f'  等待 {delay}s 避免限流...', flush=True)
            time.sleep(delay)
        try:
            stats = fetch_league_stats(code, season)

            # 空数据保护：新赛季未开赛时 FBref 返回空表，不能覆盖已有旧赛季数据
            # （脚本默认抓当前年份赛季；开赛前 2026 为空 → 保留 2025 旧数据）
            team_count = len(stats.get('shooting') or [])
            if team_count < 10:
                print(f'  [SKIP] {code}: 仅 {team_count} 队(赛季未开赛?), 保留旧文件', flush=True)
                continue

            output = {
                'generatedAt': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'league': code,
                'season': season,
                'data': stats,
            }
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
            print(f'  [OK] {code}.json saved ({team_count} teams)', flush=True)

        except Exception as e:
            print(f'  [FAIL] {code}: {e}', flush=True)


if __name__ == '__main__':
    season = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f'FBref 高级数据抓取 (赛季 {season or "自动"})', flush=True)
    fetch_all(season)
    print('完成!', flush=True)
