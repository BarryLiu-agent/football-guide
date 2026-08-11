"""
FBref 多赛季赛程抓取(通过 soccerdata)
抓取 N 个赛季的比赛结果,输出 data/season_20XX.json
格式与现有 season_2025.json 兼容(供 backtest.py 使用)
"""
import json, sys, os, warnings, time, random
from datetime import datetime, timezone
from soccerdata import FBref

warnings.filterwarnings('ignore')

LEAGUE_MAP = {
    'PL': 'ENG-Premier League',
    'PD': 'ESP-La Liga',
    'BL1': 'GER-Bundesliga',
    'SA': 'ITA-Serie A',
    'FL1': 'FRA-Ligue 1',
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_season(season_year: int):
    """抓取单赛季全部 5 联赛的比赛结果"""
    all_matches = []

    for i, (code, league_name) in enumerate(LEAGUE_MAP.items()):
        if i > 0:
            delay = random.randint(30, 60)
            print(f'  等待 {delay}s...', flush=True)
            time.sleep(delay)

        try:
            print(f'  抓取 {league_name} ({season_year})...', flush=True)
            fb = FBref(leagues=league_name, seasons=season_year)
            df = fb.read_schedule()

            count = 0
            for idx, row in df.iterrows():
                # row index: (league, season, game_id) or similar
                home_goals = None
                away_goals = None
                score_str = str(row.get('score', ''))

                if score_str and score_str != 'nan':
                    parts = score_str.replace('-', '-').split('-')
                    if len(parts) == 2:
                        try:
                            home_goals = int(parts[0].strip())
                            away_goals = int(parts[1].strip())
                        except ValueError:
                            pass

                # 只保留有比分的已完赛比赛
                if home_goals is None:
                    continue

                match = {
                    'league': code,
                    'homeTeam': str(row.get('home_team', '')),
                    'awayTeam': str(row.get('away_team', '')),
                    'homeGoals': home_goals,
                    'awayGoals': away_goals,
                    'utcDate': str(row.get('date', '')),
                    'season': f'{season_year}/{str(season_year+1)[2:]}',
                }
                all_matches.append(match)
                count += 1

            print(f'    {count} 场有结果', flush=True)

        except Exception as e:
            print(f'    [FAIL] {e}', flush=True)

    return all_matches


def main():
    if len(sys.argv) < 2:
        print('用法: python fbref_seasons.py <起始年> [结束年]')
        print('示例: python fbref_seasons.py 2021 2025')
        sys.exit(1)

    start = int(sys.argv[1])
    end = int(sys.argv[2]) if len(sys.argv) > 2 else start

    for year in range(start, end + 1):
        out_path = os.path.join(DATA_DIR, f'season_{year}.json')
        if os.path.exists(out_path):
            print(f'[SKIP] season_{year}.json exists', flush=True)
            continue

        print(f'\n=== 赛季 {year}/{str(year+1)[2:]} ===', flush=True)
        matches = fetch_season(year)

        # 全部联赛失败 → 保留旧文件，避免写空数据覆盖
        if not matches and os.path.exists(out_path):
            print(f'[SKIP] 本次无数据，保留旧 season_{year}.json', flush=True)
            continue

        output = {
            'generatedAt': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'total': len(matches),
            'matches': matches,
            'source': 'FBref (soccerdata)',
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False)

        print(f'[OK] season_{year}.json: {len(matches)} 场', flush=True)


if __name__ == '__main__':
    main()
