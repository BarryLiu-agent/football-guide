@echo off
cd /d C:/Users/QZ/Desktop/FOOTBALL
echo [XG matches sync] gate check ...
python scripts/xg_matches_gate.py
if %errorlevel% neq 0 (
    echo 无比赛窗口, 跳过抓取
    exit /b 0
)
echo 有比赛窗口, 抓取单场 xG ...
python scripts/xg_fetch_local.py --matches --push
echo done
