@echo off
chcp 65001 >nul
cd /d C:\Users\QZ\Desktop\FOOTBALL
echo [XG matches sync] gate check ...
python scripts\xg_matches_gate.py
if %errorlevel% neq 0 (
    echo No match window, skip fetch
    exit /b 0
)
echo Match window, fetch match xG ...
python scripts\xg_fetch_local.py --matches --push
echo done
