@echo off
chcp 65001 >nul
cd /d C:\Users\QZ\Desktop\FOOTBALL
echo [%date% %time%] xG 联赛数据抓取 + 推送
python scripts\xg_fetch_local.py --push
echo 完成
