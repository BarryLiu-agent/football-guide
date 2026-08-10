@echo off
chcp 65001 >nul
cd /d C:\Users\QZ\Desktop\FOOTBALL
echo [%date% %time%] 竞彩 SP 抓取 + 推送
python scripts\jingcai_fetch_local.py --push
echo 完成
