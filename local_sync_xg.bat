@echo off
cd /d C:/Users/QZ/Desktop/FOOTBALL
echo [XG sync] xg fetch + push ...
python scripts/xg_fetch_local.py --push
echo done
