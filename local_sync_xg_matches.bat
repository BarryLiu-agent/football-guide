@echo off
cd /d C:/Users/QZ/Desktop/FOOTBALL
echo [XG matches sync] live/played match xG fetch + push ...
python scripts/xg_fetch_local.py --matches --push
echo done
