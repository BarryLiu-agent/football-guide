@echo off
cd /d C:/Users/QZ/Desktop/FOOTBALL
echo [FBref sync] advanced stats fetch + push ...
python scripts/fbref_advanced.py
git add data/advanced
git commit -m "chore: local FBref advanced data update" -q
git pull --rebase --autostash origin main -q
git push origin main -q && echo push OK || echo push FAILED
echo done
