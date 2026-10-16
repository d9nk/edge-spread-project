#!/bin/zsh
set -e
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
fi
WAIT_UNTIL=${1:-23:57}

now=$(date +%s)
target=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $WAIT_UNTIL" +%s)
if [ "$target" -le "$now" ]; then target=$((target + 86400)); fi
echo "$(date) sleeping until $(date -r $target)"
sleep $((target - now))

day=$(date +%F)
echo "$(date) fetching trades for $day"
$PY scripts/fetch_moex_trades.py
rm -rf data/moex/candles/1m/$day data/moex/candles/10m/$day
$PY scripts/compute_moex_quoted_benchmark.py --include-today
$PY scripts/make_figures.py
$PY scripts/make_notebook.py
git add -A
git commit -q -m "MOEX benchmark: full day $day" || true
echo "$(date) done"
