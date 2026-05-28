#!/usr/bin/env bash
# Usage: bash push_to_github.sh [J1|J2|J3]
#   J1 = this machine's intersection label (default: J1)
set -e

JUNCTION=${1:-J1}
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== [1/3] Generate intersection.rou.xml for ${JUNCTION} ==="
cd "$REPO_DIR"
python3 generate_rou.py --junction "$JUNCTION"

echo ""
echo "=== [2/3] Stage files ==="
git add nets/intersection.net.xml \
        nets/intersection.rou.xml \
        nets/intersection.sumocfg \
        generate_rou.py \
        push_to_github.sh
# cache files (ignore if absent)
git add traffic_cache.json aggregate_traffic.json 2>/dev/null || true

echo ""
echo "=== [3/3] Commit & push ==="
if git diff --cached --quiet; then
    echo "Nothing new to commit."
else
    git commit -m "feat(${JUNCTION}): update route xml from cache [$(date +%Y-%m-%d)]"
    git push origin main
    echo ""
    echo "Pushed. Others can now:"
    echo "  git pull"
    echo "  python3 generate_rou.py --junction J2   # change to their junction number"
fi
