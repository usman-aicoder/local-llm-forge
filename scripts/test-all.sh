#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Local LLM Forge — Full Regression Test Runner
#
# Runs all backend unit tests and frontend smoke tests.
# Must be executed from the repository root:
#   bash scripts/test-all.sh
#
# Requirements:
#   backend/venv must exist (pip install -r requirements.txt -r requirements-test.txt)
#   MongoDB must be running on localhost:27017
#   Node.js / npm must be installed
#   Playwright chromium must be installed (npx playwright install chromium)
#
# Optional:
#   Start the backend (uvicorn) and frontend (npm run dev) before running
#   to also execute the API-dependent Playwright tests.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
DASHBOARD_DIR="$REPO_ROOT/dashboard"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✔ $1${NC}"; }
fail() { echo -e "${RED}✘ $1${NC}"; }
info() { echo -e "${YELLOW}→ $1${NC}"; }

echo ""
echo "════════════════════════════════════════════"
echo "  Local LLM Forge — Regression Test Suite"
echo "════════════════════════════════════════════"
echo ""

# ── Pre-flight: check MongoDB is up ──────────────────────────────────────────

info "Checking MongoDB..."
# Use Python (already needed for backend) to ping MongoDB
MONGO_PING=$(cd "$BACKEND_DIR" && source venv/bin/activate && \
    python -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017',serverSelectionTimeoutMS=3000).admin.command('ping'); print('ok')" 2>/dev/null)
if [ "$MONGO_PING" = "ok" ]; then
    pass "MongoDB is running"
else
    fail "MongoDB is not running. Start it first: sudo systemctl start mongod"
    exit 1
fi

# ── Backend unit tests ────────────────────────────────────────────────────────

echo ""
info "Running backend unit tests..."
echo "────────────────────────────────────────────"

cd "$BACKEND_DIR"

if [ ! -d "venv" ]; then
    fail "backend/venv not found. Run: cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt -r requirements-test.txt"
    exit 1
fi

source venv/bin/activate
BACKEND_EXIT=0
python -m pytest tests/ -v --tb=short 2>&1 || BACKEND_EXIT=$?
deactivate

echo ""
if [ $BACKEND_EXIT -eq 0 ]; then
    pass "All backend tests passed"
else
    fail "Backend tests FAILED (exit code $BACKEND_EXIT)"
fi

# ── Frontend smoke tests ──────────────────────────────────────────────────────

echo ""
info "Running frontend Playwright smoke tests..."
echo "────────────────────────────────────────────"

cd "$DASHBOARD_DIR"

if [ ! -d "node_modules" ]; then
    fail "dashboard/node_modules not found. Run: cd dashboard && npm install"
    exit 1
fi

PLAYWRIGHT_EXIT=0
npx playwright test --reporter=list 2>&1 || PLAYWRIGHT_EXIT=$?

echo ""
if [ $PLAYWRIGHT_EXIT -eq 0 ]; then
    pass "All Playwright tests passed (skipped = backend not running)"
else
    fail "Playwright tests FAILED (exit code $PLAYWRIGHT_EXIT)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════"
TOTAL_EXIT=$((BACKEND_EXIT + PLAYWRIGHT_EXIT))
if [ $TOTAL_EXIT -eq 0 ]; then
    echo -e "${GREEN}  ALL TESTS PASSED — safe to proceed${NC}"
else
    echo -e "${RED}  TESTS FAILED — do not proceed to next phase${NC}"
fi
echo "════════════════════════════════════════════"
echo ""

exit $TOTAL_EXIT
