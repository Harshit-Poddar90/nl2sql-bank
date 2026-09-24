#!/usr/bin/env bash
# Builds the warehouse into the mounted volume on first start, then runs the command:
#   serve (default) | ui | eval [args] | build | shell | any other command verbatim
set -euo pipefail

DB_PATH="${NL2SQL_DB_PATH:-/app/data/db/bank.sqlite}"
log() { printf '[entrypoint] %s\n' "$*" >&2; }

ensure_warehouse() {
    # Size check, not just -f: an interrupted build can leave a small, broken file.
    if [ -f "$DB_PATH" ] && [ "$(stat -c%s "$DB_PATH" 2>/dev/null || echo 0)" -gt 1000000 ]; then
        return
    fi
    log "no warehouse found -- building it (downloads 68 MB, about a minute)"
    nl2sql data build
}

case "${1:-serve}" in
    serve)
        ensure_warehouse
        exec uvicorn nl2sql.api.main:app --host "${NL2SQL_API_HOST:-0.0.0.0}" --port "${NL2SQL_API_PORT:-8000}"
        ;;
    ui)
        ensure_warehouse
        exec streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 \
            --server.headless true --browser.gatherUsageStats false
        ;;
    eval)
        ensure_warehouse
        shift
        exec nl2sql eval "$@"
        ;;
    build) exec nl2sql data build ;;
    shell) exec /bin/bash ;;
    *) exec "$@" ;;
esac
