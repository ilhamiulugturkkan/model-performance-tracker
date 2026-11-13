#!/usr/bin/env bash
# Nightly local backup of the monitoring database.
# Uses SQLite's own backup API (scripts/snapshot.py), which is safe while the scheduler is writing.
# Settings can be overridden from the environment: BACKUP_DIR, KEEP_DAYS, PYTHON.
set -euo pipefail

cd "$(dirname "$0")/.."                      # the repo folder, wherever this is run from
BACKUP_DIR="${BACKUP_DIR:-$HOME/backups}"
KEEP_DAYS="${KEEP_DAYS:-14}"
PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p "$BACKUP_DIR"
target="$BACKUP_DIR/mpt-$(date -u +%F).db"   # one file per UTC day; a second run the same day replaces it
"$PYTHON" scripts/snapshot.py "$target" mpt.db
find "$BACKUP_DIR" -name 'mpt-*.db' -mtime +"$KEEP_DAYS" -delete   # forget backups older than KEEP_DAYS
echo "backup written: $target"