import sqlite3
import sys
from pathlib import Path


def snapshot(source, target):
    source_connection = sqlite3.connect(
        f"file:{source}?mode=ro", uri=True
    )  # read-only: it can never change the original
    target_connection = sqlite3.connect(target)
    try:
        source_connection.backup(target_connection)
        problems = target_connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        target_connection.close()
        source_connection.close()
    if problems != "ok":
        raise SystemExit(f"snapshot is damaged: {problems}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/snapshot.db"
    source = (
        sys.argv[2]
        if len(sys.argv) > 2
        else str(Path(__file__).resolve().parent.parent / "mpt.db")
    )
    snapshot(source, target)
    print(f"snapshot written to {target}")
