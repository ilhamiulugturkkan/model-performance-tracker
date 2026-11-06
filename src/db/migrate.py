from pathlib import Path

from src.db.connection import DB_PATH, connect

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def migrate(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)
    applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}

    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in applied:
            continue
        conn.executescript(path.read_text())
        conn.execute(
            "INSERT INTO schema_migrations (filename, applied_at) VALUES (?, datetime('now'))",
            (path.name,),
        )
        conn.commit()
        print(f"applied {path.name}")


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    conn = connect(db_path)
    try:
        migrate(conn)
    finally:
        conn.close()
