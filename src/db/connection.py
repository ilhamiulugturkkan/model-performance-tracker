import sqlite3

DB_PATH = "/Users/ilhamiulugturkkan/Desktop/model-performance-tracker/mpt.db"


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
