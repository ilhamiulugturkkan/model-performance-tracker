import sqlite3

DB_PATH = "/Users/ilhamiulugturkkan/Desktop/model-performance-tracker/mpt.db"


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
