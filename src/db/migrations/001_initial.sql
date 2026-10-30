CREATE TABLE IF NOT EXISTS probes (
    probe_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    config TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    probe_id INTEGER NOT NULL REFERENCES probes(probe_id),
    provider TEXT NOT NULL,
    request_string TEXT NOT NULL,
    received_string TEXT,
    fingerprint TEXT,
    parameters TEXT NOT NULL,
    raw_prompt TEXT NOT NULL,
    raw_response TEXT,
    raw_output TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost REAL,
    latency REAL,
    http_status INTEGER,
    error TEXT,
    time_stamp TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER NOT NULL REFERENCES calls(call_id),
    scorer TEXT NOT NULL,
    score REAL NOT NULL,
    details TEXT
);
