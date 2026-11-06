ALTER TABLE calls ADD COLUMN repeat_no INTEGER NOT NULL DEFAULT 1;

ALTER TABLE calls ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1;

ALTER TABLE calls ADD COLUMN finish_reason TEXT;

ALTER TABLE calls ADD COLUMN reasoning_tokens INTEGER;

ALTER TABLE calls ADD COLUMN request_id TEXT;

-- A retry or a re-run must not create a second row for the same call.
CREATE UNIQUE INDEX idx_calls_one_per_repeat
    ON calls (run_id, probe_id, provider, request_string, repeat_no);

-- The question we ask all day: how did this model do over time?
CREATE INDEX idx_calls_model_time ON calls (request_string, time_stamp);

CREATE UNIQUE INDEX idx_scores_one_per_scorer ON scores (call_id, scorer);