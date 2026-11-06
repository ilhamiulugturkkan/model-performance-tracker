
-- The full probe as canonical JSON: everything except split and topic.
-- content_hash is the SHA-256 of exactly this text, so the two can be checked against each other.
-- Storing it makes the database self-contained: the runner and the scorer need no probe files.
ALTER TABLE probes ADD COLUMN body TEXT;
 
-- Analysis label (for example letter_ban), kept in its own column so results can be sliced by it.
-- It is not part of the hash, because it describes a probe and does not change it.
ALTER TABLE probes ADD COLUMN topic TEXT;
 