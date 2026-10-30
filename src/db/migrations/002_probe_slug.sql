ALTER TABLE probes ADD COLUMN slug TEXT;
CREATE UNIQUE INDEX idx_probes_slug_version ON probes (slug, version);