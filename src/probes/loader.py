"""Seed the probes table from the probe files.

python3 -m src.probes.loader --db mpt_test.db probes/*.jsonl
"""

import argparse
import hashlib
import json
import sys

from src.db.connection import DB_PATH, connect
from src.probes.validate import check_common, check_scorer, check_set, load_lines

# Not part of a probe's identity: split is assigned later, topic is a label doesn't affect the model performance anyway.
NOT_HASHED = ("split", "topic")


class ProbeFileError(Exception):
    """One or more probe files failed validation."""

    def __init__(self, errors):
        super().__init__("\n".join(errors))
        self.errors = errors


class ProbeChangedError(Exception):
    """A loaded probe was edited in place, or its split disagrees with the database."""

    def __init__(self, problems):
        super().__init__("\n".join(problems))
        self.problems = problems


def canonical_body(probe):
    """The probe's content as one stable piece of text (sorted keys, no extra spaces)."""
    content = {key: value for key, value in probe.items() if key not in NOT_HASHED}
    return json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def content_hash(body):
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_probe_files(paths):
    """Return (entries, errors). entries is a list of (path, line_number, probe)."""
    entries, errors = [], []
    for path in paths:
        for number, probe, error in load_lines(path):
            if error:
                errors.append(f"{path}:{number}: {error}")
                continue
            problems = check_common(probe, str(path)) + check_scorer(probe)
            errors += [f"{path}:{number}: {problem}" for problem in problems]
            if isinstance(probe, dict):
                entries.append((str(path), number, probe))
    set_errors, _warnings, _summary = check_set(entries)
    return entries, errors + set_errors


def load_probes(conn, paths):
    """Put the probes from `paths` into the database. Returns a summary dict."""
    entries, errors = read_probe_files(paths)
    if errors:
        raise ProbeFileError(errors)

    summary = {
        "inserted": 0,
        "unchanged": 0,
        "split_set": 0,
        "topic_updated": 0,
        "only_in_db": [],
    }
    problems = []
    try:
        for path, number, probe in entries:
            slug, version = probe["id"], probe["version"]
            body = canonical_body(probe)
            new_hash = content_hash(body)
            row = conn.execute(
                "SELECT probe_id, content_hash, split, topic FROM probes WHERE slug = ? AND version = ?",
                (slug, version),
            ).fetchone()

            if row is None:  # does't exist, insert
                conn.execute(
                    "INSERT INTO probes (category, version, content_hash, slug, split, body, topic) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        probe["category"],
                        version,
                        new_hash,
                        slug,
                        probe.get("split"),
                        body,
                        probe.get("topic"),
                    ),
                )
                summary["inserted"] += 1
                continue

            probe_id, old_hash, old_split, old_topic = row
            if old_hash != new_hash:
                problems.append(
                    f"{path}:{number}: {slug} {version} was edited after it was loaded. "
                    f"Put the change in a new version (v{int(version[1:]) + 1}) and keep the old line."
                )
                continue

            new_split = probe.get("split")
            if old_split is None and new_split is not None:
                conn.execute(
                    "UPDATE probes SET split = ? WHERE probe_id = ?",
                    (new_split, probe_id),
                )
                summary["split_set"] += 1
            elif old_split != new_split:
                problems.append(
                    f"{path}:{number}: {slug} {version} has split {old_split!r} in the database "
                    f"but {new_split!r} in the file. A split is set once and never changed."
                )
                continue

            if probe.get("topic") != old_topic:
                conn.execute(
                    "UPDATE probes SET topic = ? WHERE probe_id = ?",
                    (probe.get("topic"), probe_id),
                )
                summary["topic_updated"] += 1
            if (
                old_hash == new_hash
                and old_split == new_split
                and probe.get("topic") == old_topic
            ):
                summary["unchanged"] += 1

        if problems:
            raise ProbeChangedError(problems)

        in_files = {(probe["id"], probe["version"]) for _path, _n, probe in entries}
        in_db = conn.execute("SELECT slug, version FROM probes").fetchall()
        summary["only_in_db"] = sorted(
            f"{slug} {version}"
            for slug, version in in_db
            if (slug, version) not in in_files
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return summary


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--db",
        default=str(DB_PATH),
        help="database file (default: the project database)",
    )
    parser.add_argument(
        "files", nargs="+", help="probe files, for example probes/*.jsonl"
    )
    args = parser.parse_args()

    conn = connect(args.db)
    try:
        summary = load_probes(conn, args.files)
    except (ProbeFileError, ProbeChangedError) as error:
        print(error)
        print("nothing was written")
        conn.close()
        sys.exit(1)
    conn.close()
    print(
        f"inserted {summary['inserted']}, unchanged {summary['unchanged']}, "
        f"split set {summary['split_set']}, topic updated {summary['topic_updated']}"
    )
    if summary["only_in_db"]:
        print(
            f"in the database but not in these files (kept): {', '.join(summary['only_in_db'])}"
        )


if __name__ == "__main__":
    _main()
