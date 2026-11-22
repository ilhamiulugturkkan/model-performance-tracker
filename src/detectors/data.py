from src.db.connection import connect


# returns a list of rows, one per call, as dicts. It only fetches, with no maths.
def load_window(model, category, run_ids, conn=None):
    # category "all" means every category (no category filter).
    # Each row has: run_id, probe_id, repeat_no, score, output_tokens, latency, raw_output, time_stamp.
    run_ids = list(run_ids)
    if not run_ids:
        return []
    if conn is None:
        conn = connect()

    placeholders = ",".join("?" * len(run_ids))
    cur = conn.execute(
        f"""SELECT c.run_id, c.probe_id, c.repeat_no, AVG(s.score) AS score,
                   c.output_tokens, c.latency, c.raw_output, c.time_stamp
            FROM calls c
            JOIN probes p ON p.probe_id = c.probe_id
            JOIN scores s ON s.call_id = c.call_id
            WHERE c.request_string = ? AND (? = 'all' OR p.category = ?) AND p.split = 'scheduled'
              AND c.error IS NULL
              AND c.run_id IN ({placeholders})
            GROUP BY c.call_id
            ORDER BY c.run_id, c.probe_id, c.repeat_no""",
        (model, category, category, *run_ids),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def model_provider(model, conn=None):
    """Provider name for a model (e.g. 'anthropic'), or None if the model has no calls."""
    if conn is None:
        conn = connect()
    row = conn.execute(
        "SELECT provider FROM calls WHERE request_string = ? LIMIT 1", (model,)
    ).fetchone()
    return row[0] if row else None


if __name__ == "__main__":
    ref_window = load_window("claude-sonnet-5-5", "math", [1, 2, 3, 4])
    print(len(ref_window))
