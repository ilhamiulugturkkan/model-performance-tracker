import json
from pathlib import Path

for path in sorted(Path("probes").glob("*.jsonl")):
    probes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for probe in probes:
        probe["split"] = "scheduled"
    path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in probes),
        encoding="utf-8",
    )
    print(f"{path.name}: {len(probes)} probes labelled scheduled")
