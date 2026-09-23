#!/usr/bin/env python3
"""Collect completed E1/E2 official results without inventing missing cells."""
import argparse
import csv
import io
import json
from pathlib import Path
from e1e2_common import atomic_file, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root', required=True)
    args = p.parse_args()
    root = Path(args.work_root); rows = []
    for path in sorted((root/'test').glob('*/official/*/*/*/summary.json')):
        data = json.loads(path.read_text())
        rows.append({k: data[k] for k in ('variant', 'method', 'split', 'case', 'mean_accuracy', 'sample_interval', 'seen_role')})
    if not rows:
        raise RuntimeError('No completed official summaries; finish PHASES=eval first')
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    with atomic_file(root/'comparison.csv') as f:
        f.write(buf.getvalue().encode())
    save_json(root/'comparison.json', rows)
    print(f'[E1E2] {len(rows)} completed cells: {root / "comparison.csv"}')


if __name__ == '__main__':
    main()
