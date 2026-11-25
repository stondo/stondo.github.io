#!/usr/bin/env python3
"""Run selected core cases of the 2026-11-05 suite through one server (same file format as benchmark.py).

  run_cases.py --label LABEL --model MODEL [--thinking] [--max-tokens N] CASE_ID [CASE_ID ...]
API base and sampling come from BENCH_API and BENCH_REQUEST_KWARGS, as in the other scripts of this copy.
"""
import argparse
import json
import os
from pathlib import Path

from benchmark import Sample, quality_task
from tasks import dataset

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--thinking', action='store_true')
parser.add_argument('--max-tokens', type=int, default=int(os.environ.get('BENCH_MAX_TOKENS', '4096')))
parser.add_argument('cases', nargs='+')
args = parser.parse_args()
dest = ROOT / 'results' / args.label
dest.mkdir(mode=0o700, parents=True, exist_ok=True)
kwargs = json.loads(os.environ.get('BENCH_REQUEST_KWARGS', '{}')) or None
by_id = {t['id']: t for t in dataset()}
rows = []
for case in args.cases:
    task = by_id[case]
    path = dest / (case + ('.thinking' if args.thinking else '') + '.json')
    if path.exists():
        rows.append(json.loads(path.read_text()))
        continue
    with Sample() as sample:
        r = quality_task(task, os.environ.get('BENCH_API', 'http://127.0.0.1:8080') + '/v1', args.model, dest / case,
                         args.thinking, max_tokens=args.max_tokens, request_kwargs=kwargs)
    r['sampled_resources'] = sample.result()
    path.write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
    rows.append(r)
    print(json.dumps({'label': args.label, 'task': case, 'thinking': args.thinking, 'passed': r['grade']['passed'],
                      'seconds': r.get('seconds'), 'completion_tokens': r.get('completion_tokens'),
                      'finish': (r.get('turns') or [{}])[-1].get('finish_reason')}), flush=True)
