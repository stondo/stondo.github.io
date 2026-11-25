"""Recreate the frozen subset from its public source and expected SHA256."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parent
provenance = json.loads((root/'humaneval-provenance.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--data-file', type=Path, help='Use an existing HumanEval.jsonl.gz offline')
args = parser.parse_args()
if args.data_file:
    raw = args.data_file.read_bytes()
else:
    with urllib.request.urlopen(provenance['source'], timeout=60) as response:
        raw = response.read()
assert hashlib.sha256(raw).hexdigest() == provenance['sha256'], 'Upstream dataset bytes differ'
tasks = {row['task_id']: row for line in gzip.decompress(raw).splitlines()
         if (row := json.loads(line))}
selected = [tasks[key] for key in provenance['selected_ids']]
(root/'humaneval-subset.json').write_text(json.dumps(selected, indent=2)+'\n')
print(f'Prepared {len(selected)} fixed tasks; upstream SHA256 matches.')
