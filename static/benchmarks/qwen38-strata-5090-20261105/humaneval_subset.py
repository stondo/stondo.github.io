"""Run a fixed public HumanEval subset through the local API and isolated tests."""
import argparse
import json
import time
from pathlib import Path

from benchmark import Sample, extract_code, mem, model_ram, request, sandbox

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()
dest = ROOT / 'results' / args.label / 'humaneval'
dest.mkdir(parents=True, exist_ok=True)
tasks = json.loads((ROOT / 'humaneval-subset.json').read_text())
for task in tasks:
    name = task['task_id'].replace('/', '-')
    output = dest / (name + '.json')
    if output.exists():
        continue
    prompt = ('Complete the Python function below. Return the complete Python module '
              'including the original imports, function signature and your implementation. '
              'Use only the Python standard library. Return code only.\n\n' + task['prompt'])
    with Sample() as sample:
        result = request('http://127.0.0.1:8080/v1', args.model,
                         [{'role': 'user', 'content': prompt}], max_tokens=4096,
                         thinking=False, temperature=0)
    tests = ('import random\nrandom.seed(20261105)\nfrom solution import *\n' + task['test'] + '\nfrom solution import ' + task['entry_point']
             + '\ncheck(' + task['entry_point'] + ')\n')
    grade = sandbox(extract_code(result['content']), tests, dest / name)
    row = {'task_id': task['task_id'], 'grade': grade, 'test_random_seed': 20261105, 'response': result,
           'sampled_resources': sample.result()}
    output.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps({'task': task['task_id'], 'passed': grade['passed'],
                      'seconds': result['seconds']}), flush=True)
rows = [json.loads((dest / (t['task_id'].replace('/', '-') + '.json')).read_text())
        for t in tasks]
summary = {'passed': sum(r['grade']['passed'] for r in rows), 'total': len(rows),
           'memory_after': mem(), 'gpu_process_memory_after': model_ram(),
           'completed_at': time.time()}
(dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary), flush=True)
