#!/usr/bin/env python3
"""Exercise the local context window using synthetic data only."""
import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--target-tokens', type=int, default=250000)
parser.add_argument('--label', default='native256')
parser.add_argument('--model', required=True)
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
from benchmark import Sample, mem, request

API = 'http://127.0.0.1:8080'
MODEL = args.model

def post(path, payload):
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)

rng = random.Random(5090262144)
rows = [f'record {i}: sector {rng.randrange(100)} quantity {rng.randrange(1000000)} '
        f'checksum {rng.getrandbits(48):012x}.' for i in range(30000)]
expected = {'early': 'MIST-4827', 'middle': 'PINE-9361', 'late': 'RIVER-2058'}

def prompt(n):
    selected = rows[:n].copy()
    for fraction, (key, value) in zip((0.1, 0.5, 0.9), expected.items()):
        selected[int(n * fraction)] += f'\nARCHIVE_NOTE anchor={key} code={value}\n'
    return ('Read this synthetic archive. Remember the three ARCHIVE_NOTE codes.\n'
            + '\n'.join(selected)
            + '\nReturn only a JSON object mapping early, middle, and late to their '
              'ARCHIVE_NOTE code values. Ignore ordinary record checksums.')

def messages(text):
    return [{'role': 'user', 'content': text}]

def count(n):
    result = post('/v1/messages/count_tokens', {'model': MODEL, 'messages': messages(prompt(n))})
    return result['input_tokens']

initial = count(1000)
n = min(len(rows), int(args.target_tokens / initial * 1000))
tokens = count(n)
if tokens > args.target_tokens * 1.03:
    n = int(n * args.target_tokens / tokens)
    tokens = count(n)
assert args.target_tokens * 0.9 < tokens < args.target_tokens * 1.03, tokens
text = prompt(n)
print(json.dumps({'phase': 'long-context-request', 'counted_tokens': tokens,
                  'records': n, 'memory_before': mem()}), flush=True)
(ROOT / f'{args.label}-synthetic-request.json').write_text(json.dumps({'messages': messages(text),
                                                         'expected': expected}) + '\n')
started = time.time()
try:
    with Sample() as sample:
        result = request(API + '/v1', MODEL, messages(text), max_tokens=256,
                         thinking=False, temperature=0)
    result.update(gpu=sample.result(), memory_after=mem(), counted_tokens=tokens,
                  expected=expected, started=started)
    answer = result['content'].strip()
    if answer.startswith('```'):
        answer = answer.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        result['retrieval_passed'] = json.loads(answer) == expected
    except (ValueError, TypeError):
        result['retrieval_passed'] = False
    (ROOT / f'{args.label}-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
except Exception as error:
    failure = {'error': str(error), 'type': type(error).__name__,
               'elapsed_seconds': time.time() - started, 'counted_tokens': tokens,
               'memory_after': mem()}
    (ROOT / f'{args.label}-failure.json').write_text(json.dumps(failure, indent=2) + '\n')
    raise
