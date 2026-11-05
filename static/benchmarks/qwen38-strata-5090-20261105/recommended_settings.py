"""Focused coding/agent checks using the publisher's recommended thinking settings."""
import argparse
import json
from pathlib import Path
from benchmark import Sample, quality_task
from tasks import dataset

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()
dest = ROOT/'results'/args.label/'recommended-settings'
dest.mkdir(parents=True, exist_ok=True)
selected = [t for t in dataset() if t['kind']=='agent' or
            t['id'] in ('sse_parser','batch_transactions','async_ordered_map')]
settings = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0,
                reasoning_effort='xhigh', preserve_thinking=True, seed=729)
(dest/'settings.json').write_text(json.dumps({
    'sampling':settings, 'output_budget_tokens':8192,
    'preserved_assistant_reasoning':True,
    'tasks':[t['id'] for t in selected],
    'limitations':'Six focused checks, one seeded sample each; not a full agentic benchmark.'},indent=2)+'\n')
rows=[]
for task in selected:
    p=dest/(task['id']+'.json')
    if p.exists():
        rows.append(json.loads(p.read_text()))
        continue
    with Sample() as sample:
        result=quality_task(task,'http://127.0.0.1:8080/v1',args.model,
                            dest/task['id'],thinking=True,max_tokens=8192,
                            request_kwargs=settings,preserve_reasoning=True)
    result['sampled_resources']=sample.result()
    p.write_text(json.dumps(result,indent=2)+'\n')
    rows.append(result)
    print(json.dumps({'task':task['id'],'passed':result['grade']['passed'],
                      'seconds':result['seconds']}),flush=True)
summary={'passed':sum(x['grade']['passed'] for x in rows),'total':len(rows),
         'failures':[x['id'] for x in rows if not x['grade']['passed']],
         'total_seconds':sum(x['seconds'] for x in rows)}
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
