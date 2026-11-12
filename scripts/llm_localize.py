"""LLM localization probe. Phase 1 (this script, `ask`): give a local Gemma the packed context of each method and ask
which function must change. No gold is read here. Phase 2 (`score`) compares the answers with the gold patch."""
import json, sys, time, urllib.request, hashlib
from pathlib import Path

MODEL = 'gemma4:e2b-mlx'
SYSTEM = ('You are a software engineer fixing a bug. You get an issue and code excerpts from the repository. '
          'Name the one function or method that most needs to be edited to fix the issue. '
          'Choose only from the excerpts. Answer with JSON: {"file": "<path>", "function": "<name>"}.')


def ask(issue, context):
    prompt = f'ISSUE:\n{issue[:6000]}\n\nCODE EXCERPTS:\n{context}\n\nWhich function must be edited? JSON only.'
    body = json.dumps({'model': MODEL, 'system': SYSTEM, 'prompt': prompt, 'stream': False, 'format': 'json', 'think': False,
                       'options': {'temperature': 0, 'seed': 0, 'num_ctx': 8192, 'num_predict': 120}}).encode()
    req = urllib.request.Request('http://localhost:11434/api/generate', body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out.get('response', ''), out.get('prompt_eval_count'), out.get('eval_count')


def cmd_ask(run, issues_path, out_path, methods, budget, limit):
    issues = {json.loads(l)['instance_id']: json.loads(l) for l in open(issues_path)}
    preds = [json.loads(l) for l in open(Path(run) / 'predictions.jsonl')]
    preds = [p for p in preds if p['budget_est'] == budget and p['method'] in methods]
    ids = sorted({p['instance_id'] for p in preds})[:limit]
    done = set()
    if Path(out_path).is_file():
        done = {(json.loads(l)['instance_id'], json.loads(l)['method']) for l in open(out_path)}
    with open(out_path, 'a') as handle:
        for n, p in enumerate(sorted((p for p in preds if p['instance_id'] in ids), key=lambda p: (p['instance_id'], p['method']))):
            if (p['instance_id'], p['method']) in done:
                continue
            t = time.time()
            try:
                text, n_in, n_out = ask(issues[p['instance_id']]['problem_statement'], p['context'])
            except Exception as exc:
                text, n_in, n_out = f'ERROR {type(exc).__name__}: {exc}', None, None
            handle.write(json.dumps({'instance_id': p['instance_id'], 'method': p['method'], 'budget': budget, 'model': MODEL,
                                     'answer': text, 'prompt_tokens': n_in, 'answer_tokens': n_out, 'sec': round(time.time() - t, 2),
                                     'context_sha': hashlib.sha256(p['context'].encode()).hexdigest()[:12]}) + '\n')
            handle.flush()
            print(n, p['instance_id'], p['method'], round(time.time() - t, 1), text[:90].replace('\n', ' '), flush=True)


def cmd_score(run, answers_path):
    sys.path.insert(0, 'src')
    from codegraph_rag.audit import paired_bootstrap
    from codegraph_rag.experiments import load_scoring_dataset, verify_snapshot
    from codegraph_rag.line_metrics import build_line_label
    meta = json.loads((Path(run) / 'run.json').read_text())
    issues = {i['instance_id']: i for i in meta['issues']}
    source = {r['instance_id']: r for r in load_scoring_dataset()}
    gold = {}
    rows = [json.loads(l) for l in open(answers_path)]
    out = []
    for r in rows:
        iid = r['instance_id']
        if iid not in gold:
            g = build_line_label(source[iid]['patch'], verify_snapshot(issues[iid], meta['snapshots']))
            gold[iid] = (set(g['gold_python_files']), {(f['path'], f['qualname'].split('.')[-1]) for f in g['functions'].values()})
        files, fns = gold[iid]
        try:
            a = json.loads(r['answer']); path = str(a.get('file', '')).strip().lstrip('./'); fn = str(a.get('function', '')).strip().split('(')[0].split('.')[-1]
            valid = True
        except Exception:
            path, fn, valid = '', '', False
        file_hit = any(path and (g == path or g.endswith('/' + path) or path.endswith('/' + g)) for g in files)
        fn_hit = None if not fns else any(fn and fn == name and (gp == path or gp.endswith('/' + path) or path.endswith('/' + gp)) for gp, name in fns)
        out.append({**r, 'valid_json': valid, 'llm_file_hit': float(file_hit), 'llm_function_hit': None if fn_hit is None else float(fn_hit)})
    Path(answers_path.replace('.jsonl', '_scored.jsonl')).write_text(''.join(json.dumps(o) + '\n' for o in out))
    methods = sorted({o['method'] for o in out})
    summary = {}
    for m in methods:
        part = [o for o in out if o['method'] == m]
        fh = [o['llm_function_hit'] for o in part if o['llm_function_hit'] is not None]
        summary[m] = {'n': len(part), 'valid_json': sum(o['valid_json'] for o in part) / len(part),
                      'llm_file_acc': sum(o['llm_file_hit'] for o in part) / len(part),
                      'llm_function_acc': sum(fh) / len(fh) if fh else None, 'n_function': len(fh),
                      'mean_sec': sum(o['sec'] for o in part) / len(part)}
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[m].items()})
    comp = {}
    for cand in methods:
        for base in methods:
            if cand == base or base != 'pathpack-v4':
                continue
            for key in ('llm_file_hit', 'llm_function_hit'):
                a = {o['instance_id']: o[key] for o in out if o['method'] == cand and o[key] is not None}
                b = {o['instance_id']: o[key] for o in out if o['method'] == base and o[key] is not None}
                s = paired_bootstrap(a, b, iterations=2000, seed=42)
                comp[f'{cand} vs {base} {key}'] = s
                print(f"{cand} - {base} {key}: {s['difference']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] {s['wins']}/{s['ties']}/{s['losses']} n={s['n']}")
    Path(answers_path.replace('.jsonl', '_summary.json')).write_text(json.dumps({'model': MODEL, 'summary': summary, 'paired': comp}, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'ask':
        cmd_ask(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[6:], int(sys.argv[5].split(':')[0]), int(sys.argv[5].split(':')[1]))
    else:
        cmd_score(sys.argv[2], sys.argv[3])
