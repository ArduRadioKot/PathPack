"""Pinned, streamed experiments. Retrieval completes before the gold-only score_run stage."""
from __future__ import annotations
import csv,gzip,hashlib,json,platform,random,subprocess,sys,time,shutil
from pathlib import Path
from dataclasses import replace
from importlib.metadata import distributions
from collections import defaultdict
from .benchmark import read_jsonl
from .config import V4Config, ABLATIONS
from .parser import scan_repository
from .retrieval import BM25, make_chunks, select_context
from .adaptive_chunker import build_chunk_index, context_header
from .hybrid_ranker import HybridIndex
from .pathpack_v4 import PreparedQuery
from .context_packer import token_count, COUNTER
from .audit import paired_bootstrap
from .evaluate import _SAFE_INSTANCE

METHODS=('bm25','graph-expand','pathpack','pathpack-v3','pathpack-v4')


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def write_csv(path,rows):
    rows=list(rows)
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def code_fingerprint():
    base=Path(__file__).parent;h=hashlib.sha256()
    for path in sorted(base.glob('*.py')):h.update(path.name.encode());h.update(path.read_bytes())
    return h.hexdigest()


def verify_snapshot(row,snapshots):
    if not _SAFE_INSTANCE.fullmatch(row['instance_id']):raise ValueError('Unsafe instance id')
    root=Path(snapshots).resolve();repo=(root/row['instance_id']).resolve()
    if not repo.is_relative_to(root) or not repo.is_dir():raise ValueError(f'Missing snapshot: {repo}')
    head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if head!=row['base_commit']:raise ValueError(f'{repo}: HEAD {head} != base_commit {row["base_commit"]}')
    dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],check=True,capture_output=True,text=True).stdout
    if dirty.strip():raise ValueError(f'Modified tracked files in {repo}; benchmark requires clean snapshots')
    return repo


def serialized_legacy(repo,graph,infos):
    result=[]
    for c in make_chunks(repo,graph):
        node=graph.nodes[c.id];lines=(repo/c.path).read_text(encoding='utf-8-sig').splitlines(keepends=True)
        spans=tuple((ln,0,len(lines[ln-1])) for ln in range(node.line_start,node.line_end+1) if ln<=len(lines))
        if not spans:continue
        body=''.join(lines[ln-1][a:b] for ln,a,b in spans)
        signature=infos[c.id].signature if c.id in infos else ''
        text=context_header(c.path,c.qualname,node.line_start,node.line_end,signature)+body+'\n\n'
        result.append(replace(c,text=text,tokens_est=token_count(text),function_id=c.id,
                              line_start=node.line_start,line_end=node.line_end,source_spans=spans,signature=signature,body=body))
    return result


def serialize_hits(hits):
    context='';selected=[];previous=0
    for hit in hits:
        context+=hit.chunk.text;total=token_count(context)
        selected.append({**hit.to_dict(include_text=False),'tokens_est':total-previous});previous=total
    return context,selected


def diagnostic_stem(instance_id,method,budget):
    if method=='pathpack-v4':tag='full'
    elif method.startswith('pathpack-v4/'):tag=method.split('/',1)[1]
    elif method=='pathpack-v10':tag='v10joint'
    elif method.startswith('pathpack-v10/'):tag='v10_'+method.split('/',1)[1]
    elif method=='pathpack-v9':tag='v9full'
    elif method.startswith('pathpack-v9/'):tag='v9_'+method.split('/',1)[1]
    elif method=='pathpack-v8':tag='v8full'
    elif method.startswith('pathpack-v8/'):tag='v8_'+method.split('/',1)[1]
    elif method=='pathpack-v7':tag='v7full'
    elif method.startswith('pathpack-v7/'):tag='v7_'+method.split('/',1)[1]
    elif method=='pathpack-v6':tag='v6full'
    elif method.startswith('pathpack-v6/'):tag='v6_'+method.split('/',1)[1]
    elif method=='pathpack-v51':tag='v51full'
    elif method.startswith('pathpack-v51/'):tag='v51_'+method.split('/',1)[1]
    elif method=='pathpack-v5':tag='v5full'
    elif method.startswith('pathpack-v5/'):tag='v5_'+method.split('/',1)[1]
    else:raise ValueError(method)
    return f'{instance_id}_{tag}_{budget}'


def run_benchmark(issues_path,snapshots,out,*,split='dev',methods=METHODS,budgets=(1024,2048,4096),
                  config=None,include_ablations=False,limit=None,v5_config=None,include_v5_ablations=False,
                  v51_config=None,include_v51_ablations=False,v6_config=None,include_v6_ablations=False,
                  v7_config=None,include_v7_ablations=False,                  v8_config=None,include_v8_ablations=False,v9_config=None,include_v9_ablations=False,v10_config=None,include_v10_ablations=False):
    from .config import V5Config, V5_ABLATIONS, V51_ABLATIONS, V6Config, V6_ABLATIONS, V7Config, V7_ABLATIONS, V8Config, V8_ABLATIONS, V9Config, V9_ABLATIONS, V10Config, V10_ABLATIONS
    from .embedding_retriever import probe_embedding
    cfg=config or V4Config();out=Path(out).resolve();methods=tuple(methods)
    if any(b<=0 for b in budgets):raise ValueError('Budgets must be positive')
    allowed=set(METHODS)|{'pathpack-v5','pathpack-v51','pathpack-v6','pathpack-v7','pathpack-v8','pathpack-v9','pathpack-v10','embedding'}
    if any(m not in allowed for m in methods):raise ValueError('Unknown method')
    embedding_note=probe_embedding();embedding_note['requested']='embedding' in methods
    if embedding_note['status']!='available':
        embedding_note['status']='not_run'
        methods=tuple(m for m in methods if m!='embedding')
    v5_variants=list(V5_ABLATIONS) if include_v5_ablations else (['full'] if 'pathpack-v5' in methods else [])
    v5_cfg=v5_config or (V5Config() if v5_variants else None)
    v51_variants=list(V51_ABLATIONS) if include_v51_ablations else (['full'] if 'pathpack-v51' in methods else [])
    v51_cfg=v51_config or (V5Config() if v51_variants else None)
    v6_variants=list(V6_ABLATIONS) if include_v6_ablations else (['full'] if 'pathpack-v6' in methods else [])
    v6_cfg=v6_config or (V6Config() if v6_variants else None)
    v7_variants=list(V7_ABLATIONS) if include_v7_ablations else (['full'] if 'pathpack-v7' in methods else [])
    v7_cfg=v7_config or (V7Config() if v7_variants else None)
    v8_variants=list(V8_ABLATIONS) if include_v8_ablations else (['full'] if 'pathpack-v8' in methods else [])
    v8_cfg=v8_config or (V8Config() if v8_variants else None)
    v9_variants=list(V9_ABLATIONS) if include_v9_ablations else (['full'] if 'pathpack-v9' in methods else [])
    v9_cfg=v9_config or (V9Config() if v9_variants else None)
    v10_variants=list(V10_ABLATIONS) if include_v10_ablations else (['joint'] if 'pathpack-v10' in methods else [])
    v10_cfg=v10_config or (V10Config() if v10_variants else None)
    issues=[r for r in read_jsonl(issues_path) if r['split']==split]
    if len({r['instance_id'] for r in issues})!=len(issues):raise ValueError('Duplicate issues')
    if any(any(k == 'patch' or k.startswith('gold_') for k in r) for r in issues):raise ValueError('Retrieval manifest must not contain gold')
    if limit:issues=issues[:limit]
    if not issues:raise ValueError('No issues selected')
    if (out/'run.json').exists():raise ValueError(f'Run exists: {out}; use a fresh output directory to preserve results')
    out.mkdir(parents=True,exist_ok=True)
    source_snapshot=out/'code_snapshot';source_snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,source_snapshot/path.name)
    try:commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    except subprocess.CalledProcessError:commit=None
    meta={'schema_version':'1.0' if v10_variants else '0.9' if v9_variants else '0.8' if v8_variants else '0.7' if v7_variants else '0.6' if v6_variants else '0.5.1' if v51_variants else '0.5' if v5_variants else '0.4','code_version':'1.0.0' if v10_variants else '0.9.0' if v9_variants else '0.8.0' if v8_variants else '0.7.0' if v7_variants else '0.6.0','code_sha256':code_fingerprint(),'git_commit':commit,
          'config':cfg.to_dict(),'v5_config':None if v5_cfg is None else v5_cfg.to_dict(),
          'v51_config':None if v51_cfg is None else v51_cfg.to_dict(),
          'v6_config':None if v6_cfg is None else v6_cfg.to_dict(),
          'v7_config':None if v7_cfg is None else v7_cfg.to_dict(),
          'v8_config':None if v8_cfg is None else v8_cfg.to_dict(),
          'v9_config':None if v9_cfg is None else v9_cfg.to_dict(),
          'v10_config':None if v10_cfg is None else v10_cfg.to_dict(),'split':split,'seed':42,'budgets':sorted(set(budgets)),'methods':list(methods),
          'ablations':list(ABLATIONS) if include_ablations else [],'v5_ablations':v5_variants,'v51_ablations':v51_variants,'v6_ablations':v6_variants,'v7_ablations':v7_variants,'v8_ablations':v8_variants,'v9_ablations':v9_variants,'v10_ablations':v10_variants,'issues':issues,'snapshots':str(Path(snapshots).resolve()),
          'token_counter':COUNTER,'hardware':{'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor()},
          'python':sys.version,'dependencies':{d.metadata['Name']:d.version for d in distributions()},
          'function_coverage_threshold':0.20,'embedding_baseline':embedding_note,
          'status':'retrieving','latency_definition':'prebuilt indexes; query scoring + ordering + prefix packing; diagnostic serialization excluded',
          'comparison_note':'Legacy methods, v0.4 and v0.5 share serialized headers and the UTF-8/4 counter. v0.5 file/chunk/graph/context components are logged separately. Embedding is not substituted when the pinned model is absent.'}
    write_json(out/'run.json',meta);inventories=[];all_predictions=[]
    for number,row in enumerate(issues,1):
        iid=row['instance_id'];repo=verify_snapshot(row,snapshots);t=time.perf_counter();graph=scan_repository(repo);scan_sec=time.perf_counter()-t
        with gzip.open(out/f'{iid}_graph.json.gz','wt') as f:json.dump(graph.to_dict(),f)
        t=time.perf_counter();adaptive=build_chunk_index(repo,graph,cfg);index=HybridIndex(adaptive,graph);adaptive_sec=time.perf_counter()-t
        legacy=[];lexindex=None
        legacy_methods=[m for m in methods if m not in {'pathpack-v4','pathpack-v5','pathpack-v51','pathpack-v6','pathpack-v7','pathpack-v8','pathpack-v9','pathpack-v10'}]
        if legacy_methods:
            t=time.perf_counter();legacy=serialized_legacy(repo,graph,adaptive.functions);lexindex=BM25(legacy);legacy_sec=time.perf_counter()-t
        else:legacy_sec=0.0
        inventories.append({'instance_id':iid,'scan_sec':scan_sec,'legacy_index_sec':legacy_sec,'adaptive_index_sec':adaptive_sec,
                            'nodes':len(graph.nodes),'edges':len(graph.edges),'chunks':len(adaptive.chunks),'functions':len(adaptive.functions),'parse_errors':len(adaptive.errors)})
        predictions=[]
        if lexindex:
            t=time.perf_counter();lexical=lexindex.scores(row['problem_statement']);query_sec=time.perf_counter()-t
            for method in legacy_methods:
                for budget in sorted(set(budgets)):
                    t=time.perf_counter();hits=select_context(legacy,graph,row['problem_statement'],method=method,budget=budget,lexical_scores=lexical)
                    context,selected=serialize_hits(hits);elapsed=query_sec+time.perf_counter()-t
                    predictions.append({'instance_id':iid,'split':split,'method':method,'budget_est':budget,'tokens_est':token_count(context),
                                        'token_counter':COUNTER,'latency_sec':elapsed,'selected':selected,'context':context})
        variants=list(ABLATIONS) if include_ablations else (['full'] if 'pathpack-v4' in methods else [])
        no_chunks=None
        for variant in variants:
            variant_cfg=cfg.ablation(variant);variant_index=index
            if not variant_cfg.enable_chunking:
                t=time.perf_counter();no_chunks=HybridIndex(build_chunk_index(repo,graph,variant_cfg),graph);extra_index_sec=time.perf_counter()-t
                variant_index=no_chunks;inventories[-1]['no_chunk_index_sec']=extra_index_sec
            t=time.perf_counter();prepared=PreparedQuery(variant_index,row['problem_statement'],variant_cfg);rank_sec=time.perf_counter()-t
            method='pathpack-v4' if variant=='full' else f'pathpack-v4/{variant}'
            for budget in sorted(set(budgets)):
                t=time.perf_counter();result=prepared.pack(budget,instance_id=iid,include_candidates=False);elapsed=rank_sec+time.perf_counter()-t
                _,selected=serialize_hits(result.hits)
                prediction={'instance_id':iid,'split':split,'method':method,'budget_est':budget,'tokens_est':token_count(result.context),
                            'token_counter':COUNTER,'latency_sec':elapsed,'rank_sec':rank_sec,'selected':selected,'context':result.context}
                predictions.append(prediction)
                diagnostic=prepared.pack(budget,instance_id=iid).diagnostics;diagnostic['method']=method
                dest=out/'diagnostics'/f'{iid}_{variant}_{budget}.json.gz';dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt',compresslevel=1) as f:json.dump(diagnostic,f,ensure_ascii=False)
                if budget==max(budgets):
                    with gzip.open(dest.with_suffix('').with_suffix('.csv.gz'),'wt',newline='',compresslevel=1) as f:
                        fields=['id','path','function_id','line_start','line_end','tokens_est','bm25_score','lexical_score','symbol_score','graph_score','diversity_contribution','score','utility','order','selected','decision']
                        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(diagnostic['candidates'])
            del prepared
        if v5_variants:
            from .hierarchical_retriever import HierarchicalIndex
            from .pathpack_v5 import PreparedQueryV5
            chunk_cfg=v5_cfg.to_v4()
            same_chunks=(cfg.target_chunk_tokens,cfg.max_chunk_tokens,cfg.min_chunk_tokens,cfg.chunk_overlap_lines,cfg.enable_chunking)==(
                chunk_cfg.target_chunk_tokens,chunk_cfg.max_chunk_tokens,chunk_cfg.min_chunk_tokens,chunk_cfg.chunk_overlap_lines,chunk_cfg.enable_chunking)
            t=time.perf_counter()
            v5_chunks=adaptive if same_chunks else build_chunk_index(repo,graph,chunk_cfg)
            v5_index=HierarchicalIndex(v5_chunks,graph,repo)
            inventories[-1]['v5_index_sec']=time.perf_counter()-t
            whole_index=None
            for variant in v5_variants:
                variant_cfg=v5_cfg.ablation(variant);variant_index=v5_index
                if not variant_cfg.enable_chunking:
                    if whole_index is None:
                        t=time.perf_counter();whole_index=HierarchicalIndex(build_chunk_index(repo,graph,variant_cfg.to_v4()),graph,repo)
                        inventories[-1]['v5_no_chunk_index_sec']=time.perf_counter()-t
                    variant_index=whole_index
                t=time.perf_counter();prepared=PreparedQueryV5(variant_index,row['problem_statement'],variant_cfg);rank_sec=time.perf_counter()-t
                method_name='pathpack-v5' if variant=='full' else f'pathpack-v5/{variant}'
                for budget in sorted(set(budgets)):
                    t=time.perf_counter();result=prepared.pack(budget,instance_id=iid,include_candidates=False);elapsed=rank_sec+time.perf_counter()-t
                    _,selected=serialize_hits(result.hits)
                    predictions.append({'instance_id':iid,'split':split,'method':method_name,'budget_est':budget,'tokens_est':token_count(result.context),
                                        'token_counter':COUNTER,'latency_sec':elapsed,'rank_sec':rank_sec,'profile':result.diagnostics['profile'],
                                        'selected':selected,'context':result.context})
                    diagnostic=prepared.pack(budget,instance_id=iid).diagnostics;diagnostic['method']=method_name
                    dest=out/'diagnostics'/f'{diagnostic_stem(iid,method_name,budget)}.json.gz';dest.parent.mkdir(exist_ok=True)
                    with gzip.open(dest,'wt',compresslevel=1) as f:json.dump(diagnostic,f,ensure_ascii=False)
                    if budget==max(budgets):
                        with gzip.open(dest.with_suffix('').with_suffix('.csv.gz'),'wt',newline='',compresslevel=1) as f:
                            fields=['id','path','function_id','line_start','line_end','tokens_est','file_score','chunk_score','graph_score','context_score','lexical_score','symbol_score','local_lexical','diversity_contribution','file_penalty','formula_score','score','utility','order','selected','decision']
                            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(diagnostic['candidates'])
                del prepared
            del v5_index,whole_index
        if v51_variants:
            from dataclasses import replace
            from .config import v51_variant
            from .hierarchical_retriever import HierarchicalIndex
            from .pathpack_v5 import PreparedQueryV5
            fine_v4=v51_cfg.to_v4()
            coarse_v4=replace(v51_cfg,enable_chunking=False).to_v4()
            same_fine=(cfg.target_chunk_tokens,cfg.max_chunk_tokens,cfg.min_chunk_tokens,cfg.chunk_overlap_lines,cfg.enable_chunking)==(
                fine_v4.target_chunk_tokens,fine_v4.max_chunk_tokens,fine_v4.min_chunk_tokens,fine_v4.chunk_overlap_lines,True)
            t=time.perf_counter()
            fine_chunks=adaptive if same_fine and fine_v4.enable_chunking else build_chunk_index(repo,graph,fine_v4)
            coarse_chunks=build_chunk_index(repo,graph,coarse_v4)
            fine_index=HierarchicalIndex(fine_chunks,graph,repo,file_chunk_index=coarse_chunks)
            coarse_index=HierarchicalIndex(coarse_chunks,graph,repo)
            inventories[-1]['v51_fine_index_sec']=time.perf_counter()-t
            inventories[-1]['v51_coarse_chunks']=len(coarse_chunks.chunks)
            for variant in v51_variants:
                variant_cfg=v51_variant(v51_cfg,variant)
                variant_index=coarse_index if not variant_cfg.enable_chunking else fine_index
                t=time.perf_counter();prepared=PreparedQueryV5(variant_index,row['problem_statement'],variant_cfg);rank_sec=time.perf_counter()-t
                method_name='pathpack-v51' if variant=='full' else f'pathpack-v51/{variant}'
                for budget in sorted(set(budgets)):
                    t=time.perf_counter();result=prepared.pack(budget,instance_id=iid,include_candidates=False);elapsed=rank_sec+time.perf_counter()-t
                    _,selected=serialize_hits(result.hits)
                    predictions.append({'instance_id':iid,'split':split,'method':method_name,'budget_est':budget,'tokens_est':token_count(result.context),
                                        'token_counter':COUNTER,'latency_sec':elapsed,'rank_sec':rank_sec,'profile':result.diagnostics['profile'],
                                        'selected':selected,'context':result.context})
                    diagnostic=prepared.pack(budget,instance_id=iid).diagnostics;diagnostic['method']=method_name
                    dest=out/'diagnostics'/f'{diagnostic_stem(iid,method_name,budget)}.json.gz';dest.parent.mkdir(exist_ok=True)
                    with gzip.open(dest,'wt',compresslevel=1) as f:json.dump(diagnostic,f,ensure_ascii=False)
                    if budget==max(budgets):
                        with gzip.open(dest.with_suffix('').with_suffix('.csv.gz'),'wt',newline='',compresslevel=1) as f:
                            fields=['id','path','function_id','line_start','line_end','tokens_est','file_score','chunk_score','graph_score','context_score','lexical_score','symbol_score','local_lexical','diversity_contribution','file_penalty','formula_score','score','utility','order','selected','decision']
                            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(diagnostic['candidates'])
                del prepared
            del fine_index,coarse_index
        if v6_variants:
            from .pathpack_v6 import PreparedQueryV6
            from .config import v6_variant
            t=time.perf_counter()
            fine_chunks=build_chunk_index(repo,graph,v6_cfg.to_v4())
            coarse_chunks=fine_chunks if not v6_cfg.enable_chunking else build_chunk_index(repo,graph,replace(v6_cfg.to_v4(),enable_chunking=False))
            inventories[-1]['v6_index_sec']=time.perf_counter()-t
            inventories[-1]['v6_fine_chunks']=len(fine_chunks.chunks)
            for variant in v6_variants:
                variant_cfg=v6_variant(v6_cfg,variant)
                variant_fine=coarse_chunks if not variant_cfg.enable_chunking else fine_chunks
                t=time.perf_counter();prepared=PreparedQueryV6(variant_fine,coarse_chunks,graph,row['problem_statement'],variant_cfg);rank_sec=time.perf_counter()-t
                method_name='pathpack-v6' if variant=='full' else f'pathpack-v6/{variant}'
                for budget in sorted(set(budgets)):
                    t=time.perf_counter();result=prepared.pack(budget,instance_id=iid,include_candidates=False);elapsed=rank_sec+time.perf_counter()-t
                    _,selected=serialize_hits(result.hits)
                    predictions.append({'instance_id':iid,'split':split,'method':method_name,'budget_est':budget,'tokens_est':token_count(result.context),
                                        'token_counter':COUNTER,'latency_sec':elapsed,'rank_sec':rank_sec,'profile':result.diagnostics['profile'],
                                        'selected':selected,'context':result.context})
                    diagnostic=prepared.pack(budget,instance_id=iid).diagnostics;diagnostic['method']=method_name
                    dest=out/'diagnostics'/f'{diagnostic_stem(iid,method_name,budget)}.json.gz';dest.parent.mkdir(exist_ok=True)
                    with gzip.open(dest,'wt',compresslevel=1) as f:json.dump(diagnostic,f,ensure_ascii=False)
                    if budget==max(budgets):
                        with gzip.open(dest.with_suffix('').with_suffix('.csv.gz'),'wt',newline='',compresslevel=1) as f:
                            fields=['id','path','function_id','line_start','line_end','tokens_est','lexical_score','symbol_score','parent_score','graph_score','redundancy','same_function_penalty','evidence','score','utility','order','selected','decision']
                            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(diagnostic['candidates'])
                del prepared
            del fine_chunks,coarse_chunks
        if v7_variants:
            from .pathpack_v7 import PreparedQueryV7
            from .config import v7_variant
            for variant in v7_variants:
                variant_cfg=v7_variant(v7_cfg,variant)
                t=time.perf_counter();prepared=PreparedQueryV7(index,row['problem_statement'],variant_cfg,cfg);rank_sec=time.perf_counter()-t
                method_name='pathpack-v7' if variant=='full' else f'pathpack-v7/{variant}'
                for budget in sorted(set(budgets)):
                    t=time.perf_counter();result=prepared.pack(budget,instance_id=iid,include_candidates=False);elapsed=rank_sec+time.perf_counter()-t
                    _,selected=serialize_hits(result.hits)
                    predictions.append({'instance_id':iid,'split':split,'method':method_name,'budget_est':budget,'tokens_est':token_count(result.context),
                                        'token_counter':COUNTER,'latency_sec':elapsed,'rank_sec':rank_sec,'profile':result.diagnostics['profile'],
                                        'selected':selected,'context':result.context})
                    diagnostic=prepared.pack(budget,instance_id=iid).diagnostics;diagnostic['method']=method_name
                    dest=out/'diagnostics'/f'{diagnostic_stem(iid,method_name,budget)}.json.gz';dest.parent.mkdir(exist_ok=True)
                    with gzip.open(dest,'wt',compresslevel=1) as f:json.dump(diagnostic,f,ensure_ascii=False)
                del prepared
        if v8_variants:
            from .pathpack_v8 import PreparedQueryV8
            from .config import v8_variant
            for variant in v8_variants:
                variant_cfg=v8_variant(v8_cfg,variant)
                t=time.perf_counter();prepared=PreparedQueryV8(index,row['problem_statement'],variant_cfg,cfg);rank_sec=time.perf_counter()-t
                method_name='pathpack-v8' if variant=='full' else f'pathpack-v8/{variant}'
                for budget in sorted(set(budgets)):
                    t=time.perf_counter();result=prepared.pack(budget,instance_id=iid,include_candidates=False);elapsed=rank_sec+time.perf_counter()-t
                    _,selected=serialize_hits(result.hits)
                    predictions.append({'instance_id':iid,'split':split,'method':method_name,'budget_est':budget,'tokens_est':token_count(result.context),
                                        'token_counter':COUNTER,'latency_sec':elapsed,'rank_sec':rank_sec,'profile':result.diagnostics['profile'],
                                        'selected':selected,'context':result.context})
                    diagnostic=prepared.pack(budget,instance_id=iid).diagnostics;diagnostic['method']=method_name
                    dest=out/'diagnostics'/f'{diagnostic_stem(iid,method_name,budget)}.json.gz';dest.parent.mkdir(exist_ok=True)
                    with gzip.open(dest,'wt',compresslevel=1) as f:json.dump(diagnostic,f,ensure_ascii=False)
                del prepared
        if v9_variants:
            from .pathpack_v9 import PreparedQueryV9
            from .config import v9_variant
            for variant in v9_variants:
                variant_cfg=v9_variant(v9_cfg,variant)
                t=time.perf_counter();prepared=PreparedQueryV9(index,row['problem_statement'],variant_cfg,cfg);rank_sec=time.perf_counter()-t
                method_name='pathpack-v9' if variant=='full' else f'pathpack-v9/{variant}'
                for budget in sorted(set(budgets)):
                    t=time.perf_counter();result=prepared.pack(budget,instance_id=iid,include_candidates=False);elapsed=rank_sec+time.perf_counter()-t
                    _,selected=serialize_hits(result.hits)
                    predictions.append({'instance_id':iid,'split':split,'method':method_name,'budget_est':budget,'tokens_est':token_count(result.context),
                                        'token_counter':COUNTER,'latency_sec':elapsed,'rank_sec':rank_sec,'profile':result.diagnostics['profile'],
                                        'selected':selected,'context':result.context})
                    diagnostic=prepared.pack(budget,instance_id=iid).diagnostics;diagnostic['method']=method_name
                    dest=out/'diagnostics'/f'{diagnostic_stem(iid,method_name,budget)}.json.gz';dest.parent.mkdir(exist_ok=True)
                    with gzip.open(dest,'wt',compresslevel=1) as f:json.dump(diagnostic,f,ensure_ascii=False)
                del prepared
        if v10_variants:
            from .pathpack_v10 import PreparedQueryV10
            from .config import v10_variant
            for variant in v10_variants:
                variant_cfg=v10_variant(v10_cfg,variant)
                t=time.perf_counter();prepared=PreparedQueryV10(index,row['problem_statement'],variant_cfg,repo);rank_sec=time.perf_counter()-t
                method_name='pathpack-v10' if variant=='joint' else f'pathpack-v10/{variant}'
                for budget in sorted(set(budgets)):
                    t=time.perf_counter();result=prepared.pack(budget,instance_id=iid,include_candidates=False);elapsed=rank_sec+time.perf_counter()-t
                    _,selected=serialize_hits(result.hits)
                    predictions.append({'instance_id':iid,'split':split,'method':method_name,'budget_est':budget,'tokens_est':token_count(result.context),
                                        'token_counter':COUNTER,'latency_sec':elapsed,'rank_sec':rank_sec,'profile':result.diagnostics['profile'],
                                        'fallback_used':result.diagnostics.get('fallback_used',False),'selected':selected,'context':result.context})
                    diagnostic=prepared.pack(budget,instance_id=iid).diagnostics;diagnostic['method']=method_name
                    dest=out/'diagnostics'/f'{diagnostic_stem(iid,method_name,budget)}.json.gz';dest.parent.mkdir(exist_ok=True)
                    with gzip.open(dest,'wt',compresslevel=1) as f:json.dump(diagnostic,f,ensure_ascii=False)
                del prepared
        write_jsonl(out/f'{iid}_predictions.jsonl',predictions);all_predictions.extend(predictions)
        write_csv(out/'graph_inventory.csv',inventories)
        print(f'[{number}/{len(issues)}] {iid}: graph={scan_sec:.2f}s, adaptive chunks={len(adaptive.chunks)}, predictions={len(predictions)}',flush=True)
        del index,adaptive,graph,no_chunks
    write_jsonl(out/'predictions.jsonl',all_predictions)
    from .v05_ablation import mean_profile, profile_table
    profiles=profile_table(all_predictions)
    write_csv(out/'stage_profile.csv',profiles);write_csv(out/'stage_profile_mean.csv',mean_profile(profiles))
    write_csv(out/'budget_summary.csv',[{k:r[k] for k in ('instance_id','method','budget_est','tokens_est','latency_sec')}|
               {'remaining_tokens':r['budget_est']-r['tokens_est'],'utilization':r['tokens_est']/r['budget_est'],
                'selected_files':len({c['path'] for c in r['selected']}),'selected_functions':len({c['function_id'] for c in r['selected'] if c['kind']!='module'}),'selected_chunks':len(r['selected'])} for r in all_predictions])
    meta['status']='retrieval_complete';meta['prediction_count']=len(all_predictions);write_json(out/'run.json',meta)
    return {'run':str(out),'tasks':len(issues),'predictions':len(all_predictions)}


def load_scoring_dataset(raw=None):
    if raw:return read_jsonl(raw)
    # Only evaluation uses the cache with patches. Never called by run_benchmark.
    import os
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('HF_DATASETS_CACHE',str(Path('output/benchmark/hf_cache').resolve()))
    from .benchmark import load_verified
    return load_verified()


def score_run(run,*,raw=None):
    from .evaluation_v4 import score_prediction,diagnose_failures
    from .line_metrics import build_line_label, extra_metrics
    from .v05_ablation import comparisons as v5_comparisons
    from .graph import Edge
    from types import SimpleNamespace
    run=Path(run);meta=json.loads((run/'run.json').read_text())
    if meta['status'] not in {'retrieval_complete','scored'}:raise ValueError('Retrieval must finish before scoring')
    scoring_snapshot=run/'scoring_code_snapshot';scoring_snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,scoring_snapshot/path.name)
    source={r['instance_id']:r for r in load_scoring_dataset(raw)};scores=[];labels=[];diagnostics=[];pool_rows=[]
    predictions=read_jsonl(run/'predictions.jsonl');by_id=defaultdict(list)
    for p in predictions:by_id[p['instance_id']].append(p)
    for issue in meta['issues']:
        iid=issue['instance_id'];repo=verify_snapshot(issue,meta['snapshots'])
        gold=build_line_label(source[iid]['patch'],repo)
        labels.append({'instance_id':iid,**{k:v for k,v in gold.items() if k!='source_lines'}})
        with gzip.open(run/f'{iid}_graph.json.gz','rt') as f:graph_data=json.load(f)
        graph=SimpleNamespace(edges=[Edge(**e) for e in graph_data['edges']])
        for p in by_id[iid]:
            s=score_prediction(p,gold,function_coverage_threshold=meta['function_coverage_threshold']);s.update(extra_metrics(p['selected'],gold));scores.append(s)
            if p['method'].startswith(('pathpack-v4','pathpack-v5','pathpack-v6','pathpack-v7','pathpack-v8','pathpack-v9','pathpack-v10')):
                with gzip.open(run/'diagnostics'/f'{diagnostic_stem(iid,p["method"],p["budget_est"])}.json.gz','rt') as f:candidates=json.load(f)['candidates']
                diagnostics.append({'instance_id':iid,'method':p['method'],'budget':p['budget_est'],
                                    'flags':diagnose_failures(p,gold,s,candidates,graph),'missed_functions':sorted(set(gold['functions'])-set(s['found_functions']))})
                if str(p['method']).startswith('pathpack-v10'):
                    files=set(gold['gold_python_files'])
                    functions={fn.get('graph_id', fn['id']) for fn in gold['functions'].values()}
                    pool_files={c.get('path') for c in candidates}
                    pool_fns={c.get('function_id') for c in candidates if c.get('function_id')}
                    packed_files={c['path'] for c in p['selected']}
                    packed_fns={c.get('function_id') for c in p['selected'] if c.get('function_id')}
                    pool_rows.append({'instance_id':iid,'method':p['method'],'budget':p['budget_est'],
                                      'pool_chunks':len(candidates),'gold_files':len(files),'gold_functions':len(functions),
                                      'file_pool_recall': None if not files else len(files & pool_files)/len(files),
                                      'function_pool_recall': None if not functions else len(functions & pool_fns)/len(functions),
                                      'file_packed_recall': None if not files else len(files & packed_files)/len(files),
                                      'function_packed_recall': None if not functions else len(functions & packed_fns)/len(functions),
                                      'files_in_pool_not_packed': len((files & pool_files) - packed_files),
                                      'functions_in_pool_not_packed': len((functions & pool_fns) - packed_fns),
                                      'files_absent_from_pool': len(files - pool_files),
                                      'functions_absent_from_pool': len(functions - pool_fns)})
    write_csv(run/'pool_recall.csv',pool_rows)
    write_jsonl(run/'scores.jsonl',scores);write_jsonl(run/'scoring_labels.jsonl',labels);write_jsonl(run/'retrieval_failures.jsonl',diagnostics)
    groups=defaultdict(list)
    for s in scores:groups[(s['method'],s['budget'])].append(s)
    summary=[]
    metrics=('file_recall','function_recall','complete_function_recall','changed_line_recall','relevant_line_density','insertion_anchor_recall','base_touch_recall','latency_sec','token_utilization','tokens_est','useful_functions_per_1k')
    for (method,budget),rows in sorted(groups.items()):
        item={'method':method,'budget':budget,'tasks':len(rows)}
        for metric in metrics:
            values=[r[metric] for r in rows if r[metric] is not None]
            item[metric]=sum(values)/len(values) if values else None
            if metric.endswith('recall') or metric=='relevant_line_density':item[metric+'_evaluable']=len(values)
        item['complete_label_tasks']=sum(r['label_completeness']=='complete' for r in rows)
        item['partial_label_tasks']=len(rows)-item['complete_label_tasks'];summary.append(item)
    write_csv(run/'summary.csv',summary);comparisons=[];losing=[]
    for budget in meta['budgets']:
        full={r['instance_id']:r for r in groups.get(('pathpack-v4',budget),[])}
        for (method,b),rows in sorted(groups.items()):
            if b!=budget or method=='pathpack-v4':continue
            for metric in ('file_recall','function_recall','changed_line_recall'):
                a={i:r[metric] for i,r in full.items() if r[metric] is not None}
                other={r['instance_id']:r[metric] for r in rows if r[metric] is not None}
                comparisons.append({'candidate':'pathpack-v4','baseline':method,'budget':budget,'metric':metric,**paired_bootstrap(a,other,iterations=2000,seed=42)})
            if method in {'bm25','pathpack','pathpack-v3'}:
                for row in rows:
                    iid=row['instance_id'];candidate=full.get(iid)
                    if candidate is None:continue
                    for metric in ('file_recall','function_recall'):
                        if candidate[metric] is not None and row[metric] is not None and candidate[metric]<row[metric]:
                            losing.append({'instance_id':iid,'budget':budget,'baseline':method,'metric':metric,'v4':candidate[metric],'baseline_value':row[metric],
                                           'diagnostics':next((d['flags'] for d in diagnostics if d['instance_id']==iid and d['method']=='pathpack-v4' and d['budget']==budget),[])})
    comparisons.extend(v5_comparisons(groups,meta['budgets'],paired_bootstrap))
    write_json(run/'comparisons.json',comparisons);write_csv(run/'comparisons.csv',[{**r,'ci95':json.dumps(r['ci95'])} for r in comparisons])
    write_json(run/'losing_cases.json',losing);write_csv(run/'losing_cases.csv',losing)
    meta['status']='scored';meta['scoring_code_sha256']=code_fingerprint();meta['scoring_source_snapshot']='scoring_code_snapshot';meta['scoring_note']='Changed Line Recall is nonempty removed/replaced base lines only. Insertion-anchor recall and base-touch recall are separate and are not mixed into it. Relevant line density is changed-line hits over emitted nonempty lines, null when the task has no evaluable changed lines or the context is empty. Function recall still requires 20% owned body lines; a function id or signature is not coverage.'
    write_json(run/'run.json',meta)
    audit=audit_run(run);return {'run':str(run),'groups':len(summary),'audit':audit['ok'],'losing_cases':len(losing)}


def audit_run(run):
    run=Path(run);predictions=read_jsonl(run/'predictions.jsonl');groups=defaultdict(list);violations=[];non_nested=[]
    seen=set();loaded_iid=None;symbols={}
    for p in predictions:
        if p['instance_id']!=loaded_iid:
            with gzip.open(run/f'{p["instance_id"]}_graph.json.gz','rt') as f:
                symbols={n['id']:n for n in json.load(f)['nodes']}
            loaded_iid=p['instance_id']
        key=(p['instance_id'],p['method'],p['budget_est'])
        if key in seen:violations.append({'type':'duplicate_prediction','key':key})
        seen.add(key)
        used=token_count(p['context'])
        if used!=p['tokens_est'] or used>p['budget_est'] or sum(c['tokens_est'] for c in p['selected'])!=used:violations.append({'type':'budget_or_accounting','key':key})
        ids=[c['id'] for c in p['selected']]
        if len(ids)!=len(set(ids)):violations.append({'type':'duplicate_chunk','key':key})
        if p['method'].startswith(('pathpack-v4','pathpack-v5','pathpack-v6','pathpack-v7','pathpack-v8','pathpack-v9','pathpack-v10')):
            intervals=defaultdict(list)
            for chunk in p['selected']:
                owner=symbols.get(chunk.get('function_id'))
                if owner is None or owner['path']!=chunk['path']:
                    violations.append({'type':'missing_dependency','key':key,'chunk':chunk['id']})
                for line,start,end in chunk.get('source_spans',[]):
                    if line<1 or start<0 or end<=start:
                        violations.append({'type':'invalid_source_span','key':key,'chunk':chunk['id']})
                    source_key=(chunk['path'],line)
                    if any(max(start,a)<min(end,b) for a,b in intervals[source_key]):
                        violations.append({'type':'duplicate_source_interval','key':key,'source':source_key})
                    intervals[source_key].append((start,end))
        groups[(p['instance_id'],p['method'])].append(p)
    for key,rows in groups.items():
        rows.sort(key=lambda r:r['budget_est'])
        for a,b in zip(rows,rows[1:]):
            if not b['context'].startswith(a['context']):
                record={'instance_id':key[0],'method':key[1],'small':a['budget_est'],'large':b['budget_est']}
                if key[1].endswith('/nested') or key[1].startswith(('pathpack-v4','pathpack-v5','pathpack-v6','pathpack-v7','pathpack-v8','pathpack-v9')):violations.append({'type':'non_prefix',**record})
                else:non_nested.append(record)
    meta=json.loads((run/'run.json').read_text())
    ablations=meta.get('ablations') or [];v5_ablations=meta.get('v5_ablations') or [];v51_ablations=meta.get('v51_ablations') or [];v6_ablations=meta.get('v6_ablations') or []
    extra=(len(ablations)-1 if ablations and 'pathpack-v4' in meta['methods'] else len(ablations))
    extra+=(len(v5_ablations)-1 if v5_ablations and 'pathpack-v5' in meta['methods'] else len(v5_ablations))
    extra+=(len(v51_ablations)-1 if v51_ablations and 'pathpack-v51' in meta['methods'] else len(v51_ablations))
    extra+=(len(v6_ablations)-1 if v6_ablations and 'pathpack-v6' in meta['methods'] else len(v6_ablations))
    v7_ablations=meta.get('v7_ablations') or []
    extra+=(len(v7_ablations)-1 if v7_ablations and 'pathpack-v7' in meta['methods'] else len(v7_ablations))
    v8_ablations=meta.get('v8_ablations') or []
    extra+=(len(v8_ablations)-1 if v8_ablations and 'pathpack-v8' in meta['methods'] else len(v8_ablations))
    v9_ablations=meta.get('v9_ablations') or []
    extra+=(len(v9_ablations)-1 if v9_ablations and 'pathpack-v9' in meta['methods'] else len(v9_ablations))
    v10_ablations=meta.get('v10_ablations') or []
    extra+=(len(v10_ablations)-1 if v10_ablations and 'pathpack-v10' in meta['methods'] else len(v10_ablations))
    expected=len(meta['issues'])*len(meta['budgets'])*(len(meta['methods'])+extra)
    if len(predictions)!=expected:violations.append({'type':'missing_predictions','expected':expected,'actual':len(predictions)})
    report={'ok':not violations,'tasks':len(meta['issues']),'predictions':len(predictions),'violations':violations,'legacy_non_prefix':non_nested,'token_counter':COUNTER}
    write_json(run/'audit.json',report);return report


def prepare_holdout(issues_path,out,*,raw=None,count=60,seed=43):
    existing=read_jsonl(issues_path);excluded={r['instance_id'] for r in existing};rows=load_scoring_dataset(raw)
    candidates=sorted((r for r in rows if r['instance_id'] not in excluded),key=lambda r:r['instance_id'])
    if len(candidates)<count:raise ValueError('Not enough independent task ids')
    random.Random(seed).shuffle(candidates);chosen=candidates[:count];out=Path(out)
    if (out/'issues.jsonl').exists():raise ValueError('Holdout already frozen; refusing to reshuffle')
    write_jsonl(out/'issues.jsonl',[{k:r[k] for k in ('instance_id','repo','base_commit','problem_statement')}|{'split':'holdout'} for r in chosen])
    write_json(out/'metadata.json',{'seed':seed,'count':count,'excluded_ids':sorted(excluded),'instance_overlap':0,'repo_overlap':sorted({r['repo'] for r in chosen}&{r['repo'] for r in existing}),'note':'Instance-disjoint, not repository-disjoint; no retrieval, patches, or metric inspection performed. Freeze code/config before evaluation.'})
    return {'tasks':count,'path':str(out),'evaluated':False}
