"""PathPack v0.5 retrieval contracts. Gold is used only in the metric tests."""
import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from codegraph_rag.adaptive_chunker import build_chunk_index
from codegraph_rag.config import V51_ABLATIONS, V5_ABLATIONS, V5Config, v51_variant
from codegraph_rag.context_packer import token_count
from codegraph_rag.graph import CodeGraph
from codegraph_rag.hierarchical_retriever import HierarchicalIndex, apply_signature_anchor
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_v4 import retrieve_v4
from codegraph_rag.pathpack_v5 import PreparedQueryV5, retrieve_v5
from codegraph_rag.retrieval import retrieve


class HierarchicalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        filler='\n'.join(f'        step_{i} = {i}' for i in range(40))
        (self.root/'auth.py').write_text(
            'def decorate(fn):\n    return fn\n\n'
            'class Session:\n'
            '    def login(self, user):\n'
            f'{filler}\n'
            '        marker = "zxqmarker"\n'
            '        return user\n')
        (self.root/'store.py').write_text('def obtain_session(key):\n    return key\n')
        (self.root/'routes.py').write_text('from store import obtain_session\ndef login(cookie):\n    return obtain_session(cookie)\n')
        self.graph=scan_repository(self.root)
        self.cfg=V5Config(target_chunk_tokens=48,max_chunk_tokens=80,min_chunk_tokens=16,min_files=1,max_files=6,per_file_candidates=8,candidate_limit=32)

    def test_config_file_and_ablations(self):
        loaded=V5Config.load(Path(__file__).resolve().parents[1]/'configs'/'pathpack_v05.yaml')
        self.assertAlmostEqual(loaded.alpha_file+loaded.beta_chunk+loaded.gamma_graph+loaded.delta_context,1)
        self.assertTrue(loaded.enable_hierarchy)
        with self.assertRaises(ValueError):V5Config(alpha_file=-1)
        with self.assertRaises(ValueError):self.cfg.ablation('no-symbol')
        for name in V5_ABLATIONS:
            packed=self._pack(self.cfg.ablation(name),256)
            self.assertLessEqual(token_count(packed.context),256)

    def _index(self,cfg):
        return HierarchicalIndex(build_chunk_index(self.root,self.graph,cfg.to_v4()),self.graph,self.root)

    def _pack(self,cfg,budget,query='zxqmarker login failure'):
        return PreparedQueryV5(self._index(cfg),query,cfg).pack(budget)

    def test_links_file_function_and_chunk_and_prefers_the_marker(self):
        result=self._pack(self.cfg,240)
        hits=[hit for hit in result.hits if 'zxqmarker' in hit.chunk.body]
        self.assertTrue(hits)
        hit=hits[0]
        self.assertEqual(hit.chunk.path,'auth.py')
        self.assertEqual(hit.chunk.function_id,'auth.py::Session.login')
        self.assertTrue(any(span[0]>=1 for span in hit.chunk.source_spans))
        lines=(self.root/'auth.py').read_text(encoding='utf-8').splitlines(keepends=True)
        self.assertEqual(hit.chunk.body,''.join(lines[ln-1][a:b] for ln,a,b in hit.chunk.source_spans))

    def test_signature_anchor_does_not_pull_neighboring_body_lines(self):
        index=build_chunk_index(self.root,self.graph,self.cfg.to_v4())
        chunk=next(c for c in index.chunks if c.function_id=='auth.py::Session.login' and 'zxqmarker' in c.body and c.line_start>1)
        anchored=apply_signature_anchor(chunk,(self.root/'auth.py').read_text().splitlines(keepends=True),index.functions[chunk.function_id],self.cfg)
        numbers=[ln for ln,_,_ in anchored.source_spans]
        self.assertIn(chunk.line_start,numbers)
        self.assertTrue(any('def login' in (self.root/'auth.py').read_text().splitlines()[ln-1] for ln in numbers))
        middle=[ln for ln in numbers if ln not in {line for line,_,_ in chunk.source_spans}]
        for ln in middle:
            self.assertRegex((self.root/'auth.py').read_text().splitlines()[ln-1],r'^\s*(?:@|async\s+def|def)\b')

    def test_budget_nesting_determinism_and_no_duplicate_source(self):
        prepared=PreparedQueryV5(self._index(self.cfg),'zxqmarker login',self.cfg)
        previous=''
        for budget in (32,64,128,256,1024,2048,4096):
            result=prepared.pack(budget)
            again=PreparedQueryV5(self._index(self.cfg),'zxqmarker login',self.cfg).pack(budget)
            self.assertLessEqual(token_count(result.context),budget)
            self.assertTrue(result.context.startswith(previous));previous=result.context
            self.assertEqual(result.context,again.context)
            self.assertEqual(sum(hit.chunk.tokens_est for hit in result.hits),token_count(result.context))
            spans={}
            seen=set()
            for hit in result.hits:
                self.assertNotIn(hit.chunk.id,seen);seen.add(hit.chunk.id)
                for ln,a,b in hit.chunk.source_spans:
                    for x,y in spans.get((hit.chunk.path,ln),[]):
                        self.assertFalse(max(a,x)<min(b,y))
                    spans.setdefault((hit.chunk.path,ln),[]).append((a,b))
        components=prepared.pack(512,include_candidates=True).diagnostics['candidates']
        self.assertTrue(components)
        for row in components:
            for key in ('file_score','chunk_score','graph_score','context_score','lexical_score'):
                self.assertGreaterEqual(row[key],0);self.assertLessEqual(row[key],1)

    def test_missing_graph_edges_and_ablation_switches(self):
        bare=CodeGraph(nodes=dict(self.graph.nodes),edges=set(),errors={})
        cfg=self.cfg.ablation('no-graph')
        index=HierarchicalIndex(build_chunk_index(self.root,bare,cfg.to_v4()),bare,self.root)
        result=PreparedQueryV5(index,'login cookie',cfg).pack(400)
        self.assertTrue(result.context)
        self.assertTrue(all(row['graph_score']==0 for row in result.diagnostics['candidates']))
        no_file=self._pack(self.cfg.ablation('no-file-score'),300)
        self.assertTrue(all(row['file_score']==0 for row in no_file.diagnostics['candidates']))
        flat=self._pack(self.cfg.ablation('no-hierarchy'),300)
        self.assertEqual(flat.diagnostics['local_blend'],0)
        self.assertFalse(flat.diagnostics['hierarchy'])
        quiet=self._pack(self.cfg.ablation('no-diversity'),300)
        self.assertTrue(all(row['diversity_contribution']==0 and row['file_penalty']==0 for row in quiet.diagnostics['candidates']))

    def test_hierarchy_reserves_a_second_file_under_a_tiny_global_cap(self):
        (self.root/'big.py').write_text('def popular():\n    return "session session session session"\n')
        (self.root/'target.py').write_text('def edge():\n    return "session"\n')
        graph=scan_repository(self.root)
        cfg=V5Config(candidate_limit=1,per_file_candidates=4,max_files=4,min_files=1,file_relative_threshold=0.05,enable_diversity=False,enable_graph=False,include_signature_anchor=False)
        def ordered(flag):
            conf=cfg if flag else cfg.ablation('no-hierarchy')
            index=HierarchicalIndex(build_chunk_index(self.root,graph,conf.to_v4()),graph,self.root)
            return {c.chunk.path for c in PreparedQueryV5(index,'session',conf).order}
        self.assertIn('target.py',ordered(True))
        self.assertNotIn('target.py',ordered(False))

    def test_empty_query_profile_and_legacy_methods_still_run(self):
        empty=self._pack(self.cfg,200,query='zzzznotpresent')
        self.assertEqual(empty.context,'')
        profile=self._pack(self.cfg,200).diagnostics['profile']
        for key in ('candidate_sec','graph_sec','file_rank_sec','chunk_score_sec','diversity_sec','anchor_sec','pack_stream_sec','prefix_sec'):
            self.assertIn(key,profile)
        for method in ('bm25','graph-expand','pathpack','pathpack-v3'):
            _graph,hits=retrieve(self.root,'login cookie',method=method,budget=300)
            self.assertTrue(hits)
        self.assertTrue(retrieve_v4(self.root,self.graph,'login',budget=200,config=self.cfg.to_v4()).context)

    def test_retrieval_modules_do_not_import_gold_or_read_a_planted_label(self):
        package=Path(__file__).resolve().parents[1]/'src'/'codegraph_rag'
        for name in ('pathpack_v5','hierarchical_retriever','file_ranker','chunk_ranker','embedding_retriever'):
            tree=ast.parse((package/f'{name}.py').read_text())
            modules=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
            self.assertFalse(any(token in module for module in modules for token in ('evaluation','function_gold','line_metrics','error_localization')))
        (self.root/'gold_labels.jsonl').write_text('{"patch":"SECRET GOLD"}')
        first=retrieve_v5(self.root,self.graph,'zxqmarker login',budget=300,config=self.cfg).context
        (self.root/'gold_labels.jsonl').write_text('DIFFERENT GOLD')
        second=retrieve_v5(self.root,self.graph,'zxqmarker login',budget=300,config=self.cfg).context
        self.assertEqual(first,second)
        self.assertNotIn('SECRET GOLD',first)

    def test_v5_benchmark_ablations_and_score_without_leaking_into_retrieval(self):
        from codegraph_rag.experiments import run_benchmark, score_run
        git=lambda *args:subprocess.run(['git','-C',str(self.root),*args],check=True,capture_output=True,text=True)
        git('init','-q');git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
        commit=git('rev-parse','HEAD').stdout.strip()
        with tempfile.TemporaryDirectory() as other:
            base=Path(other);snapshots=base/'snapshots';snapshots.mkdir()
            subprocess.run(['git','clone','--quiet',str(self.root),str(snapshots/'task')],check=True)
            row={'instance_id':'task','repo':'local/auth','base_commit':commit,'problem_statement':'zxqmarker login failure','split':'dev'}
            issues=base/'issues.jsonl';issues.write_text(json.dumps(row)+'\n')
            raw=base/'raw.jsonl';raw.write_text(json.dumps({**row,'patch':'--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-def decorate(fn):\n+def decorate(fn):  # patched\n'})+'\n')
            run=base/'run'
            run_benchmark(issues,snapshots,run,methods=('bm25','pathpack-v4','pathpack-v5'),config=self.cfg.to_v4(),v5_config=self.cfg,include_v5_ablations=True,budgets=(64,128))
            scored=score_run(run,raw=raw)
            self.assertTrue(scored['audit'])
            self.assertTrue((run/'stage_profile.csv').is_file())
            summary=(run/'summary.csv').read_text()
            for name in V5_ABLATIONS:
                label='pathpack-v5' if name=='full' else f'pathpack-v5/{name}'
                self.assertIn(label,summary)
            note=json.loads((run/'run.json').read_text())['embedding_baseline']
            self.assertEqual(note['status'],'not_run')
            self.assertIn('sentence-transformers',note['reason'])

    def test_v51_drops_the_file_cap_and_the_first_chunk_bonus(self):
        from types import SimpleNamespace
        from codegraph_rag.chunk_ranker import ChunkCandidate, order_chunks
        loaded=V5Config.load(Path(__file__).resolve().parents[1]/'configs/pathpack_v051.yaml')
        self.assertFalse(loaded.enable_file_filter)
        self.assertEqual(loaded.file_evidence,'function')
        self.assertEqual(loaded.first_file_bonus,0)
        for name in V51_ABLATIONS:
            variant=v51_variant(loaded,name)
            self.assertEqual(variant.enable_chunking, name!='coarse-coarse')
        (self.root/'big.py').write_text('def popular():\n    return "session session session session"\n')
        (self.root/'target.py').write_text('def edge():\n    return "session"\n')
        graph=scan_repository(self.root)
        soft=V5Config(candidate_limit=1,max_files=4,min_files=1,file_relative_threshold=0.05,enable_file_filter=False,enable_diversity=False,enable_graph=False,include_signature_anchor=False)
        index=HierarchicalIndex(build_chunk_index(self.root,graph,soft.to_v4()),graph,self.root)
        ranked=PreparedQueryV5(index,'session',soft)
        self.assertNotIn('target.py',{item.chunk.path for item in ranked.order})
        self.assertFalse(any(row['reason']=='outside_file_pool' or row['decision']=='outside_file_pool' for row in ranked.pack(200).diagnostics['candidates']))
        def chunk(identifier,path,score):
            return ChunkCandidate(SimpleNamespace(id=identifier,path=path,tokens_est=40),0.4,score,0,score,0,score,0,0,[],[])
        def ids(bonus):
            pool=[chunk('strong','gold.py',0.9),chunk('weak','gold.py',0.2)]+[chunk(f'n{i}',f'n{i}.py',0.55) for i in range(5)]
            cfg=V5Config(enable_cost=False,alpha_file=0,beta_chunk=1,gamma_graph=0,delta_context=0,first_file_bonus=bonus,diversity_weight=0.4,local_resistance=1,redundancy_strength=0)
            return [item.chunk.id for item in order_chunks(pool,{item.chunk.id:frozenset([item.chunk.id]) for item in pool},cfg)]
        self.assertLess(ids(0).index('strong'), ids(0).index('n0'))
        self.assertLess(ids(0.4).index('n0'), ids(0.4).index('weak'))
