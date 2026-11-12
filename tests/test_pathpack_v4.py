import ast,json,math,random,subprocess,tempfile,unittest
from pathlib import Path
from dataclasses import replace
from codegraph_rag.config import V4Config,ABLATIONS
from codegraph_rag.parser import scan_repository
from codegraph_rag.adaptive_chunker import build_chunk_index
from codegraph_rag.hybrid_ranker import HybridIndex
from codegraph_rag.pathpack_v4 import PreparedQuery,retrieve_v4
from codegraph_rag.context_packer import token_count
from codegraph_rag.evaluation_v4 import label_patch,score_prediction


class V4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.code='''def decorate(f):
    return f

class Auth:
    @decorate
    async def login(self, user):
        result = (
            user
            + " value"
        )
        def nested(value):
            return value.upper()
        try:
            for item in result:
                if item:
                    result += item
        except ValueError:
            return None
        return nested(result)

'''+ 'def very_long_login(user):\n'+''.join(f'    part_{i} = user + "login field {i}"\n' for i in range(100))+'    return user\n'
        (self.root/'auth.py').write_text(self.code)
        (self.root/'broken.py').write_text('def broken(:')
        (self.root/'empty.py').write_text('')
        self.graph=scan_repository(self.root);self.cfg=V4Config(target_chunk_tokens=96,max_chunk_tokens=192,min_chunk_tokens=24)

    def index(self,cfg=None):return build_chunk_index(self.root,self.graph,cfg or self.cfg)

    def test_ast_features_and_stable_ids(self):
        index=self.index();again=self.index()
        self.assertEqual([c.id for c in index.chunks],[c.id for c in again.chunks])
        self.assertIn('broken.py',index.errors)
        self.assertIn('auth.py::Auth.login.nested',index.functions)
        fn=index.functions['auth.py::Auth.login'];self.assertEqual(fn.parent_id,'auth.py::Auth')
        self.assertTrue(fn.signature.startswith('async def login'))
        short=[c for c in index.chunks if c.qualname=='decorate'];self.assertEqual(len(short),1)
        self.assertGreater(len([c for c in index.chunks if c.qualname=='very_long_login']),5)
        decorated=[c for c in index.chunks if c.qualname=='Auth.login']
        self.assertTrue(any('@decorate' in c.body for c in decorated))
        self.assertTrue(any('user\n            + " value"' in c.body for c in decorated))
        for c in index.chunks:
            self.assertIn(c.function_id,self.graph.nodes);self.assertLessEqual(c.tokens_est,self.cfg.max_chunk_tokens)
            self.assertTrue(c.body.strip());self.assertTrue(c.source_spans)
        # Indexing does not create a cartesian graph of chunks.
        before=len(self.graph.edges);self.index();self.assertEqual(before,len(self.graph.edges))

    def test_nested_source_has_single_owner(self):
        index=self.index();owners={}
        for c in index.chunks:
            for ln,_,_ in c.source_spans:
                owners.setdefault((c.path,ln),set()).add(c.function_id)
        self.assertTrue(all(len(ids)==1 for ids in owners.values()))

    def test_prefix_safety_determinism_and_no_source_repetition(self):
        index=HybridIndex(self.index(),self.graph)
        prepared=PreparedQuery(index,'login user fails ValueError',self.cfg)
        previous=''
        for budget in (1,20,64,128,257,1024,2048,4096):
            result=prepared.pack(budget);again=PreparedQuery(index,'login user fails ValueError',self.cfg).pack(budget)
            self.assertLessEqual(token_count(result.context),budget)
            self.assertTrue(result.context.startswith(previous));previous=result.context
            self.assertEqual(result.to_dict(),again.to_dict())
            self.assertEqual(sum(h.chunk.tokens_est for h in result.hits),token_count(result.context))
            spans={}
            for h in result.hits:
                for ln,a,b in h.chunk.source_spans:
                    key=(h.chunk.path,ln)
                    for x,y in spans.get(key,[]):self.assertFalse(max(a,x)<min(b,y))
                    spans.setdefault(key,[]).append((a,b))
        chosen=prepared.pack(1024).diagnostics
        self.assertTrue(all(c['decision'] for c in chosen['candidates']))
        self.assertTrue(any(c['decision'] in {'budget_exceeded','after_budget_cutoff','selected_partial'} for c in chosen['candidates']))

    def test_graph_evidence_and_ablations(self):
        (self.root/'routes.py').write_text('from store import obtain_session\ndef login(cookie):\n    return obtain_session(cookie)\n')
        (self.root/'store.py').write_text('def obtain_session(key):\n    return key\n')
        graph=scan_repository(self.root);cfg=replace(self.cfg,candidate_limit=1000,seeds=1)
        index=HybridIndex(build_chunk_index(self.root,graph,cfg),graph)
        p=PreparedQuery(index,'login cookie',cfg)
        self.assertTrue(any(c.graph>0 and c.chunk.function_id=='store.py::obtain_session' for c in p.candidates))
        for ablation in ABLATIONS:
            conf=cfg.ablation(ablation)
            prepared=PreparedQuery(index,'login cookie',conf)
            for c in prepared.candidates:
                self.assertTrue(0<=c.lexical<=1 and 0<=c.symbol<=1 and 0<=c.graph<=1)
                if ablation=='no-graph':self.assertEqual(c.graph,0)
                if ablation=='no-symbol':self.assertEqual(c.symbol,0)
                if ablation=='no-diversity':self.assertEqual(c.diversity,0)
            if ablation=='no-cost':self.assertTrue(all(c.utility==c.score for c in prepared.order))
        both=PreparedQuery(index,'login cookie',replace(cfg,graph_direction='both'))
        self.assertEqual([(c.chunk.id,c.graph) for c in both.candidates],[(c.chunk.id,c.graph) for c in p.candidates])
        outward=PreparedQuery(index,'login cookie',replace(cfg,graph_direction='out'))
        inward=PreparedQuery(index,'login cookie',replace(cfg,graph_direction='in'))
        self.assertTrue(any(c.graph>0 and c.chunk.function_id=='store.py::obtain_session' for c in outward.candidates))
        self.assertTrue(all(c.graph==0 or c.chunk.function_id!='store.py::obtain_session' for c in inward.candidates))
        caller=PreparedQuery(index,'obtain_session key',replace(cfg,graph_direction='in',seeds=1))
        self.assertTrue(any(c.graph>0 and c.chunk.function_id=='routes.py::login' for c in caller.candidates))
        with self.assertRaises(ValueError):V4Config(graph_direction='sideways')

    def test_unicode_and_very_long_single_line(self):
        (self.root/'unicode.py').write_text('def login():\n    value = "'+'ёж'*1500+'"\n    return value\n')
        graph=scan_repository(self.root);index=build_chunk_index(self.root,graph,self.cfg)
        self.assertTrue([c for c in index.chunks if c.path=='unicode.py'])
        self.assertTrue(all(c.tokens_est<=self.cfg.max_chunk_tokens for c in index.chunks))
        p=PreparedQuery(HybridIndex(index,graph),'login value',self.cfg)
        for b in (10,40,100,1024,2048):self.assertLessEqual(token_count(p.pack(b).context),b)

    def test_empty_query_and_invalid_configuration(self):
        p=PreparedQuery(HybridIndex(self.index(),self.graph),'zzznotpresent',self.cfg)
        self.assertEqual(p.pack(1024).context,'')
        with self.assertRaises(ValueError):p.pack(0)
        with self.assertRaises(ValueError):V4Config(max_chunk_tokens=10)
        with self.assertRaises(ValueError):V4Config(graph_weight=float('nan'))

    def test_duplicate_source_bodies_and_no_chunking_mode(self):
        duplicate='def cookie_session(user):\n    return user.strip()\n'
        for name in ('copy_a.py','copy_b.py'):(self.root/name).write_text(duplicate)
        graph=scan_repository(self.root)
        index=HybridIndex(build_chunk_index(self.root,graph,self.cfg),graph)
        result=PreparedQuery(index,'cookie session user',self.cfg).pack(100000)
        copies=[h for h in result.hits if h.chunk.path in {'copy_a.py','copy_b.py'}]
        self.assertEqual(len(copies),1)
        self.assertTrue(any(c['decision']=='duplicate_content' for c in result.diagnostics['candidates']))
        conf=self.cfg.ablation('no-chunking')
        whole=build_chunk_index(self.root,graph,conf)
        self.assertEqual(len([c for c in whole.chunks if c.qualname=='very_long_login']),1)
        prepared=PreparedQuery(HybridIndex(whole,graph),'very long login',conf)
        self.assertEqual(prepared.pack(1).context,'')

    def test_renamed_file_nested_method_and_unknown_labels(self):
        old='class C:\n    def outer(self):\n        def inner(x):\n            return x\n        return inner(1)\n'
        (self.root/'before.py').write_text(old)
        patch='diff --git a/before.py b/after.py\n--- a/before.py\n+++ b/after.py\n@@ -4 +4 @@\n-            return x\n+            return x + 1\n'
        gold=label_patch(patch,self.root)
        self.assertEqual(gold['gold_python_files'],['before.py'])
        self.assertEqual(set(gold['functions']),{'before.py::C.outer.inner'})
        unknown=label_patch('--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+answer=42\n',self.root)
        prediction={'instance_id':'unknown','method':'pathpack-v4','budget_est':100,'tokens_est':0,'latency_sec':0,'selected':[]}
        score=score_prediction(prediction,unknown)
        self.assertIsNone(score['function_recall']);self.assertIsNone(score['file_recall'])
        self.assertIsNone(score['useful_functions_per_1k'])
        (self.root/'first.py').write_text('def f(x):\n    return x\n')
        outside=label_patch('--- a/first.py\n+++ b/first.py\n@@ -1,2 +1,3 @@\n+import math\n def f(x):\n     return x\n',self.root)
        self.assertFalse(outside['functions'])
        inside=label_patch('--- a/first.py\n+++ b/first.py\n@@ -1,2 +1,3 @@\n def f(x):\n+    x += 1\n     return x\n',self.root)
        self.assertEqual(set(inside['functions']),{'first.py::f'})

    def test_conditional_redefinitions_have_distinct_chunk_and_gold_ids(self):
        code='if True:\n    def b(user):\n        return user.encode()\nelse:\n    def b(user):\n        return user.decode()\n'
        (self.root/'six.py').write_text(code)
        graph=scan_repository(self.root);index=build_chunk_index(self.root,graph,self.cfg)
        chunks=[c for c in index.chunks if c.path=='six.py']
        self.assertEqual(len(chunks),2);self.assertEqual(len({c.id for c in chunks}),2)
        self.assertTrue(all(c.function_id=='six.py::b' for c in chunks))
        self.assertEqual(len(index.functions['six.py::b'].definitions),2)
        p=PreparedQuery(HybridIndex(index,graph),'b user encode decode',self.cfg.ablation('no-diversity')).pack(100000)
        ids=[h.chunk.id for h in p.hits];self.assertEqual(len(ids),len(set(ids)))
        patch='--- a/six.py\n+++ b/six.py\n@@ -3 +3 @@\n-        return user.encode()\n+        return user.encode("utf8")\n@@ -6 +6 @@\n-        return user.decode()\n+        return user.decode("utf8")\n'
        gold=label_patch(patch,self.root)
        self.assertEqual(len(gold['functions']),2)

    def test_metric_requires_body_and_exact_changed_lines(self):
        (self.root/'metric.py').write_text('def target(x):\n    a = x\n    b = a + 1\n    return b\n\ndef other():\n    return 0\n')
        patch='--- a/metric.py\n+++ b/metric.py\n@@ -1,7 +1,7 @@\n def target(x):\n     a = x\n-    b = a + 1\n+    b = a + 2\n     return b\n \n def other():\n     return 0\n'
        gold=label_patch(patch,self.root);self.assertEqual(set(gold['functions']),{'metric.py::target'})
        p={'instance_id':'x','method':'pathpack-v4','budget_est':100,'tokens_est':10,'latency_sec':0,'selected':[{'path':'metric.py','id':'metric.py::target::chunk_0','function_id':'metric.py::target','source_spans':[(1,0,15)]}]}
        s=score_prediction(p,gold);self.assertEqual(s['file_recall'],1);self.assertEqual(s['function_recall'],0);self.assertEqual(s['changed_line_recall'],0)
        p['selected'][0]['source_spans']=[(3,0,14)]
        s=score_prediction(p,gold);self.assertEqual(s['function_recall'],1);self.assertEqual(s['changed_line_recall'],1)
        self.assertEqual(s['useful_functions_per_1k'],100)
        new=label_patch('--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+def new():\n+    return 0\n',self.root)
        self.assertFalse(new['functions']);self.assertEqual(new['exclusions'][0]['reason'],'new_file_no_base_source')

    def test_retrieval_imports_no_evaluation_and_no_gold_fields(self):
        package=Path(__file__).resolve().parents[1]/'src/codegraph_rag'
        for name in ('adaptive_chunker','hybrid_ranker','context_packer','pathpack_v4'):
            source=(package/f'{name}.py').read_text();tree=ast.parse(source)
            imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any('evaluation' in name or 'function_gold' in name for name in imports))
        (self.root/'gold_labels.jsonl').write_text('{"patch":"SECRET GOLD"}')
        a=retrieve_v4(self.root,self.graph,'login user',budget=1024,config=self.cfg)
        (self.root/'gold_labels.jsonl').write_text('totally different gold')
        b=retrieve_v4(self.root,self.graph,'login user',budget=1024,config=self.cfg)
        self.assertEqual(a.context,b.context);self.assertNotIn('SECRET GOLD',a.context)

    def test_tiny_benchmark_and_scoring_pipeline(self):
        from codegraph_rag.experiments import run_benchmark,score_run,prepare_holdout
        git=lambda *args:subprocess.run(['git','-C',str(self.root),*args],check=True,capture_output=True,text=True)
        git('init','-q');git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
        commit=git('rev-parse','HEAD').stdout.strip()
        with tempfile.TemporaryDirectory() as other:
            base=Path(other);snapshots=base/'snapshots';snapshots.mkdir()
            # A symlink is deliberately disallowed by snapshot containment; use a local clone.
            subprocess.run(['git','clone','--quiet',str(self.root),str(snapshots/'task')],check=True)
            row={'instance_id':'task','repo':'local/auth','base_commit':commit,'problem_statement':'login user failure','split':'dev'}
            issues=base/'issues.jsonl';issues.write_text(json.dumps(row)+'\n')
            raw=base/'raw.jsonl';raw.write_text(json.dumps({**row,'patch':'--- a/auth.py\n+++ b/auth.py\n@@ -2 +2 @@\n-    return f\n+    return str(f)\n'})+'\n')
            run=base/'run';run_benchmark(issues,snapshots,run,config=replace(self.cfg,candidate_limit=64),include_ablations=True,budgets=(128,256))
            result=score_run(run,raw=raw);self.assertTrue(result['audit'])
            self.assertTrue((run/'summary.csv').is_file());self.assertTrue((run/'comparisons.csv').is_file())
            self.assertEqual(json.loads((run/'run.json').read_text())['status'],'scored')
            with self.assertRaisesRegex(ValueError,'Run exists'):run_benchmark(issues,snapshots,run)

    def test_legacy_scoring_rejects_chunk_ids_instead_of_reporting_false_zero(self):
        from codegraph_rag.evaluate import score_retrieval
        from codegraph_rag.audit import audit_predictions
        pred=self.root/'prediction.jsonl';gold=self.root/'gold.jsonl'
        pred.write_text(json.dumps({'instance_id':'x','method':'pathpack-v4','budget_est':100,'tokens_est':10,'selected':[{'id':'a.py::f::chunk_0','function_id':'a.py::f','path':'a.py','tokens_est':10}]})+'\n')
        gold.write_text(json.dumps({'instance_id':'x','gold_python_files':['a.py'],'gold_functions':['a.py::f']})+'\n')
        with self.assertRaisesRegex(ValueError,'body-line coverage'):score_retrieval(pred,gold)
        with self.assertRaisesRegex(ValueError,'audit --run'):audit_predictions([pred],gold)
