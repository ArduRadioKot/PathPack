"""Changed-line definitions stay separate from insertion anchors."""
import unittest
from pathlib import Path
import tempfile

from codegraph_rag.evaluation_v4 import score_prediction
from codegraph_rag.line_metrics import build_line_label, evaluable_changed, extra_metrics


def _pred(spans,tokens=10):
    return {'instance_id':'t','method':'pathpack-v5','budget_est':100,'tokens_est':tokens,'latency_sec':0,
            'selected':[{'path':'a.py','id':'a.py::f::chunk_0','function_id':'a.py::f','source_spans':spans}]}


class LineMetricTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_replacement_uses_old_line_numbers_and_ignores_the_added_text(self):
        (self.root/'a.py').write_text('def f(x):\n    return x\n')
        patch=('--- a/a.py\n+++ b/a.py\n@@ -2 +2 @@\n-    return x\n+    return x + INSERTED_ONLY_IN_PATCH\n')
        gold=build_line_label(patch,self.root)
        self.assertEqual(evaluable_changed(gold),{('a.py',2)})
        self.assertNotIn('INSERTED_ONLY_IN_PATCH',(self.root/'a.py').read_text())
        self.assertEqual(gold['insertion_anchors'],{})
        self.assertFalse(gold['line_audit']['metrics_are_mixed'])
        bare=_pred([(1,0,9)])
        score=score_prediction(bare,gold)
        self.assertEqual(score['changed_line_recall'],0)
        self.assertEqual(score['function_recall'],0)
        self.assertEqual(extra_metrics(bare['selected'],gold)['relevant_line_density'],0)
        hit=_pred([(2,0,14)])
        self.assertEqual(score_prediction(hit,gold)['changed_line_recall'],1)
        self.assertEqual(score_prediction(hit,gold)['function_recall'],1)
        self.assertEqual(extra_metrics(hit['selected'],gold)['relevant_line_density'],1)

    def test_pure_insertion_is_not_changed_line_recall(self):
        (self.root/'a.py').write_text('def f(x):\n    return x\n')
        patch=('--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,3 @@\n def f(x):\n+    x += 1\n     return x\n')
        gold=build_line_label(patch,self.root)
        self.assertFalse(evaluable_changed(gold))
        self.assertEqual(set(gold['insertion_anchors'].get('a.py',[])),{1,2})
        score=score_prediction(_pred([(2,0,14)]),gold)
        self.assertIsNone(score['changed_line_recall'])
        extra=extra_metrics(_pred([(2,0,14)])['selected'],gold)
        self.assertEqual(extra['insertion_anchor_recall'],0.5)
        self.assertNotEqual(extra['insertion_anchor_reason'],'changed_line_recall')
        self.assertIsNone(extra['relevant_line_density'])
        self.assertEqual(extra['relevant_line_density_reason'],'no_evaluable_changed_lines')
        self.assertIsNotNone(extra['base_touch_recall'])
        empty=extra_metrics([],gold)
        self.assertIsNone(empty['relevant_line_density'])
        self.assertEqual(empty['relevant_line_density_reason'],'no_evaluable_changed_lines')

    def test_unmapped_insertion_and_new_file_are_excluded(self):
        (self.root/'a.py').write_text('\n\n\n')
        patch='--- a/a.py\n+++ b/a.py\n@@ -2,0 +3 @@\n+added\n'
        gold=build_line_label(patch,self.root)
        self.assertFalse(gold['insertion_anchors'])
        self.assertTrue(any(item['reason']=='insertion_anchor_unmapped' for item in gold['anchor_exclusions']))
        new=build_line_label('--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+answer=42\n',self.root)
        self.assertEqual(new['exclusions'][0]['reason'],'new_file_no_base_source')
        self.assertIsNone(score_prediction({'instance_id':'t','method':'pathpack-v5','budget_est':10,'tokens_est':0,'latency_sec':0,'selected':[]},new)['file_recall'])

    def test_class_method_nested_function_and_module_line(self):
        (self.root/'a.py').write_text('VALUE = 1\nclass Box:\n    def method(self, x):\n        def inner(y):\n            return y\n        return inner(x)\n')
        method=('--- a/a.py\n+++ b/a.py\n@@ -6 +6 @@\n-        return inner(x)\n+        return inner(x + 1)\n')
        gold=build_line_label(method,self.root)
        self.assertEqual(set(gold['functions']),{'a.py::Box.method'})
        nested=('--- a/a.py\n+++ b/a.py\n@@ -5 +5 @@\n-            return y\n+            return y + 1\n')
        inner=build_line_label(nested,self.root)
        self.assertEqual(set(inner['functions']),{'a.py::Box.method.inner'})
        self.assertIn('a.py::Box.method.inner',inner['line_audit']['nested_functions'])
        module=('--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n')
        outside=build_line_label(module,self.root)
        self.assertFalse(outside['functions'])
        self.assertTrue(outside['line_audit']['changed_lines_outside_function_bodies'])
        self.assertEqual(evaluable_changed(outside),{('a.py',1)})

    def test_mentioning_the_name_is_not_function_coverage(self):
        (self.root/'a.py').write_text('def target(x):\n    return x\n')
        patch='--- a/a.py\n+++ b/a.py\n@@ -2 +2 @@\n-    return x\n+    return str(x)\n'
        gold=build_line_label(patch,self.root)
        name_only=_pred([(1,0,13)])
        name_only['selected'][0]['function_id']='a.py::target'
        name_only['selected'][0]['id']='a.py::target::chunk_0'
        score=score_prediction(name_only,gold)
        self.assertEqual(name_only['selected'][0]['function_id'],'a.py::target')
        self.assertEqual(score['function_recall'],0)
        self.assertEqual(score['changed_line_recall'],0)
        self.assertEqual(score['complete_function_recall'],0)

    def test_density_counts_only_emitted_lines_on_labeled_tasks(self):
        (self.root/'a.py').write_text('def f(x):\n    a = x\n    return a\n')
        patch='--- a/a.py\n+++ b/a.py\n@@ -2 +2 @@\n-    a = x\n+    a = x + 1\n'
        gold=build_line_label(patch,self.root)
        both={'selected':[{'path':'a.py','source_spans':[(2,0,10),(3,0,13)]}]}
        extra=extra_metrics(both['selected'],gold)
        self.assertEqual(extra['emitted_nonempty_lines'],2)
        self.assertEqual(extra['relevant_line_density'],0.5)
        self.assertIsNone(extra_metrics([],gold)['relevant_line_density'])
        self.assertEqual(extra_metrics([],gold)['relevant_line_density_reason'],'empty_context')
