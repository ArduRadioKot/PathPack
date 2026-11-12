from pathlib import Path
import tempfile
import unittest

from codegraph_rag.function_gold import functions_touched, old_changed_lines


class GoldFunctionTests(unittest.TestCase):
    def test_deleted_old_lines(self):
        patch='''diff --git a/auth.py b/auth.py
--- a/auth.py
+++ b/auth.py
@@ -1,4 +1,4 @@
 def login(x):
-    return x
+    return str(x)
 
 def other(x):
'''
        self.assertIn(2, old_changed_lines(patch)['auth.py'])
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'auth.py').write_text('def login(x):\n    return x\n\ndef other(x):\n    return x\n')
            self.assertEqual(functions_touched(patch,tmp),['auth.py::login'])

    def test_new_files_not_labeled(self):
        patch='''diff --git a/new.py b/new.py
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+def foo():
+    return 2
'''
        self.assertEqual(old_changed_lines(patch),{})

if __name__ == '__main__': unittest.main()
