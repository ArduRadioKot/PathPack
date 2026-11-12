from pathlib import Path
import tempfile
import unittest

from codegraph_rag.benchmark import files_in_patch, build_manifests, read_jsonl


class BenchmarkTests(unittest.TestCase):
    def test_pre_fix_paths_for_rename_and_new_file(self):
        patch = ("diff --git a/old.py b/new.py\n--- a/old.py\n+++ b/new.py\n"
                 "diff --git a/brand_new.py b/brand_new.py\n--- /dev/null\n+++ b/brand_new.py\n")
        self.assertEqual(files_in_patch(patch), ["old.py"])

    def test_file_labels_and_no_patch_in_inputs(self):
        patch = "diff --git a/app/a.py b/app/a.py\n--- a/app/a.py\n+++ b/app/a.py\n@@ -1 +1 @@\n-old\n+new\n"
        self.assertEqual(files_in_patch(patch), ["app/a.py"])
        rows = [dict(instance_id=f"repo__a-{n}", repo="repo/a", base_commit="abc", problem_statement=f"bug {n}", patch=patch) for n in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            result = build_manifests(rows, tmp, limit=3, dev_count=1)
            inputs = read_jsonl(Path(tmp) / "issues.jsonl")
            labels = read_jsonl(Path(tmp) / "gold_labels.jsonl")
            self.assertEqual(result["test"], 2)
            self.assertNotIn("patch", inputs[0])
            self.assertNotIn("gold_python_files", inputs[0])
            self.assertEqual(labels[0]["gold_python_files"], ["app/a.py"])
            self.assertEqual({x["instance_id"] for x in inputs}, {x["instance_id"] for x in labels})


if __name__ == "__main__":
    unittest.main()
