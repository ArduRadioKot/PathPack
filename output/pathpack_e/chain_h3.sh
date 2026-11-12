while kill -0 81216 2>/dev/null; do sleep 15; done
for r in django__django sympy__sympy sphinx-doc__sphinx matplotlib__matplotlib scikit-learn__scikit-learn; do
  until [ -f output/pathpack_e/loro/$r/config.yaml ]; do sleep 15; done
done
shasum -a 256 output/pathpack_e/loro/*/*.pkl > output/pathpack_e/loro/SHA256SUMS
python -m codegraph_rag.pathpack_r benchmark --issues output/pathpack_r_splits_v2/final_holdout3_issues.jsonl --split final_holdout3 --out output/pathpack_holdout3 --method bm25 pathpack-v4 pathpack-r/fusion pathpack-h/nomemory pathpack-u pathpack-u/nobundle pathpack-u/noisyor > output/pathpack_holdout3.log 2>&1
for r in django/django sympy/sympy sphinx-doc/sphinx matplotlib/matplotlib scikit-learn/scikit-learn; do
  d=$(echo $r | sed 's#/#__#')
  grep "\"repo\": \"$r\"" output/pathpack_r_splits_v2/final_holdout3_issues.jsonl > output/pathpack_e/loro/$d/issues.jsonl
  python -m codegraph_rag.pathpack_r benchmark --issues output/pathpack_e/loro/$d/issues.jsonl --split final_holdout3 --out output/pathpack_holdout3_loro_$d --config output/pathpack_e/loro/$d/config.yaml --method pathpack-h/nomemory pathpack-u > output/pathpack_holdout3_loro_$d.log 2>&1
done
echo H3DONE >> output/pathpack_holdout3.log
