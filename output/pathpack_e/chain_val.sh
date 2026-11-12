set -e
until [ -f output/pathpack_e/train_history/sympy__sympy.done ]; do sleep 10; done
sleep 5
EX="output/pathpack_r_splits_v2/validation_issues.jsonl output/pathpack_r_splits_v2/final_holdout_issues.jsonl output/benchmark/issues.jsonl"
python -m codegraph_rag.pathpack_r.edit_train fit --train-dir output/pathpack_e/train --out output/pathpack_e/model.pkl --exclude $EX
python -m codegraph_rag.pathpack_r.edit_train fit --train-dir output/pathpack_e/train_history --out output/pathpack_e/model_history.pkl --exclude $EX
python -m codegraph_rag.pathpack_r.edit_train fit --train-dir output/pathpack_e/train output/pathpack_e/train_history --out output/pathpack_e/model_both.pkl --exclude $EX
python -m codegraph_rag.pathpack_r benchmark --issues output/pathpack_r_splits_v2/validation_issues.jsonl --split validation --out output/pathpack_e_val --method pathpack-e pathpack-eh pathpack-h pathpack-h/nograph pathpack-h/nohist pathpack-h/nomemory pathpack-h/ranked pathpack-h/skeleton pathpack-r/fusion-epack
echo CHAINDONE
