"""Copy the rendered figures out of notebooks/11_pathpack_h.ipynb into docs/figures. No numbers are recomputed."""
import base64
from pathlib import Path
import nbformat

NAMES = ['holdout_summary', 'holdout_differences', 'holdout_budgets', 'ablations', 'pool_vs_packed', 'per_repository',
         'u_holdout2', 'u_differences', 'holdout3_summary', 'holdout3_differences', 'llm_probe']
nb = nbformat.read('notebooks/11_pathpack_h.ipynb', 4)
out = Path('docs/figures'); out.mkdir(parents=True, exist_ok=True)
n = 0
for cell in nb.cells:
    for o in cell.get('outputs', []):
        if 'data' in o and 'image/png' in o['data']:
            (out / f'{NAMES[n]}.png').write_bytes(base64.b64decode(o['data']['image/png'])); n += 1
print(n, 'figures')
