"""Bounded checks of supplementary public source using fictional observations."""
from pathlib import Path
import ast
import importlib
import importlib.util
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research-code'
files = sorted(SOURCE.rglob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

origin = SOURCE / 'llm-origin-bias'
sys.path.insert(0, str(origin))
modules = sorted((origin / 'analysis/paper_home_bias').rglob('*.py'))
for path in modules:
    module = '.'.join(path.relative_to(origin).with_suffix('').parts)
    importlib.import_module(module)

for path in sorted((SOURCE / 'llm-political-risk-signals/experiments').glob('*.py')):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

from analysis.paper_home_bias.inference_utils import (
    cluster_covariance, ols_fit, prepare_design, residualise_multiway,
)

rng = np.random.default_rng(20260907)
rows = []
for country in range(8):
    for year in range(4):
        for model in range(4):
            x1, x2 = rng.normal(size=2)
            rows.append({
                'country': f'Fictional-{country}',
                'country_year': f'Fictional-{country}:{year}',
                'model': f'Toy-rater-{model}',
                'us_on_us': x1,
                'cn_on_cn': x2,
                'score': country + year + model + 1.25*x1 - 0.75*x2,
            })
panel = pd.DataFrame(rows)
design = prepare_design(panel)
fit = ols_fit(design['y'], design['x'])
np.testing.assert_allclose(fit['beta'], [1.25, -0.75], atol=1e-9)
residual = residualise_multiway(panel, ['score'], ('country_year', 'model'))
for group in ('country_year', 'model'):
    check = residual.assign(group=panel[group]).groupby('group')['score'].mean()
    np.testing.assert_allclose(check, 0, atol=1e-9)
noisy_y = design['y'] + rng.normal(scale=0.1, size=len(panel))
cov = cluster_covariance(noisy_y, design['x'], design['clusters'], kind='cr0')
assert cov.shape == (2, 2) and np.isfinite(cov).all()
np.testing.assert_allclose(cov, cov.T, atol=1e-12)
assert np.linalg.eigvalsh(cov).min() >= -1e-12
print(f'Parsed {len(files)} source files; imported {len(modules) + 2} analysis modules; fictional fixed-effect and covariance checks passed.')
print('Scope: synthetic helper checks only. No empirical result or full producer rerun is claimed.')
