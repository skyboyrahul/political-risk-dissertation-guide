"""Import and fictional arithmetic checks for the additional appendix source."""
from pathlib import Path
import ast
import importlib
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'research-code/prs-nowcast'
files = sorted(ROOT.rglob('*.py'))
for path in files:
    ast.parse(path.read_text(), filename=str(path))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis/thesis_figures'))
sys.path.insert(0, str(ROOT / 'analysis'))
modules = []
for path in files:
    if path.name == 'build_anchor_ablation.py':
        continue  # External inference helpers and scikit-learn are documented.
    name = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
    importlib.import_module(name)
    modules.append(name)
from analysis.decision_use import review_trigger_experiment as trigger
from analysis.repeat_generation import reanalyse

np.testing.assert_array_equal(trigger.below_threshold([49, 50, 51], 50), [True, False, False])
counts = trigger.confusion_counts(np.array([True, True, False, False]), np.array([True, False, True, False]))
assert [counts[k] for k in ('tp', 'fn', 'fp', 'tn')] == [1, 1, 1, 1]
assert trigger.decision_loss(counts, 10) == 12
summary = reanalyse.metrics([(1, 1), (2, 2), (3, 3)])
assert summary['mad'] == 0 and summary['exact_match_pct'] == 100
assert summary['icc_3_1'] == 1
print(f'Parsed {len(files)} additional source files; imported {len(modules)} modules; fictional review-rule and repeatability checks passed.')
print('No empirical rerun, provider call or Wikimedia request; anchor producer is syntax-checked only.')
