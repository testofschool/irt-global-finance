# Correcting Fund Performance Rankings Under Regime Difficulty with IRT

## Reproduction

### Path A: Full reproduction from Yahoo Finance (requires internet)
```bash
pip install -r requirements.txt
python src/download_data.py       # Downloads prices -> data/prices.csv
python src/experiment.py          # Full analysis (canonical results), ~3 min
```

### Path B: Offline verification from canonical parameters
```bash
pip install -r requirements.txt
python src/reproduce_from_derived.py              # ~15s, full verification
python src/reproduce_from_derived.py --quick      # ~15s, rankings + crisis only (skips LLTM)
python src/reproduce_from_derived.py --stress-refit  # re-runs 1PL stress test (runtime varies)
```

Path B loads canonical IRT parameters from `data/canonical_theta.csv` and `data/canonical_period_params.csv`. It does NOT re-fit the 2PL model. Rankings, LLTM, and crisis correlation reproduce canonical values exactly. The stress test result is loaded from `data/canonical_stress.json` by default; use `--stress-refit` to re-run the 1PL stress test independently.

**Note on `--stress-refit`:** The canonical result (7/10 IRT wins) was obtained with numpy 1.26 / scipy 1.12 on Ubuntu 24.04. Because the 1PL L-BFGS-B optimizer path is sensitive to numerical library versions, re-fitting may yield 6/10 or 7/10 and may take substantially longer than 60 seconds on newer NumPy/SciPy versions. This does not affect the default verification path, which loads the canonical result directly.

Verification figures are written to `paper/figures_verification/` (never overwrites canonical `paper/figures/`). Results are written to `outputs/results_from_derived.json`.

### Compile the paper
```bash
cd paper
pdflatex main.tex && pdflatex main.tex
```
(`main.bbl` is pre-generated; BibTeX is not required.)

### Smoke test
```bash
python -m py_compile src/experiment.py
python -m py_compile src/reproduce_from_derived.py
python -m py_compile src/download_data.py
python src/reproduce_from_derived.py --quick
```

## Environment
- Tested: Python 3.11/3.12, Ubuntu 24.04, numpy 1.26, scipy 1.12
- Runtime: ~3 min (Path A), ~15s (Path B default/quick), variable (Path B stress-refit; see note below)

## Data
Raw prices are not redistributed (Yahoo Finance terms apply). `data/` contains:
- `response_matrix.csv` — derived 50×217 binary matrix
- `period_features.csv` — macro features per period
- `canonical_theta.csv` — canonical IRT fund ability parameters
- `canonical_period_params.csv` — canonical IRT period difficulty/discrimination
- `canonical_stress.json` — canonical stress test results
- `ticker_manifest.json`, `data_provenance.json` — provenance

## License
MIT. See [LICENSE](LICENSE).
