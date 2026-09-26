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

**Note on `--stress-refit`:** The canonical stress-test result (7/10 IRT wins) comes from `src/experiment.py`; only the win count is committed (`data/canonical_stress.json`), not the per-level values. `--stress-refit` is *not* the same procedure as the canonical run, so a different count is expected independently of library versions:

| | `src/experiment.py` (canonical) | `src/reproduce_from_derived.py --stress-refit` (offline) |
|---|---|---|
| Masking replicates per level | 5 (`for b in range(5)`, seeds 42–46) | 2 (`for seed in range(2)`, seeds 42–43) |
| Mask generation | per-cell loop, one `rng.random()` draw per observed cell | vectorized `rng.random(R_v.shape)` over the full matrix |
| 1PL optimizer | L-BFGS-B, `maxiter` 300 | L-BFGS-B, `maxiter` 200 |

Because the masks and iteration limits differ, the two procedures do not draw the same masked matrices even with the same seeds. The offline refit gives 6/10 (per-level values in `data/stress_refit_levels.csv`, generated with Python 3.11, numpy 2.4.4, scipy 1.17.1; IRT loses at s = 0.5, 0.7, 0.8, 0.9). In both procedures the reference ranking is the full-data 2PL θ (`data/canonical_theta.csv`), i.e. IRT's own estimate, and s = 0 (no masking) is counted as one of the 10 levels. Runtime of `--stress-refit` varies by machine and library version. This does not affect the default verification path, which loads the canonical result directly.

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
- `canonical_stress.json` — canonical stress test results (win count only; no per-level values)
- `derived_rankings.csv` — derived rankings (not read by any script in `src/`)
- `stress_refit_levels.csv` — per-level results of the offline `--stress-refit` procedure (not the canonical run; see the note above)
- `ticker_manifest.json`, `data_provenance.json` — provenance

**Generator status:** `src/experiment.py` writes `canonical_theta.csv` and `canonical_period_params.csv`, but no script in this repository writes the other Path B inputs — `response_matrix.csv`, `period_features.csv`, `canonical_stress.json`, and `derived_rankings.csv`. They are provided as-is and cannot be regenerated from the code here.

## License
MIT. See [LICENSE](LICENSE).
