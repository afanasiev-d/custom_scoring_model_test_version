# Test suite

Unit + integration tests for the credit-scoring pipeline. The guiding principle
is **pin every non-trivial computation to an independent ground truth** rather
than to its own output, and to test the *contracts* a scorecard must satisfy.

## Running

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                     # everything
pytest -m "not slow"       # skip the end-to-end Streamlit AppTest (~fast)
pytest -m integration      # only the cross-module / end-to-end tests
pytest --cov=. --cov-report=term-missing -m "not slow"   # with coverage
```

## Layout

| File | Scope | Highlights |
|---|---|---|
| `conftest.py` | fixtures | a permissive **Streamlit/stqdm stub** so `st`-calling functions run headless; synthetic-data fixtures; Agg backend; per-test figure cleanup |
| `test_bootstrap.py` | unit | AUC vs `roc_auc_score` (exact), KS vs direct CDF, **closed-form jackknife vs brute-force leave-one-out**, BCa invariants + percentile fallback, Gini = exact image of AUC, reproducibility, and a `slow` Monte-Carlo **coverage** check |
| `test_feature_engineering.py` | unit | Box–Cox maps are monotone-increasing; transforms **inherit the source's declared trend**; undeclared predictors produce **no** transforms; negative-value shift; degenerate-input handling |
| `test_preprocessing.py` | unit | numeric coercion, high-cardinality drop, num/cat split, IV (sum identity, signal > noise, no `inf` on pure bins), the `*Match` business guard, geographic/score/sparse filtering |
| `test_woe.py` | unit | smoothed WoE formula, **neutral-NaN contract** (encoding 0 but *fair* WoE retained), shares, direction |
| `test_correlation.py` | unit | Cramér's V (independent ≈ 0, dependent ≈ 1, degenerate = 0); Pearson + Cramér's V redundancy filters keep the stronger predictor |
| `test_scoring.py` | unit + integration | `_cutoff_metrics` (monotone approval, threshold KS), `_ci_panel` (2-dp strings), `_approval_dashboard` (trace set, sign-coloured diffs); controlled `scoring()` asserting **scorecard additivity**, **zero-coefficient drop**, threshold-KS CI |
| `test_eva.py` | unit | KS-lift cumulatives; documents that `eva_dfkslift` keeps the per-row (tie-splitting) KS ≥ threshold KS |
| `test_scorecard_ppt.py` | unit | Excel export: sheet set/order, the **Confidence-intervals sheet** values + 2-dp number format, sheet-name sanitisation |
| `test_integration.py` | integration | a fast **synthetic full pipeline** (split → bin → WoE → fit → score) asserting the scorecard/CI contracts, plus a `slow` **AppTest** driving the real Streamlit app on the bundled example dataset |
| `test_inference_engine.py` | unit | the inference engine: scorecard parse + apply on a hand-built scorecard with **known expected scores** (numerical/categorical/present bins, boundaries, string coercion, missing features), and the PSI / CSI / bad-rate-by-band drift functions |
| `test_inference_app.py` | integration | smoke AppTest that `inference.py` imports and renders its initial state (file uploads can't be simulated via AppTest) |

## Notes

* The Streamlit stub lets us test the *computational* behaviour of `st`-calling
  functions without a server. Functions that touch `st`/`stqdm` take the
  `stub_streamlit` fixture; pure functions don't.
* Synthetic signals are deliberately **moderate** — strong enough to be selected,
  but kept under OptBinning's `IV < 1` "suspiciously strong" cutoff and under the
  Pearson filter's target-correlation threshold.
* Markers: `slow` (full model build / AppTest), `integration` (multi-module).
