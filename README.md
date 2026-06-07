<h1 align="center">Credit Scoring — Custom Model Studio</h1>

<p align="center">
  <em>An interpretable, end-to-end application credit scorecard engine — from raw bureau data to a points-based, regulator-friendly scorecard — built on Weight-of-Evidence logistic regression, mathematically-optimal binning, and Bayesian hyperparameter search.</em>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.13-blue">
  <img alt="streamlit" src="https://img.shields.io/badge/UI-Streamlit-ff4b4b">
  <img alt="optbinning" src="https://img.shields.io/badge/binning-OptBinning%20(MIP)-1FB8A6">
  <img alt="optuna" src="https://img.shields.io/badge/HPO-Optuna%20(TPE)-0A2540">
  <img alt="license" src="https://img.shields.io/badge/methodology-WoE%2FIV%20scorecard-16A34A">
</p>

---

## 1. Overview

This repository implements a **complete probability-of-default (PD) scorecard pipeline** wrapped in an interactive Streamlit studio. It takes a labelled credit dataset and produces a **points-based scorecard** that a credit risk team can read, challenge, and deploy — together with the full battery of diagnostics (binning tables, WoE/IV, KS/AUC/Gini, score distributions, and a performance-projection / approval-strategy table).

The design philosophy follows the discipline of modern scorecard development (Siddiqi): **every modelling choice is made to maximise predictive power *subject to* interpretability, monotonicity, and business plausibility** — not raw discrimination at any cost. A scorecard that a risk officer cannot explain, or that contradicts business intuition, is worthless regardless of its Gini.

The scorecard itself is a **constrained, interpretable generalized linear model**: a logistic regression on Weight-of-Evidence–transformed, optimally-binned features, where each attribute contributes a fixed, additive number of points. This is the canonical form used across the industry precisely because it is *transparent by construction*.

---

## 2. Why these methods? (design rationale & trade-offs)

This section is the heart of the project — the *reasoning*, not just the recipe.

### 2.1 Weight of Evidence (WoE) encoding instead of one-hot dummies
A scorecard must be **linear in the log-odds** and additive in points. WoE achieves exactly this: it re-expresses every binned attribute on the log-odds scale, so a single logistic-regression coefficient per characteristic suffices.

$$
\text{WoE}_i = \ln\left(\frac{g_i / G}{b_i / B}\right), \qquad G = \sum_i g_i, \quad B = \sum_i b_i
$$

where `g_i`, `b_i` are the counts of *goods* and *bads* in bin `i`. We apply **Laplace (additive) smoothing** to avoid divergence on pure bins:

$$
\text{WoE}_i = \ln\left(\frac{(g_i + \alpha)/G}{(b_i + \alpha)/B}\right), \qquad \alpha = 0.5
$$

**Trade-off.** WoE discards within-bin variation and collapses each characteristic to one degree of freedom. We accept this loss deliberately: it linearises the relationship, neutralises outliers, gives one interpretable coefficient per feature, and makes regularisation operate at the *characteristic* level rather than the dummy level. One-hot encoding would re-introduce dozens of collinear, individually-unstable coefficients and destroy the additive points structure.

### 2.2 Mathematically-optimal binning (OptBinning, MIP formulation)
Coarse classing is the single most consequential step in scorecard development. Rather than ad-hoc equal-width / equal-frequency binning, we use **OptBinning** (Navas-Palencia), which casts coarse-classing as a **mixed-integer programming** problem: merge pre-bins to **maximise Information Value** *subject to* a prescribed **monotonic event-rate trend**, minimum bin support, and a maximum number of bins.

**Why monotonicity is a feature, not a restriction.** Enforcing a monotone WoE/event-rate trend is simultaneously (a) a **regularizer** — it forbids the model from chasing non-monotone noise in sparse bins, and (b) a **business/regulatory requirement** — risk must move in one direction with the characteristic (e.g. more delinquencies ⇒ higher risk). The MIP formulation guarantees a *globally optimal* binning under these constraints rather than a greedy local one.

**Engineering trade-off — `mip` over `cp`.** OptBinning offers a CP-SAT (`cp`) and a MIP (`mip`) solver. We use `mip`: in repeated/concurrent invocation the CP-SAT backend exhibited a cumulative threading deadlock, whereas `mip` is stable and thread-safe — which is what makes Step 4 *parallelisable* (see §2.6). We also **round numerical split points to one decimal** so the resulting intervals are human-readable on the scorecard, accepting a negligible deviation from the raw optimal cut-points in exchange for interpretability.

### 2.3 Neutral treatment of missing values
Missing data carries *no information we are willing to price*. A client with a missing attribute is scored **neutrally**: the `NaN` bin is encoded with **WoE = 0**, contributing **0 points**, while its *empirical* WoE and population share are still displayed for transparency.

**Reasoning.** Letting the model assign points to missingness invites it to exploit a data-collection artifact that may not generalise (and can be a fair-lending hazard). We therefore separate *what the data says* about missingness (shown) from *what the model is allowed to act on* (nothing).

### 2.4 Two-layer redundancy control
Multicollinearity destabilises coefficients and inflates points. We filter redundancy with the right tool for each data type:

- **Cramér's V** (χ²-based, Bergsma bias-corrected) on the *binned categorical* characteristics — the appropriate association measure for nominal variables:

$$
V = \sqrt{\frac{\chi^2 / n}{\min(r-1,\, k-1)}}
$$

- **Pearson correlation** on the *WoE-transformed* characteristics (now continuous).

In each highly-associated pair we retain the member **more strongly associated with the target**, mirroring the standard "keep the stronger predictor" rule.

### 2.5 Business-logic guard for `*Match` features
Domain knowledge overrides statistics when they conflict. For binary identity-`*Match` characteristics (e.g. `nameAddressMatch`), a *match* must imply *lower* risk. Any `*Match` feature whose **match category has a higher bad rate than its no-match category** is dropped automatically — it contradicts business logic and would not survive expert review.

### 2.6 Parallelism where it pays
Coarse-classing fits one optimisation per characteristic over potentially hundreds of features — the pipeline bottleneck. We parallelise it with a **thread pool**: the heavy OptBinning/scikit-learn work runs in C/C++ that releases the GIL, so threads deliver genuine speed-up *without* the serialization and process-spawn overhead (and the macOS/Streamlit pitfalls) of multiprocessing. The same reasoning underpins Optuna's threaded trials and the parallel feature-engineering step.

### 2.7 Regularised logistic regression + Bayesian search
The scorecard GLM is fit with **elastic-net logistic regression** (L1 / L2 / L1+L2, `saga` solver). The hyperparameters — penalty type, `C`, and `l1_ratio` — are tuned with **Optuna's TPE sampler** (Tree-structured Parzen Estimator), far more sample-efficient than grid search over a sparse, log-scaled space.

The objective is **overfit-aware**, not naïve discrimination (`m` is KS or AUC):

$$
\text{maximise} \quad m_{\text{val}} - \left| m_{\text{train}} - m_{\text{val}} \right|
$$

penalising the train/validation gap. With k-fold cross-validation enabled, the objective becomes **robustness-aware** — rewarding a high *and stable* out-of-fold metric:

$$
\text{maximise} \quad \overline{m}_{\text{folds}} - \sigma\left(m_{\text{folds}}\right)
$$

### 2.8 Uncertainty quantification — bootstrap confidence intervals (BCa)
A single KS / AUC / Gini number is a *point estimate*; on a finite sample it has sampling error that a risk committee must see before trusting a cut-off. We therefore report a **confidence interval** around each discrimination metric, built to a high statistical standard:

- **Stratified resampling.** KS and AUC are *two-sample* rank statistics (goods vs. bads), so we resample goods and bads **independently with replacement, preserving the original class counts** `(n_good, n_bad)`. This conditions on the class design exactly as the metrics do — the standard scheme (e.g. `pROC`'s default for AUC) — and avoids the degenerate, unbalanced resamples a pooled bootstrap can produce.
- **BCa intervals.** We use the **bias-corrected and accelerated** interval (Efron, 1987), which is *second-order accurate* and *transformation-respecting* — the gold standard over the naïve percentile interval. The endpoints are adjusted percentiles

$$
\alpha_1 = \Phi\!\left( z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a\,(z_0 + z_{\alpha/2})} \right), \qquad
\alpha_2 = \Phi\!\left( z_0 + \frac{z_0 + z_{1-\alpha/2}}{1 - a\,(z_0 + z_{1-\alpha/2})} \right)
$$

  with bias-correction `z₀` read from the bootstrap distribution and acceleration `a` from the empirical jackknife.
- **Exact, fast jackknife.** Rather than `n` brute-force re-fits, the leave-one-out values driving `a` are computed in **closed form** — for AUC via the per-point Mann–Whitney placements, for KS via prefix/suffix running maxima of the (shifted) CDF gap — so the acceleration is *exact* at `O(n \log n)`.
- **Gini consistency.** Since `Gini = 2·AUC − 1` is a strictly increasing transform, its interval is the **exact image** of the AUC interval (CIs are equivariant under monotone transforms) — the AUC and Gini intervals can never disagree.

The implementation is validated against ground truth: point estimates match scikit-learn exactly, the closed-form jackknives match brute-force leave-one-out to machine precision, and a Monte-Carlo simulation confirms the 95 % interval attains ≈ 95 % coverage.

---

## 3. Mathematical foundations (reference card)

| Quantity | Definition | Role |
|---|---|---|
| **WoE** | `ln( ((gᵢ+α)/G) / ((bᵢ+α)/B) )` | feature encoding, log-odds scale |
| **Information Value** | `IV = Σᵢ (gᵢ/G − bᵢ/B) · WoEᵢ` | univariate predictive strength & selection |
| **IV strength bands** | `<0.02` unpredictive · `0.02–0.1` weak · `0.1–0.3` medium · `0.3–0.5` strong · `>0.5` suspicious | Siddiqi heuristics |
| **Score scaling** | `Factor = PDO / ln(2)` · `Offset = Target − Factor·ln(TargetOdds)` | calibrate points |
| **Points (attribute)** | `−Factor · βⱼ · WoEⱼ` | additive scorecard points |
| **Base points** | `Offset − Factor · β₀` | intercept contribution |
| **KS** | `maxₛ │F_good(s) − F_bad(s)│` | rank-ordering / separation |
| **Gini** | `2·AUC − 1` | discrimination |

The final borrower score is the simple additive sum

$$
\text{Score} = \underbrace{\left(\text{Offset} - \text{Factor}\cdot\beta_0\right)}_{\text{base points}} + \sum_{j} \left(-\,\text{Factor}\cdot\beta_j\cdot\text{WoE}_j\right)
$$

so higher scores correspond to lower risk, and **PDO** (points-to-double-the-odds) makes the scale economically interpretable — the canonical scaling of Siddiqi.

---

## 4. Pipeline architecture

The studio executes seven transparent, individually-narrated stages:

```
1. Data            →  load, profile, sparsity/ID/geography filtering, numeric coercion, high-cardinality control
2. Split           →  numerical vs. categorical sub-datasets
3. Feature Eng.    →  (optional) Box–Cox / power transforms, λ ∈ [−2, 2] + log, trend-inferred, parallel
4. Binning         →  OptBinning (MIP, monotone), IV selection, per-feature binning charts  ── parallel
5. WoE Encoding    →  business-logic guard · Cramér's V · WoE transform · Pearson filter
6. Model           →  elastic-net LR · Optuna (TPE) · optional k-fold CV · KS/AUC objective  ── multi-thread
7. Scoring         →  scaling, scorecard, KS/AUC/Gini + bootstrap (BCa) CIs, score distribution, approval-strategy (PPT)
```

### Module map
| File | Responsibility |
|---|---|
| `main.py` | Streamlit orchestration, step UI, session-state results, downloads |
| `preprocessing.py` | filtering, numeric coercion, high-cardinality & `*Match` business guards, vectorised IV |
| `feature_engineering.py` | parallel Box–Cox/power transforms with monotonic-trend inference |
| `binning.py` | parallel OptBinning, IV selection, interpretable rounded bounds, binning plots |
| `woe.py` | smoothed WoE encoding, neutral-`NaN`, WoE map for the scorecard |
| `correlation.py` | Pearson (WoE) + Cramér's V (categorical) redundancy filtering & heatmaps |
| `model.py` | elastic-net LR, Optuna TPE + CV, KS helpers, Optuna & performance visualisations |
| `scoring.py` | scaling, WoE scorecard, metrics, bootstrap-CI panel, score distribution, performance projection table |
| `bootstrap.py` | stratified-bootstrap BCa confidence intervals for KS / AUC / Gini (exact closed-form jackknife) |
| `scorecard_ppt.py` | richly-formatted Excel export (tables + embedded charts) |
| `viz.py` | shared fintech visual identity, plot gallery, table styler |

---

## 5. Installation

```bash
git clone <your-repo-url>
cd custom_scoring_model_test_version

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run main.py
```

> **Note on dependencies.** `scikit-learn` is pinned `< 1.8` because the current `optbinning` still calls the `force_all_finite` argument removed in scikit-learn 1.8. The stack targets Python 3.13 with NumPy 2 / pandas 3.

---

## 6. User guide

### Step 0 — configure (left sidebar)
| Control | Meaning |
|---|---|
| **Project name / Target name** | report label and the exact 0/1 target column (1 = bad/default) |
| **Upload data** | CSV or Excel; or use the built-in example dataset |
| **Sparse threshold** | drop characteristics with missing rate above this % |
| **Minimum IV** | univariate selection floor (default 0.01) |
| **Max paired correlation** | Pearson cut-off for WoE features |
| **Max categorical association** | Cramér's V cut-off for binned categoricals |
| **Max distinct values** | high-cardinality cut-off (genuine free-text fields are dropped) |
| **Apply power transformations** | enable Step 3 feature engineering (off by default) |
| **Optimize on** | KS *or* AUC ROC as the tuning objective |
| **k-fold cross-validation** | robustness-oriented tuning with `k ∈ [2, 8]` |
| **Scoring parameters** | Target score, Target odds, Points to double the odds (PDO) |
| **Confidence level / Bootstrap resamples** | the level (90/95/99 %) and number of stratified resamples for the BCa confidence intervals on KS/AUC/Gini |

### Step 1 — review & curate predictors
After upload, inspect the data preview and the automatic filtering report (numeric coercions, dropped geographic / high-cardinality fields). In the **predictor configuration form** you may (optionally) add external characteristics with a known ascending/descending event-rate trend and exclude inappropriate ones. *Nothing heavy runs until you press* **🚀 Build model** — the form batches your choices so the app stays responsive.

### Steps 2–7 — build
Press **🚀 Build model**. Each stage streams its tables, progress bars, and charts inside its own status panel, so you watch the scorecard being assembled — split → (optional) feature engineering → binning (with per-feature WoE charts) → WoE/redundancy filtering → Optuna search (with optimisation-history, importance, parallel-coordinate, slice & contour plots) → scoring.

### Step 8 — consume the results
At the end you get:
- **📥 Download Current Results** — a formatted **Excel workbook**: Scorecard, **Confidence-intervals sheet** (KS/AUC/Gini with their BCa estimate, lower/upper bounds, bootstrap SE and method provenance), Performance-Projection Table, Missing-rate, Initial IV, one **binning sheet per characteristic with its binning chart embedded**, and a Visualizations sheet (which also embeds the bootstrap-CI forest plot).
- **📊 Download All Visualizations (ZIP)** — every chart as a high-resolution PNG.
- **🔄 Restart** — the *only* control that resets the run; downloading **never** restarts the app (results persist via session state).

---

## 7. Outputs in detail

- **Scorecard** — `Feature · Category · WoE · Share (%) · Points`, intercept first, grouped by characteristic; `NaN` rows always present with a neutral 0-point contribution and their fair WoE/share.
- **Performance Projection Table (PPT)** — for every cut-off score: approval rate, marginal & cumulative good rate, odds for accepted/rejected populations — i.e. the **approval-strategy curve** used to set a cut-off policy.
- **Diagnostics** — KS / AUC / Gini **with bootstrap (BCa) confidence intervals** (a forest plot + table quantifying the sampling uncertainty of each metric), score distribution by outcome with the optimal cut-off, ROC, KS-separation curve, correlation & association heatmaps, and the full Optuna search visualisations.

---

## 8. Limitations & roadmap

This is a **research / decisioning studio**, not a turnkey production engine. Honest caveats and natural extensions:

- **Validation depth.** Reporting is on a single train/test split (with optional CV during tuning) and the discrimination metrics now carry **bootstrap (BCa) confidence intervals** (§2.8). Out-of-time validation and **Population Stability Index (PSI)** monitoring are the natural next steps. *Note:* the CIs are computed on the same (full) sample as the headline metrics, so they quantify sampling — not out-of-time — uncertainty.
- **Reject inference.** The model is trained on accepts only; a production build would incorporate reject inference (e.g. parcelling / fuzzy augmentation).
- **Fairness/compliance.** Geographic proxies are filtered, but a full disparate-impact analysis is out of scope here.
- **Calibration.** Scaling is to a target odds/PDO; explicit probability calibration (e.g. isotonic) could be added if calibrated PDs are required.

---

## 9. References

The methodology draws directly on the following sources; concepts borrowed from each are noted.

1. **N. Siddiqi** — *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*, Wiley, 2006.
   → Weight-of-Evidence and Information Value, coarse-classing, the points/odds **scaling** (Factor, Offset, PDO), IV strength bands, and the primacy of **business validation** of every characteristic.
2. **N. Siddiqi** — *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards*, 2nd ed., Wiley & SAS Business Series, 2017.
   → Modern scorecard-development workflow, monotonic binning rationale, model governance and interpretability standards.
3. **D. J. Bolder** — *Credit-Risk Modelling: Theoretical Foundations, Diagnostic Tools, Practical Examples, and Numerical Recipes in Python*, Springer, 2018.
   → Theoretical underpinnings of PD modelling, diagnostic/evaluation tooling, and numerically-sound Python implementation practices.
4. **G. Navas-Palencia** — *Optimal binning: mathematical programming formulation*, 2020. [arXiv:2001.08025](https://arxiv.org/pdf/2001.08025)
   → The mixed-integer programming formulation of optimal binning with monotonicity constraints (the engine behind Step 4 via `optbinning`).
5. **B. Efron** — *Better Bootstrap Confidence Intervals*, Journal of the American Statistical Association, 82(397), 1987; and **B. Efron & R. Tibshirani**, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
   → The **BCa** (bias-corrected and accelerated) interval and the jackknife estimate of acceleration used for the KS/AUC/Gini confidence intervals (§2.8).

*Supporting tooling:* OptBinning (binning), Optuna — TPE Bayesian optimisation (hyperparameter search), scikit-learn (elastic-net logistic regression, `saga`), Streamlit, Plotly, seaborn/matplotlib.

---

<p align="center"><sub>Built with a risk-modeller's bias: interpretability and business plausibility first, discrimination second.</sub></p>
