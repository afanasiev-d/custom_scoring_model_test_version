"""Shared test fixtures and a lightweight Streamlit stub.

Most pipeline modules (``binning``, ``correlation``, ``scoring``, …) call
``st.*`` and ``stqdm`` directly. Outside a Streamlit run context those calls
would warn and, worse, ``st.columns`` unpacking and ``stqdm`` wrapping behave
unpredictably. The ``stub_streamlit`` fixture replaces them with inert no-ops so
the *computational* behaviour of those functions can be unit/integration tested
deterministically, without spinning up a Streamlit server.

Pure modules (``bootstrap``, ``feature_engineering``, ``woe``, ``preprocessing``
and the ``scoring`` helper functions) need none of this and are tested directly.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")            # headless: never open a GUI window during tests

import numpy as np
import pandas as pd
import pytest

# Make the repo-root app modules importable regardless of the pytest invocation dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────────────────
# Streamlit / stqdm stub
# ──────────────────────────────────────────────────────────────────────────
class _NoOp:
    """A chainable, callable, context-manager, iterable no-op.

    Stands in for any Streamlit element: ``st.write(x)`` returns it, ``col.metric()``
    works, ``with st.status(...) as s: s.update(...)`` works. It is intentionally
    permissive so test code never has to know which Streamlit calls a function makes.
    """
    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(())

    def __getitem__(self, _):
        return self


def _columns(spec, *a, **k):
    """``st.columns(3)`` / ``st.columns([1, 2])`` → a correctly-sized list so that
    ``a, b, c = st.columns(3)`` unpacks cleanly."""
    n = spec if isinstance(spec, int) else len(spec)
    return [_NoOp() for _ in range(n)]


def _passthrough(iterable=None, *a, **k):
    """Stand-in for ``stqdm`` — yields the wrapped iterable unchanged."""
    return iterable if iterable is not None else []


# Streamlit attributes the pipeline modules actually touch.
_ST_NOOP_ATTRS = (
    "write markdown caption subheader header title info success warning error "
    "table dataframe image pyplot plotly_chart metric text json code divider "
    "progress spinner toast latex"
).split()


@pytest.fixture
def stub_streamlit(monkeypatch):
    """Neutralise Streamlit + stqdm so st-calling functions run headless.

    Patches the *streamlit module object* (shared by every ``import streamlit as st``)
    and the ``stqdm`` name in each module that imported it by value.
    """
    import streamlit as st

    for name in _ST_NOOP_ATTRS:
        monkeypatch.setattr(st, name, _NoOp(), raising=False)
    monkeypatch.setattr(st, "columns", _columns, raising=False)
    monkeypatch.setattr(st, "status", lambda *a, **k: _NoOp(), raising=False)
    monkeypatch.setattr(st, "expander", lambda *a, **k: _NoOp(), raising=False)
    monkeypatch.setattr(st, "container", lambda *a, **k: _NoOp(), raising=False)

    # `from stqdm import stqdm` binds the name into each module's namespace, so
    # patch it where it is used rather than the stqdm module itself.
    for mod_name in ("scoring", "binning"):
        if mod_name in sys.modules:
            monkeypatch.setattr(sys.modules[mod_name], "stqdm", _passthrough, raising=False)
    return st


@pytest.fixture(autouse=True)
def _close_figures():
    """Close matplotlib figures after every test (the st-calling pipeline code
    creates many via pyplot and never closes them outside a Streamlit rerun)."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ──────────────────────────────────────────────────────────────────────────
# Synthetic-data fixtures
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture
def rng():
    return np.random.default_rng(20240607)


@pytest.fixture
def binary_scores(rng):
    """A separable-ish (score, label) sample with ties (integer scores), like a
    rounded scorecard: goods score higher, label 1 = bad."""
    n = 1500
    y = (rng.random(n) < 0.28).astype(int)
    score = np.round(rng.normal(520 + (1 - y) * 28, 42, n)).astype(float)
    return score, y


@pytest.fixture
def credit_df(rng):
    """A small raw-ish credit frame exercising the preprocessing quirks:
      * a numeric field stored as strings with '.'/'NULL' placeholders,
      * a high-cardinality free-text column,
      * a geographic column (should be filtered by initial_filtering),
      * a *Match column whose 'Y' is *better* (logical) and one where it is worse,
      * a near-empty sparse column.
    Target ``PI`` (1 = bad).
    """
    n = 600
    y = (rng.random(n) < 0.3).astype(int)
    amount = np.where(y == 1, rng.normal(800, 150, n), rng.normal(1200, 150, n))
    amount_str = np.where(rng.random(n) < 0.05, "NULL", np.round(amount, 2).astype(str))
    df = pd.DataFrame({
        "PI": y,
        "Loan_Amount_str": amount_str,                       # numeric-as-string
        "good_num": rng.normal(50 + (1 - y) * 10, 8, n),     # clean numeric signal
        "free_text": [f"note_{i}" for i in range(n)],        # high-cardinality
        "Applicant_State": rng.choice(["CA", "TX", "NY"], n),  # geographic → dropped
        # logical match: 'Y' → lower bad rate
        "idMatch": np.where(y == 1, rng.choice(["Y", "N"], n, p=[0.3, 0.7]),
                            rng.choice(["Y", "N"], n, p=[0.7, 0.3])),
        # illogical match: 'Y' → HIGHER bad rate (should be dropped by the guard)
        "ssnMatch": np.where(y == 1, rng.choice(["Y", "N"], n, p=[0.75, 0.25]),
                             rng.choice(["Y", "N"], n, p=[0.35, 0.65])),
        "mostly_missing": np.where(rng.random(n) < 0.98, np.nan, 1.0),  # sparse
    })
    return df
