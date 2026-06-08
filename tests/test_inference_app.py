"""Smoke test for the inference Streamlit app — it imports cleanly and renders
its initial state without exceptions. (file_uploader inputs can't be simulated
via AppTest, so the heavy paths are covered by test_inference_engine.py and the
shared scoring tests.)"""
import os
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.integration
def test_inference_app_renders():
    from streamlit.testing.v1 import AppTest

    cwd = os.getcwd()
    os.chdir(PROJ); sys.path.insert(0, PROJ)
    try:
        at = AppTest.from_file(os.path.join(PROJ, 'inference.py'), default_timeout=120)
        at.run()
        assert not at.exception, f'app raised: {[getattr(e, "value", e) for e in at.exception]}'
        # the upload prompt is shown before any files are provided
        infos = ' '.join(getattr(i, 'value', '') for i in at.info)
        assert 'scorecard' in infos.lower()
        # the "Run inference" button exists (disabled until files are uploaded)
        assert any('Run inference' in b.label for b in at.button)
    finally:
        os.chdir(cwd)
