"""Access layer for the DVC-tracked data files.

The raw bureau data — ``Example.xlsx`` and ``Full Dictionary.xlsx`` — is versioned
with **DVC** and kept *out of Git*, so it is never downloadable from the
repository (only the small ``*.dvc`` pointer files are committed). The working
copy is materialised with ``dvc pull``.

This helper reads such a file, transparently fetching it via DVC on demand when
it is not already present in the working tree (e.g. on a fresh clone or a cloud
deployment that has the DVC remote configured). When the working copy is already
present — the normal local case after ``dvc pull`` — it is read directly and DVC
is never invoked, so the app does not require DVC to be installed just to run.
"""
import os
import shutil
import subprocess

import pandas as pd


def ensure_local(path):
    """Ensure a DVC-tracked file exists in the working tree, pulling it if needed.

    Returns ``True`` if the file is available afterwards. ``dvc pull`` is only
    attempted when the file is missing, DVC is on the PATH, and the ``.dvc``
    pointer exists (so a configured remote can supply it)."""
    if os.path.exists(path):
        return True
    if shutil.which('dvc') and os.path.exists(path + '.dvc'):
        try:
            subprocess.run(['dvc', 'pull', path], check=False, capture_output=True, timeout=600)
        except Exception:
            pass
    return os.path.exists(path)


def read_excel(path, **kwargs):
    """Read a DVC-tracked Excel data file, fetching it via ``dvc pull`` if absent."""
    if not ensure_local(path):
        raise FileNotFoundError(
            f"'{path}' is DVC-tracked and not available locally. Run `dvc pull` "
            f"after configuring the DVC remote (see the README, 'Data & DVC'), or "
            f"place the file in the project root.")
    return pd.read_excel(path, **kwargs)
