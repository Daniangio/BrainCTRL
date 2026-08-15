from __future__ import annotations

import pytest


@pytest.mark.lsl
def test_lsl_import_smoke():
    pytest.importorskip("mne_lsl")
