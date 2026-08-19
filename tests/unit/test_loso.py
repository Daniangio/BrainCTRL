from __future__ import annotations

import pytest

from bci.evaluation.loso import loso_folds


def test_loso_folds_hold_out_each_subject_once():
    assert loso_folds([1, 2, 3]) == [(1, [2, 3]), (2, [1, 3]), (3, [1, 2])]


def test_loso_folds_require_at_least_two_subjects():
    with pytest.raises(ValueError, match="at least two"):
        loso_folds([1])
