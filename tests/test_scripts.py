from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rs():
    return _load("run_simulations")


def test_seed_is_stable_across_processes_and_distinct_across_tasks(rs):
    assert rs._seed("a") == 454468563
    assert rs._seed("fig2", "fig2", 0.01, 0.01) == rs._seed("fig2", "fig2", 0.01, 0.01)
    assert rs._seed("fig2", "fig2", 0.01, 0.01) != rs._seed("fig2", "fig2", 0.02, 0.01)
    assert 0 <= rs._seed("x") < 2**32


def test_u_shape_pattern(rs):
    pat = np.asarray(rs.u_shape(390, 3.0))
    assert pat.shape == (390,)
    assert pat.mean() == pytest.approx(1.0)
    assert pat.min() > 0.0
    assert pat[0] > pat[195] and pat[-1] > pat[195]


def test_build_tasks_counts(rs):
    panel, path = rs.build_tasks(["fig2", "fig3", "table2", "robust", "negatives"])
    assert len(path) == 15
    assert len(panel) == 78
    assert len(rs.build_tasks(["fig3"])[0]) == 11
    assert len(rs.build_tasks(["robust"])[0]) == 44
    assert all(t[1].prob == 1.0 for t in rs.build_tasks(["fig3"])[0])


def test_summarize_panel(rs):
    df = pd.DataFrame({"EDGE": [0.01, -0.01, np.nan, 0.03]})
    s = rs.summarize_panel(df).iloc[0]
    assert s["estimator"] == "EDGE"
    assert s["mean"] == pytest.approx(np.mean([0.01, 0.0, 0.03]))
    assert s["nan_frac"] == pytest.approx(1 / 4)


def test_negatives_summary(rs):
    df = pd.DataFrame({"EDGE": [0.01, -0.01], "AR": [0.02, 0.01]})
    out = rs.negatives_summary(df, {"prob": 1.0}).set_index("estimator")
    assert out.loc["EDGE", "rms_signed"] == pytest.approx(0.0)
    assert out.loc["EDGE", "neg_frac"] == 0.5
    assert out.loc["EDGE", "mean_clip_zero"] == pytest.approx(0.005)
    assert out.loc["AR", "rms_signed"] == pytest.approx(np.sqrt(np.mean([0.0004, 0.0001])))
    assert out.loc["AR", "prob"] == 1.0
