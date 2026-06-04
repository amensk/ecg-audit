import numpy as np
from ecg_audit import net_benefit, decision_curve, subgroup_calibration


def test_net_benefit_treat_all_equivalence():
    # at threshold ~0, predicting all positive => NB ~ prevalence
    y = np.array([0, 0, 1, 1, 1.0])
    nb = net_benefit(y, np.ones(5), 0.001)
    assert abs(nb - y.mean()) < 0.01


def test_decision_curve_keys_and_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.uniform(0, 1, 500) < 0.2).astype(float)
    p = rng.uniform(0, 1, 500)
    dc = decision_curve(y, p)
    assert set(["thresholds", "model_net_benefit", "treat_all", "treat_none",
                "prevalence"]).issubset(dc)
    assert abs(dc["prevalence"] - y.mean()) < 1e-9
    assert len(dc["model_net_benefit"]) == len(dc["thresholds"])


def test_subgroup_calibration_partitions():
    rng = np.random.default_rng(0)
    n = 600
    probs = rng.uniform(0, 1, (n, 1))
    labels = (rng.uniform(0, 1, (n, 1)) < probs).astype(float)
    groups = np.array(["young"] * 300 + ["old"] * 300)
    out = subgroup_calibration(probs, labels, groups, classes=["x"])
    assert set(out) == {"young", "old"}
    assert out["young"]["n"] == 300 and out["old"]["n"] == 300
