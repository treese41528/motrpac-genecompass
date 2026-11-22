"""The Stage 9 target sum follows the registered formula (deconvolution/calibrate_target_sum.py;
docs/RETRAIN_V2_PREREGISTRATION.md:155-164; docs/audit/remediation/STEP3_PREREG.md section 3)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deconvolution"))
import calibrate_target_sum as c            # noqa: E402


def test_formula_reproduces_the_v2_calibration_of_2026_10_24():
    # LIVER at T0 = 6,500 had median 0.9301966 with v2 medians; the record gives T* = 6352.28, rounded to 6,400
    t_star, t = c.target_sum(0.9301966428756714, 6500.0)
    assert abs(t_star - 6352.275395077238) < 1e-6
    assert t == 6400.0


def test_the_chosen_target_sum_puts_the_liver_median_on_the_corpus_median():
    m, t0 = 0.8784409761428833, 6400.0
    t_star, _ = c.target_sum(m, t0)
    assert abs(math.log2(1 + (2 ** m - 1) * t_star / t0) - c.V2_CORPUS_MEDIAN) < 1e-12


def test_rounding_is_to_the_nearest_hundred():
    assert c.target_sum(c.V2_CORPUS_MEDIAN, 6449.0)[1] == 6400.0
    assert c.target_sum(c.V2_CORPUS_MEDIAN, 6451.0)[1] == 6500.0
