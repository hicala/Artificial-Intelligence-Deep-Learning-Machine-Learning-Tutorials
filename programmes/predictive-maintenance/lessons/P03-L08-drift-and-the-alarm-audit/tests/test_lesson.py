"""Autograder rubric for P03-L08. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* `test_diagnose_checks_the_instrument_before_duty` is the only case where the duty rule and
  the instrument rule both fire. Every other diagnosis case passes under either order, and
  the wrong order conditions on readings a transmitter got wrong — so it is graded alone.
* `test_estimate_prices_keeps_the_guess_for_an_empty_class` is what separates a fallback from
  a zero. An implementation that averages an empty list scores full marks on every other
  price case and prices a missed failure at nothing, which makes "never alarm" optimal.
* `test_apply_fix_reprice_reads_the_history_not_the_window` hands `apply_fix` a history whose
  cheap threshold is the opposite of the scored window's. An implementation that re-derives
  in-sample is otherwise perfect and is the commonest way this whole module is defeated.
* `test_apply_fix_feature_fixes_keep_the_threshold` exists because an implementation that
  re-derives under every fix scores well everywhere and turns section 9's matrix into a
  comparison of one thing with itself.
* `test_conditional_psi_weights_by_the_reference_mix` checks an exact number built from the
  candidate's OWN `psi` and `reference_bins`, so any correct implementation passes and the
  three plausible weightings (current mix, unweighted mean, pooled bins) do not.
* `test_adjust_feature_leaves_a_clean_channel_alone` is the property that makes re-baselining
  the wrong answer to ageing. Nothing else in the rubric asks for it.
* `test_reference_bins_cuts_the_mass_not_the_range` uses a skewed sample, because on a
  uniform one equal-width and equal-mass bins agree and the bug is invisible.
* the rubric grades observable behaviour, never one particular call. `psi` is checked for the
  values the docstring promises, not for `np.histogram` or `np.searchsorted`.
* the two integration cases at the end measure the lesson's claim on the generated fleet
  rather than asserting it: the recommended fix is the cheapest response to its own drift,
  and the fix belonging to another diagnosis costs measurably more.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

Baseline = lesson.Baseline
Prices = lesson.Prices
Counts = lesson.Counts
Run = lesson.Run
Window = lesson.Window
DriftSignature = lesson.DriftSignature
AuditEntry = lesson.AuditEntry
FAILURE, CENSORED = lesson.FAILURE, lesson.CENSORED
FIXES, DIAGNOSES, OUTCOMES = lesson.FIXES, lesson.DIAGNOSES, lesson.OUTCOMES

_BAND = None


def band() -> float:
    """The measured alert level, computed once with the candidate's own psi."""
    global _BAND
    if _BAND is None:
        _BAND = lesson.null_psi_band(lesson.REFERENCE)
    return _BAND


def toy_window(health, regime=None, check=None, runs=None, name="toy") -> Window:
    h = np.asarray(health, dtype=float)
    regime = np.zeros_like(h, dtype=int) if regime is None else np.asarray(regime)
    check = np.ones((h.shape[0], 2)) if check is None else np.asarray(check, dtype=float)
    runs = tuple(Run(i, h.shape[1], CENSORED) for i in range(h.shape[0])) if runs is None \
        else tuple(runs)
    return Window(name, h, regime, check, runs)


# --- exercise 1: reference_bins ---------------------------------------------------------

def test_reference_bins_shape_and_infinite_edges():
    got = lesson.reference_bins(np.array([0.0, 1.0, 2.0, 3.0]), 2)
    assert np.array_equal(got, np.array([-np.inf, 1.5, np.inf])), (
        f"expected [-inf, 1.5, inf], got {got}"
    )
    four = lesson.reference_bins(np.arange(101.0), 4)
    assert four.size == 5, f"n_bins=4 gives 5 edges on a tie-free reference, got {four.size}"
    assert np.isneginf(four[0]) and np.isposinf(four[-1]), (
        "the outer edges must be -inf and +inf; a finite outer edge drops every later "
        "reading that falls past it, which is exactly the drift you are looking for"
    )
    assert np.all(np.diff(four) > 0), "the edges must come back strictly increasing"
    assert lesson.reference_bins(np.arange(101.0).reshape(101, 1), 4).size == 5, (
        "flatten the input: the reference arrives as a (units x hours) matrix"
    )


def test_reference_bins_cuts_the_mass_not_the_range():
    skew = lesson.reference_bins(np.array([0.0, 0.0, 0.0, 0.0, 100.0]), 2)
    assert np.isclose(skew[1], 0.0), (
        f"the median of [0,0,0,0,100] is 0.0, so that is the single interior cut; got "
        f"{skew[1]}. 50.0 means you cut the RANGE in half instead of the MASS, and on a "
        "skewed feature that puts 80% of the reference in one bin"
    )
    quart = lesson.reference_bins(np.array([1.0] * 30 + [2.0] * 30 + [50.0] * 40), 4)
    assert np.allclose(quart[1:-1], [1.0, 2.0, 50.0]), (
        f"the quartiles of this sample are 1, 2 and 50; got {quart[1:-1]}"
    )


def test_reference_bins_collapses_ties_and_validates():
    tied = lesson.reference_bins(np.array([5.0, 5.0, 5.0, 5.0]), 4)
    assert np.array_equal(tied, np.array([-np.inf, 5.0, np.inf])), (
        f"a reference with one repeated value has one distinct cut point; got {tied}. "
        "Duplicate edges make a zero-width bin and the PSI divides by its count"
    )
    part = lesson.reference_bins(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 9.0]), 4)
    assert np.all(np.diff(part) > 0) and part.size < 5, (
        f"three of this sample's quartiles tie at 0.0, so the result has fewer than 5 edges "
        f"and is still strictly increasing; got {part}"
    )
    for bad_v, bad_n in ((np.array([]), 4), (np.array([1.0, 2.0]), 1),
                         (np.array([1.0, 2.0]), 0), (np.array([1.0, 2.0]), -3),
                         (np.array([1.0, np.nan]), 4), (np.array([1.0, np.inf]), 4)):
        try:
            lesson.reference_bins(bad_v, bad_n)
        except ValueError:
            continue
        raise AssertionError(f"reference_bins({bad_v.tolist()}, {bad_n}) must raise ValueError")


# --- exercise 2: psi ---------------------------------------------------------------------

E2 = np.array([-np.inf, 0.5, np.inf])


def test_psi_value_and_zero():
    same = lesson.psi(np.array([0.0, 1.0]), np.array([0.0, 1.0]), E2)
    assert np.isclose(same, 0.0, atol=1e-9), f"identical samples have PSI 0.0, got {same!r}"
    moved = lesson.psi(np.array([0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0]), E2)
    assert np.isclose(moved, 0.274653, atol=1e-5), (
        f"50/50 against 75/25 is 0.274653, got {moved!r}. 2.197 means you summed over COUNTS "
        "instead of proportions; 0.131 is the one-sided KL divergence, which is not symmetric"
    )
    three = lesson.psi(np.array([0.0, 1.0, 2.0] * 10), np.array([0.0, 0.0, 1.0, 2.0] * 10),
                       np.array([-np.inf, 0.5, 1.5, np.inf]))
    assert np.isclose(three, 0.115525, atol=1e-5), (
        f"expected 0.115525 on three bins moving from 1/3,1/3,1/3 to 0.5,0.25,0.25; got "
        f"{three!r}. Two bins is not enough to catch an implementation that only ever looks "
        "at the first and last"
    )


def test_psi_is_symmetric_and_scale_free():
    a, b = np.array([0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])
    assert np.isclose(lesson.psi(a, b, E2), lesson.psi(b, a, E2), atol=1e-9), (
        "PSI is symmetric in its two samples; an asymmetric answer means one of the two "
        "proportion arrays is in the wrong place in the sum"
    )
    assert np.isclose(lesson.psi(np.zeros(10), np.zeros(5000), E2), 0.0, atol=1e-9), (
        "two samples of very different size that fill the bins the same way have PSI 0 — "
        "proportions, not counts"
    )


def test_psi_floors_an_empty_bin():
    empty = lesson.psi(np.array([0.0, 1.0]), np.array([0.0, 0.0]), E2)
    assert np.isfinite(empty), (
        f"a bin the current window never fills must be FLOORED, not left to take log(0); got "
        f"{empty!r}"
    )
    assert empty > 1.0, (
        f"an emptied bin is the loudest drift there is and must dominate the sum; got "
        f"{empty!r}. A small number means the bin was skipped rather than floored"
    )
    coarse = lesson.psi(np.array([0.0, 1.0]), np.array([0.0, 0.0]), E2, floor=1e-2)
    assert coarse < empty, (
        "floor is a parameter: a larger floor gives a smaller answer on an emptied bin"
    )


def test_psi_bins_are_half_open_and_validated():
    on_edge = lesson.psi(np.array([0.5] * 10), np.array([0.0] * 10), E2)
    assert on_edge > 1.0, (
        "bin i is [edges[i], edges[i+1]), so 0.5 sits in the upper bin and 0.0 in the lower; "
        "if this came back 0 your comparison at the edge is the wrong way round"
    )
    for bad in (np.array([-np.inf, np.inf]), np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.5]), np.array([1.0])):
        try:
            lesson.psi(np.array([0.0]), np.array([0.0]), bad)
        except ValueError:
            continue
        raise AssertionError(f"edges={bad.tolist()} must raise ValueError")
    for r, c, f in ((np.array([]), np.array([0.0]), 1e-6),
                    (np.array([0.0]), np.array([]), 1e-6),
                    (np.array([0.0]), np.array([0.0]), 0.0),
                    (np.array([0.0]), np.array([0.0]), -1e-6)):
        try:
            lesson.psi(r, c, E2, f)
        except ValueError:
            continue
        raise AssertionError(f"psi({r.tolist()}, {c.tolist()}, e, floor={f}) must raise")


# --- exercise 3: conditional_psi ----------------------------------------------------------

R0 = np.linspace(0.0, 1.0, 600)
R1 = np.linspace(5.0, 6.0, 400)
R_VAL = np.concatenate([R0, R1])
R_REG = np.concatenate([np.zeros(600, int), np.ones(400, int)])
C0 = np.linspace(0.0, 1.0, 120)
C1 = np.linspace(5.0, 6.0, 900)
C_VAL = np.concatenate([C0, C1])
C_REG = np.concatenate([np.zeros(120, int), np.ones(900, int)])


def test_conditional_psi_explains_a_pure_regime_change():
    same = lesson.conditional_psi(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0, 0, 1, 1]),
                                  np.array([0.0, 0.0, 1.0, 1.0]), np.array([0, 0, 1, 1]), 2)
    assert np.isclose(same, 0.0, atol=1e-9), f"nothing moved, so the answer is 0.0; got {same!r}"
    cond = lesson.conditional_psi(R_VAL, R_REG, C_VAL, C_REG, 3)
    marg = lesson.psi(R_VAL, C_VAL, lesson.reference_bins(R_VAL, 3))
    assert cond < 0.05 * marg, (
        f"within each regime these two windows are identical and only the MIX moved, so "
        f"conditioning must make the drift vanish: conditional {cond:.4f} against marginal "
        f"{marg:.4f}. If they are close you binned the pooled reference, or never split at all"
    )
    # WHOSE quantiles. With one regime, conditional_psi IS psi over that regime's reference
    # bins, so the candidate's own psi and reference_bins pin the bin source exactly without
    # pinning how it is written. These two samples overlap and are shaped differently, which
    # the far-apart pair in the weighting case below is not: binning the CURRENT readings
    # there happens to give the same number by symmetry, and it does not here.
    ref_one = np.concatenate([np.linspace(0.0, 0.4, 500), np.linspace(0.9, 1.0, 100)])
    cur_one = np.concatenate([np.linspace(0.0, 0.1, 100), np.linspace(0.3, 1.0, 500)])
    flat = np.zeros(600, dtype=int)
    one = lesson.conditional_psi(ref_one, flat, cur_one, flat, 4)
    want_one = lesson.psi(ref_one, cur_one, lesson.reference_bins(ref_one, 4))
    assert np.isclose(one, want_one, rtol=1e-9), (
        f"with a single duty point the answer is exactly psi over bins built from the "
        f"REFERENCE readings at that duty point: expected {want_one:.4f}, got {one:.4f}. "
        f"Building each regime's bins from the CURRENT readings instead gives "
        f"{lesson.psi(ref_one, cur_one, lesson.reference_bins(cur_one, 4)):.4f} — bins that "
        "move with the drift, which is the neatest way to build a detector that reports "
        "nothing has happened"
    )


def test_conditional_psi_weights_by_the_reference_mix():
    c0_far = np.linspace(90.0, 91.0, 120)
    got = lesson.conditional_psi(R_VAL, R_REG, np.concatenate([c0_far, C1]), C_REG, 3)
    psi_0 = lesson.psi(R0, c0_far, lesson.reference_bins(R0, 3))
    psi_1 = lesson.psi(R1, C1, lesson.reference_bins(R1, 3))
    expected = (600 * psi_0 + 400 * psi_1) / 1000
    assert np.isclose(got, expected, rtol=1e-8), (
        f"expected {expected:.4f} — 0.6 of regime 0's PSI plus 0.4 of regime 1's, and each "
        f"regime binned on its own reference readings — got {got:.4f}. Weighting by the "
        f"CURRENT mix gives about {(120 * psi_0 + 900 * psi_1) / 1020:.4f} and buries the "
        f"drift in the regime the plant has stopped using; an unweighted mean over regimes "
        f"gives about {(psi_0 + psi_1) / 2:.4f}"
    )
    # A regime the REFERENCE has and the current window does not is skipped, and the weight
    # it would have carried goes to the regimes that are actually compared. Here regime 1 is
    # absent from the current window, so regime 0 carries all of it.
    gone = lesson.conditional_psi(R_VAL, R_REG, c0_far, np.zeros(120, dtype=int), 3)
    assert np.isclose(gone, psi_0, rtol=1e-9), (
        f"regime 1 has no readings in this current window, so it is skipped and regime 0 "
        f"carries the whole weight: expected {psi_0:.4f}, got {gone:.4f}. Dividing by the "
        "number of regimes in the reference, rather than by the weight you actually "
        "accumulated, charges the average for a comparison that never happened"
    )


def test_conditional_psi_keeps_a_within_regime_shift():
    shifted = lesson.conditional_psi(R_VAL, R_REG, R_VAL + 3.0, R_REG, 3)
    assert shifted > 1.0, (
        f"every reading moved WITHIN its own regime, which is the one drift conditioning "
        f"must not explain away; got {shifted:.4f}"
    )
    half = lesson.conditional_psi(R_VAL, R_REG,
                                  np.concatenate([R0 + 3.0, R1]), R_REG, 3)
    assert 0.3 * shifted < half < shifted, (
        f"with only regime 0 shifted the answer is a fraction of the all-shifted one, not "
        f"zero and not the same; got {half:.4f} against {shifted:.4f}"
    )


def test_conditional_psi_rejects_an_unseen_regime_and_validates():
    # The current window holds a regime the reference DOES have, with plenty of readings, and
    # one it does not. An implementation that only loops over the reference's regimes quietly
    # drops the new duty point and returns a perfectly ordinary number for the rest.
    mixed_val = np.concatenate([C0, np.full(200, 99.0)])
    mixed_reg = np.concatenate([np.zeros(120, int), np.full(200, 5)])
    try:
        lesson.conditional_psi(R_VAL, R_REG, mixed_val, mixed_reg, 3)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a duty point the reference never contained must raise ValueError even when the "
            "other regimes look fine. There is no healthy baseline to condition on, and "
            "looping over the reference's regimes alone reports a quiet plant at exactly the "
            "moment the plant starts doing something it has never done")
    try:
        lesson.conditional_psi(R_VAL, R_REG, np.array([1.0]), np.array([7]), 3)
    except ValueError:
        pass
    else:
        raise AssertionError("an entirely unseen current regime must raise ValueError")
    for bad in ((R_VAL, np.array([0, 0]), C_VAL, C_REG),
                (R_VAL, R_REG, C_VAL, np.array([1, 1]))):
        try:
            lesson.conditional_psi(*bad, 3)
        except ValueError:
            continue
        raise AssertionError("mismatched value and regime shapes must raise ValueError")


# --- exercise 4: drift_signature ----------------------------------------------------------

def test_drift_signature_is_zero_against_itself():
    sig = lesson.drift_signature(lesson.REFERENCE, lesson.REFERENCE)
    for field in ("marginal", "conditional", "check"):
        got = getattr(sig, field)
        assert np.isclose(got, 0.0, atol=1e-9), (
            f"a window scored against itself has {field} 0.0, got {got!r}"
        )
    assert np.isclose(sig.median_shift, 0.0, atol=1e-12), (
        "median_shift is current minus reference: 0.0 against itself"
    )
    # It is a MEDIAN. On a window with one wild reading the mean says the opposite.
    skewed = lesson.drift_signature(toy_window([[0.0, 0.0, 0.0, 12.0]], name="skew_ref"),
                                    toy_window([[1.0, 1.0, 1.0, 1.0]], name="skew_cur"), 2)
    assert np.isclose(skewed.median_shift, 1.0), (
        f"the reference row's median is 0.0 and the current row's is 1.0, so median_shift is "
        f"+1.0; got {skewed.median_shift}. A MEAN gives -2.0 here — it reports the fleet "
        "getting better because one reading in the reference was enormous, and the signature "
        "table prints this column under the name 'median shift'"
    )


def test_drift_signature_bins_each_stream_on_its_own():
    recal = lesson.drift_signature(lesson.REFERENCE, lesson.LIVE["recalibration"])
    aged = lesson.drift_signature(lesson.REFERENCE, lesson.LIVE["ageing"])
    ref, cur = lesson.REFERENCE, lesson.LIVE["recalibration"]
    want_check = lesson.psi(ref.check, cur.check,
                            lesson.reference_bins(ref.check, lesson.N_BINS))
    assert np.isclose(recal.check, want_check, rtol=1e-9), (
        f"the check component must be psi(reference.check, current.check) over bins built "
        f"from reference.CHECK; expected {want_check:.6f}, got {recal.check:.6f}. Re-using "
        "the health index's bins scores one stream against another stream's quantiles, and "
        "the two do not live on the same scale"
    )
    want_marginal = lesson.psi(ref.health, cur.health,
                               lesson.reference_bins(ref.health, lesson.N_BINS))
    assert np.isclose(recal.marginal, want_marginal, rtol=1e-9), (
        f"the marginal component must be psi over bins built from reference.health; expected "
        f"{want_marginal:.6f}, got {recal.marginal:.6f}"
    )
    want_conditional = lesson.conditional_psi(ref.health, ref.regime, cur.health, cur.regime,
                                              lesson.N_BINS)
    assert np.isclose(recal.conditional, want_conditional, rtol=1e-9), (
        f"the conditional component is conditional_psi on the two windows' health and regime "
        f"arrays; expected {want_conditional:.6f}, got {recal.conditional:.6f}"
    )
    assert recal.check > 20 * aged.check, (
        f"only the recalibration window moved the instrument self-test: check "
        f"{recal.check:.4f} against ageing's {aged.check:.4f}. If they are similar you are "
        "scoring the CHECK readings against bins built from the HEALTH readings, and those "
        "two streams do not live on the same scale"
    )
    assert aged.check < band(), (
        f"the ageing fleet's instruments are perfect; its check PSI ({aged.check:.4f}) must "
        f"sit under the measured alert level ({band():.5f})"
    )


def test_drift_signature_separates_regime_from_ageing():
    regime = lesson.drift_signature(lesson.REFERENCE, lesson.LIVE["regime"])
    aged = lesson.drift_signature(lesson.REFERENCE, lesson.LIVE["ageing"])
    assert regime.conditional < 0.2 * regime.marginal, (
        f"the regime window's drift is explained by duty: conditional "
        f"{regime.conditional:.4f} against marginal {regime.marginal:.4f}. If conditional is "
        "as large as marginal, check each window is passed its OWN regime array"
    )
    assert aged.conditional > 0.5 * aged.marginal, (
        f"the ageing window's duty mix never moved, so conditioning explains none of it; got "
        f"conditional {aged.conditional:.4f} against marginal {aged.marginal:.4f}"
    )
    assert regime.median_shift > aged.median_shift, (
        "median_shift is a descriptive number and it orders these two windows the OPPOSITE "
        "way from the fix they need — that is why it is reported and never tested against"
    )


# --- exercise 5: diagnose_drift -----------------------------------------------------------

def test_diagnose_quiet_comes_first():
    assert lesson.diagnose_drift(DriftSignature(0.01, 0.01, 0.001, 0.0), 0.03) == "none"
    assert lesson.diagnose_drift(DriftSignature(0.02, 0.02, 0.9, 0.0), 0.03) == "none", (
        "the quiet test runs before the instrument test: if the feature has not moved there "
        "is nothing to fix, and a self-test wobble alone is a job for the metrology team"
    )
    assert lesson.diagnose_drift(DriftSignature(0.03, 0.03, 0.001, 0.0), 0.03) == "ageing", (
        "the quiet test is `marginal < psi_alert`, so a marginal exactly ON the alert level "
        "is drift and falls through the remaining rules"
    )


def test_diagnose_checks_the_instrument_before_duty():
    both = DriftSignature(marginal=2.0, conditional=0.5, check=0.9, median_shift=0.5)
    assert lesson.diagnose_drift(both, 0.03) == "recalibration", (
        "this signature satisfies the duty rule (0.50 <= 0.5 * 2.00) AND the instrument rule "
        "(0.90 >= 0.03). Testing duty first calls it a regime change and conditions on "
        "readings the transmitter got wrong; a moved instrument outranks a moved duty point "
        "because every other statistic is computed on what the instrument reported"
    )
    real = lesson.drift_signature(lesson.REFERENCE, lesson.LIVE["recal_and_regime"])
    assert lesson.diagnose_drift(real, band()) == "recalibration", (
        f"the turnaround window swapped transmitters AND moved the duty mix; its signature "
        f"(marginal {real.marginal:.3f}, conditional {real.conditional:.3f}, check "
        f"{real.check:.3f}) must come back 'recalibration'"
    )


def test_diagnose_duty_then_fleet_with_their_boundaries():
    assert lesson.diagnose_drift(DriftSignature(1.2, 0.01, 0.001, 0.4), 0.03) == "regime"
    assert lesson.diagnose_drift(DriftSignature(1.2, 0.6, 0.001, 0.4), 0.03) == "regime", (
        "the duty rule is `conditional <= explained_fraction * marginal`, so a conditional "
        "exactly at half of marginal is explained by duty"
    )
    assert lesson.diagnose_drift(DriftSignature(1.2, 0.61, 0.001, 0.4), 0.03) == "ageing", (
        "just past explained_fraction * marginal, duty no longer explains it and the "
        "instrument is clean, so the machines are worse"
    )
    assert lesson.diagnose_drift(DriftSignature(1.2, 1.2, 0.03, 0.4), 0.03) == "recalibration", (
        "the instrument rule is `check >= psi_alert`, against the SAME alert level the "
        "marginal is judged by; a check exactly on it counts as moved"
    )
    assert lesson.diagnose_drift(DriftSignature(1.2, 0.9, 0.001, 0.4), 0.03,
                                 explained_fraction=0.8) == "regime", (
        "explained_fraction is a parameter, not a constant baked into the comparison"
    )
    assert set(DIAGNOSES) == {"none", "recalibration", "regime", "ageing"}


def test_diagnose_validates():
    quiet = DriftSignature(0.01, 0.01, 0.001, 0.0)
    for alert, frac in ((-0.01, 0.5), (0.03, 0.0), (0.03, 1.0), (0.03, -0.5), (0.03, 1.5)):
        try:
            lesson.diagnose_drift(quiet, alert, frac)
        except ValueError:
            continue
        raise AssertionError(f"psi_alert={alert}, explained_fraction={frac} must raise")
    for sig in (DriftSignature(-0.1, 0.1, 0.1, 0.0), DriftSignature(0.1, -0.1, 0.1, 0.0),
                DriftSignature(0.1, 0.1, -0.1, 0.0)):
        try:
            lesson.diagnose_drift(sig, 0.03)
        except ValueError:
            continue
        raise AssertionError(f"a negative PSI component in {sig} must raise ValueError")


# --- exercise 6: adjust_feature -----------------------------------------------------------

B6 = Baseline(0.9, (0.0, 0.5), 1.0, 2.0, Prices(1.0, 10.0, 2.0))
H6 = np.array([[1.0, 2.0, 3.0]])
R6 = np.array([[0, 1, 1]])
C6 = np.array([[1.4, 1.4]])


def test_adjust_feature_inert_fixes_copy():
    for inert in ("none", "reprice"):
        got = lesson.adjust_feature(inert, H6, R6, C6, B6)
        assert np.allclose(got, H6), (
            f"{inert!r} leaves the readings alone, got {got}. Re-pricing moves the THRESHOLD"
        )
        assert got is not H6, (
            f"{inert!r} must return a copy; the caller's array is shared with every other "
            "fix in the matrix"
        )
    keep = np.array([[1.0, 2.0, 3.0]])
    lesson.adjust_feature("rebaseline", keep, R6, C6, B6)
    lesson.adjust_feature("condition", keep, R6, C6, B6)
    assert np.allclose(keep, [[1.0, 2.0, 3.0]]), "never modify the caller's array in place"


def test_adjust_feature_conditions_per_reading():
    cond = lesson.adjust_feature("condition", H6, R6, C6, B6)
    assert np.allclose(cond, [[1.0, 1.5, 2.5]]), (
        f"expected [[1.0, 1.5, 2.5]] — regime 1 carries an offset of 0.5 — got {cond}. One "
        "offset per UNIT gives [[0.67, 1.67, 2.67]]; duty is a property of the hour"
    )
    two = np.array([[4.0, 4.0], [4.0, 4.0]])
    reg = np.array([[0, 1], [1, 0]])
    got = lesson.adjust_feature("condition", two, reg, np.ones((2, 2)), B6)
    assert np.allclose(got, [[4.0, 3.5], [3.5, 4.0]]), (
        f"the offset follows each reading's own regime index; got {got}"
    )


def test_adjust_feature_rebaselines_against_the_check():
    reb = lesson.adjust_feature("rebaseline", H6, R6, C6, B6)
    assert np.allclose(reb, [[0.6, 1.6, 2.6]]), (
        f"this channel's self-test reads 1.4 against a kick-off level of 1.0, so every "
        f"reading drops by 0.4; expected [[0.6, 1.6, 2.6]], got {reb}. Adding the shift "
        "instead of subtracting doubles the error you meant to remove"
    )
    skew = lesson.adjust_feature("rebaseline", H6, R6, np.array([[1.0, 1.0, 1.0, 9.0]]), B6)
    assert np.allclose(skew, H6), (
        f"the shift is the MEDIAN of the self-test readings, so one absurd reading does not "
        f"move it; got {skew}. A mean gives 3.0 and shifts a healthy channel by 2.0"
    )
    per_unit = lesson.adjust_feature("rebaseline", np.array([[1.0, 2.0], [5.0, 6.0]]),
                                     np.zeros((2, 2), int),
                                     np.array([[1.4, 1.4], [1.0, 1.0]]), B6)
    assert np.allclose(per_unit, [[0.6, 1.6], [5.0, 6.0]]), (
        f"the shift is PER UNIT: unit 0 moved and unit 1 did not; got {per_unit}. A single "
        "pooled median moves both by the average and corrupts the channel that was working"
    )


def test_adjust_feature_leaves_a_clean_channel_alone():
    clean = lesson.adjust_feature("rebaseline", H6, R6, np.array([[1.0, 1.0]]), B6)
    assert np.allclose(clean, H6), (
        f"a unit whose self-test never moved must come back UNCHANGED, got {clean}. This is "
        "the property that makes re-baselining the wrong answer to ageing: it can only undo "
        "what the instrument did, and on an ageing fleet the instrument did nothing"
    )
    aged = lesson.LIVE["ageing"]
    same = lesson.adjust_feature("rebaseline", aged.health, aged.regime, aged.check,
                                 lesson.BASELINE)
    moved = float(np.max(np.abs(same - aged.health)))
    assert moved < 0.05, (
        f"on the ageing window the instruments never moved, so re-baselining must be very "
        f"nearly a no-op; the largest reading it shifted moved by {moved:.3f}. A large "
        "number means you re-baselined against the FEATURE's own quiet level, which "
        "subtracts the fleet's decline instead of measuring it"
    )


def test_adjust_feature_validates():
    for bad in (("polish", H6, R6, C6),
                ("none", np.array([1.0, 2.0]), np.array([0, 1]), np.array([[1.0]])),
                ("condition", H6, np.array([[0, 1]]), C6),
                ("condition", H6, np.array([[0, 1, 5]]), C6),
                ("condition", H6, np.array([[0, 1, -1]]), C6),
                ("rebaseline", H6, R6, np.array([[1.0], [1.0]])),
                ("rebaseline", H6, R6, np.array([1.0, 1.0]))):
        try:
            lesson.adjust_feature(*bad, B6)
        except ValueError:
            continue
        raise AssertionError(f"adjust_feature({bad[0]!r}, ...) must raise ValueError")


# --- exercise 7: estimate_prices ----------------------------------------------------------

GUESS = Prices(10.0, 100.0, 5.0)
ROWS = [AuditEntry(0, 1, "planned", 20.0), AuditEntry(1, 2, "planned", 30.0)]


def test_estimate_prices_averages_what_it_has():
    got = lesson.estimate_prices(ROWS, GUESS, min_observations=2)
    assert np.isclose(got.prices.planned, 25.0), (
        f"the mean of 20 and 30 is 25.0, got {got.prices.planned!r}"
    )
    assert got.counts == (2, 0, 0), (
        f"counts are (planned, unplanned, false_alarm), got {got.counts}"
    )
    mixed = lesson.estimate_prices(
        ROWS + [AuditEntry(2, 3, "unplanned", 900.0), AuditEntry(3, 4, "false_alarm", 7.0)],
        GUESS, 1)
    assert np.isclose(mixed.prices.unplanned, 900.0) and mixed.still_guessed == (), (
        f"with a minimum of one every class is measured; got {mixed.prices} / "
        f"{mixed.still_guessed}"
    )
    order = lesson.estimate_prices([AuditEntry(0, 0, "false_alarm", 3.0)], GUESS, 1)
    assert np.isclose(order.prices.false_alarm, 3.0) and np.isclose(order.prices.planned, 10.0), (
        f"OUTCOMES is ('planned', 'unplanned', 'false_alarm') and Prices has the same field "
        f"order; got {order.prices}. Zipping them any other way swaps the price of a "
        "call-out with the price of an overhaul"
    )
    # It is the MEAN, and on invoice amounts that is not a stylistic choice: expected_cost
    # multiplies a price by a count, so the plug-in that makes the total right is the mean.
    skewed = lesson.estimate_prices(
        [AuditEntry(i, 0, "unplanned", c) for i, c in enumerate((10.0, 10.0, 10.0, 10.0, 60.0))],
        GUESS, 5)
    assert np.isclose(skewed.prices.unplanned, 20.0), (
        f"the mean of 10, 10, 10, 10 and 60 is 20.0; got {skewed.prices.unplanned!r}. A "
        "MEDIAN gives 10.0 and halves the price of a missed failure — maintenance invoices "
        "are right-skewed, the cost function multiplies this number by a count, and a "
        "median under-prices every threshold derived from it"
    )


def test_estimate_prices_keeps_the_guess_for_an_empty_class():
    got = lesson.estimate_prices(ROWS, GUESS, min_observations=2)
    assert np.isclose(got.prices.unplanned, 100.0), (
        f"the audit has no unplanned lines, so that price keeps the kick-off guess of 100.0; "
        f"got {got.prices.unplanned!r}. A 0.0 here prices a missed failure at nothing and "
        "makes 'never alarm' the cost-optimal policy; a nan poisons every threshold sweep "
        "downstream"
    )
    empty = lesson.estimate_prices([], GUESS, 1)
    assert empty.prices == GUESS and empty.counts == (0, 0, 0), (
        f"an empty audit returns the kick-off guess untouched, got {empty}"
    )
    assert empty.still_guessed == OUTCOMES, (
        f"with no evidence at all every price is still a guess; got {empty.still_guessed}"
    )


def test_estimate_prices_reports_what_is_still_a_guess():
    got = lesson.estimate_prices(ROWS, GUESS, min_observations=2)
    assert got.still_guessed == ("unplanned", "false_alarm"), (
        f"still_guessed names the fallen-back outcomes in OUTCOMES order, got "
        f"{got.still_guessed}. Without it a number nobody measured is read as one somebody "
        "did, which is the whole failure this audit exists to stop"
    )
    thin = lesson.estimate_prices(ROWS, GUESS, min_observations=3)
    assert np.isclose(thin.prices.planned, 10.0) and thin.still_guessed == OUTCOMES, (
        f"two entries is below a minimum of three, so planned falls back too; got "
        f"{thin.prices} / {thin.still_guessed}"
    )
    assert thin.counts == (2, 0, 0), (
        "counts report what the audit HOLDS, not what was used: a class can have two entries "
        "and still be a guess"
    )
    assert np.isclose(lesson.estimate_prices(ROWS, GUESS, 2).prices.planned, 25.0), (
        "the test is `>= min_observations`: exactly enough entries is enough"
    )


def test_estimate_prices_validates():
    for bad in ([AuditEntry(0, 0, "scrapped", 1.0)], [AuditEntry(0, 0, "planned", -1.0)],
                [AuditEntry(0, 0, "planned", float("nan"))],
                [AuditEntry(0, 0, "planned", float("inf"))]):
        try:
            lesson.estimate_prices(bad, GUESS, 1)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must raise ValueError")
    for bad_min in (0, -1):
        try:
            lesson.estimate_prices(ROWS, GUESS, bad_min)
        except ValueError:
            continue
        raise AssertionError(f"min_observations={bad_min} must raise ValueError")


# --- exercise 8: apply_fix ----------------------------------------------------------------

B8 = Baseline(1.0, (0.0,), 1.0, 2.0, Prices(1.0, 100.0, 5.0))
H8 = np.array([[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
RUNS8 = (Run(0, 4, FAILURE), Run(1, 4, CENSORED))
W8 = toy_window(H8, runs=RUNS8, name="w8")
NOISY = toy_window(np.array([[1.0, 1.0, 3.0, 3.0], [2.5, 2.5, 2.5, 2.5]]), runs=RUNS8,
                   name="noisy")
GRID = np.array([2.0, 3.5])


def test_apply_fix_scores_the_matrix():
    got = lesson.apply_fix("none", W8, W8, B8, Prices(1.0, 100.0, 5.0), np.array([2.0]), 2)
    assert got == lesson.FixResult("none", 2.0, Counts(1, 0, 0, 1), 1.0), (
        f"expected FixResult('none', 2.0, Counts(1, 0, 0, 1), 1.0), got {got}"
    )
    late = lesson.apply_fix("none", W8, W8, B8, Prices(1.0, 100.0, 5.0), np.array([2.0]), 3)
    assert late.counts == Counts(0, 1, 0, 1), (
        f"lead_hours must reach the counting: the alarm lands at hour 2 and the run ends at "
        f"4, so three hours of warning was not available; got {late.counts}"
    )
    priced = lesson.apply_fix("none", NOISY, NOISY, B8, Prices(7.0, 11.0, 13.0), GRID, 2)
    assert np.isclose(priced.cost, 20.0), (
        f"one catch at 7 and one false alarm at 13 is 20; got {priced.cost}. The cost uses "
        "the prices passed in, not baseline.prices — the audit is the whole point"
    )


def test_apply_fix_feature_fixes_keep_the_threshold():
    for inert in ("none", "rebaseline", "condition"):
        r = lesson.apply_fix(inert, W8, W8, B8, Prices(1.0, 100.0, 5.0), GRID, 2)
        assert np.isclose(r.threshold, B8.threshold), (
            f"{inert!r} moves the FEATURE and leaves the threshold at baseline.threshold "
            f"({B8.threshold}); got {r.threshold}. Re-deriving under every fix turns the "
            "matrix into a comparison of one thing with itself"
        )
        assert r.fix == inert, f"the result carries the fix it was asked for, got {r.fix!r}"


def test_apply_fix_reprice_moves_with_the_prices():
    loud = lesson.apply_fix("reprice", NOISY, NOISY, B8, Prices(1.0, 100.0, 5.0), GRID, 2)
    assert np.isclose(loud.threshold, 2.0), (
        f"a missed failure at 100 against a false alarm at 5: the sweep takes 2.0 and buys "
        f"the catch; got {loud.threshold}"
    )
    quiet = lesson.apply_fix("reprice", NOISY, NOISY, B8, Prices(1.0, 1.0, 500.0), GRID, 2)
    assert np.isclose(quiet.threshold, 3.5), (
        f"the SAME window re-priced: a false alarm at 500 against a failure at 1 takes the "
        f"threshold to 3.5; got {quiet.threshold}. A threshold that will not move with the "
        "prices means you swept at baseline.prices instead of the prices you were handed"
    )


def test_apply_fix_reprice_reads_the_history_not_the_window():
    past = toy_window(np.array([[1.0, 1.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]]), runs=RUNS8,
                      name="past")
    split = lesson.apply_fix("reprice", W8, past, B8, Prices(1.0, 100.0, 5.0),
                             np.array([2.0, 6.0]), 2)
    assert np.isclose(split.threshold, 6.0), (
        f"in the HISTORY window the failing unit never alarms and the censored one always "
        f"does, so the cheap threshold there is 6.0; got {split.threshold}. Re-deriving on "
        "`window` fits the threshold to the very rows it is then scored on, and every fix "
        "in the matrix looks equally good"
    )
    assert split.counts == lesson.alarm_counts(W8.health, W8.runs, 6.0, 2), (
        "the counts are still scored on `window`; only the THRESHOLD comes from the history"
    )


def test_apply_fix_validates():
    for bad_fix, bad_thr in (("polish", np.array([2.0])), ("reprice", np.array([]))):
        try:
            lesson.apply_fix(bad_fix, W8, W8, B8, Prices(1.0, 100.0, 5.0), bad_thr, 2)
        except ValueError:
            continue
        raise AssertionError(f"apply_fix({bad_fix!r}, ..., {bad_thr}) must raise ValueError")


# --- exercise 9: respond_to_drift ---------------------------------------------------------

EXPECTED = {"steady": ("none", "none"), "recalibration": ("recalibration", "rebaseline"),
            "regime": ("regime", "condition"), "ageing": ("ageing", "reprice")}


def test_respond_maps_each_diagnosis_to_its_own_fix():
    for name, (want_d, want_f) in EXPECTED.items():
        got = lesson.respond_to_drift(lesson.LIVE[name], lesson.HISTORY[name],
                                      lesson.REFERENCE, lesson.BASELINE, lesson.KICKOFF,
                                      band())
        assert got.diagnosis == want_d, (
            f"the {name} window is a {want_d} drift, diagnosed {got.diagnosis!r}"
        )
        assert got.fix == want_f, (
            f"a {want_d} diagnosis calls for {want_f!r}, got {got.fix!r}. An instrument is "
            "re-zeroed, a duty change is conditioned on, an ageing fleet is re-priced, and "
            "nothing is done when nothing moved — wiring those four the wrong way is the "
            "mistake this module exists to prevent"
        )
        assert got.result.fix == got.fix, "the FixResult must carry the fix that was applied"


def test_respond_moves_the_threshold_only_when_it_should():
    unreachable = lesson.respond_to_drift(lesson.LIVE["steady"], lesson.HISTORY["steady"],
                                          lesson.REFERENCE, lesson.BASELINE, lesson.KICKOFF,
                                          psi_alert=1e9)
    assert unreachable.diagnosis == "none" and unreachable.fix == "none", (
        f"with an alert level nothing can reach, every window is quiet; got "
        f"{unreachable.diagnosis!r}. psi_alert must be passed through to diagnose_drift"
    )
    assert np.isclose(unreachable.result.threshold, lesson.BASELINE.threshold), (
        "the 'none' fix leaves the kick-off threshold exactly where it was"
    )
    aged = lesson.respond_to_drift(lesson.LIVE["ageing"], lesson.HISTORY["ageing"],
                                   lesson.REFERENCE, lesson.BASELINE, lesson.KICKOFF, band())
    assert not np.isclose(aged.result.threshold, lesson.BASELINE.threshold), (
        f"re-pricing the ageing window must MOVE the threshold off "
        f"{lesson.BASELINE.threshold}; got {aged.result.threshold}"
    )
    for name in ("recalibration", "regime"):
        got = lesson.respond_to_drift(lesson.LIVE[name], lesson.HISTORY[name],
                                      lesson.REFERENCE, lesson.BASELINE, lesson.KICKOFF,
                                      band())
        assert np.isclose(got.result.threshold, lesson.BASELINE.threshold), (
            f"the {name} response is a FEATURE fix and leaves the threshold alone; got "
            f"{got.result.threshold}"
        )



def test_respond_passes_its_parameters_through():
    """Every argument must reach the function that uses it, not a module constant.

    n_bins and explained_fraction go to drift_signature and diagnose_drift; thresholds and
    lead_hours go on to apply_fix. The fleet's own diagnoses are robust to the bin count from
    2 bins to 48, which is a good property of the design and a bad one for a test, so the bin
    count is graded on a toy pair built for the purpose: one where a coarse histogram cannot
    see the drift and a finer one can.
    """
    ref_row = np.linspace(0.0, 1.0, 100)
    cur_row = np.concatenate([np.linspace(0.0, 0.25, 50), np.linspace(0.5, 1.0, 50)])
    ref_w = toy_window(np.tile(ref_row, (10, 1)), name="ref")
    cur_w = toy_window(np.tile(cur_row, (10, 1)), name="cur")
    b = Baseline(0.0, (0.0,), 1.0, 2.0, Prices(1.0, 10.0, 2.0))
    grid = np.array([0.5, 2.0])
    seen = {}
    for n_bins in (2, 4):
        got = lesson.respond_to_drift(cur_w, cur_w, ref_w, b, Prices(1.0, 10.0, 2.0),
                                      psi_alert=0.1, thresholds=grid, lead_hours=1,
                                      n_bins=n_bins)
        want = lesson.diagnose_drift(lesson.drift_signature(ref_w, cur_w, n_bins), 0.1)
        assert got.diagnosis == want, (
            f"with n_bins={n_bins} the composition gives {want!r} and respond_to_drift gave "
            f"{got.diagnosis!r}; n_bins must reach drift_signature"
        )
        seen[n_bins] = got.diagnosis
    assert seen[2] == "none" and seen[4] != "none", (
        f"these two windows put the same halves of their mass either side of a single "
        f"median cut and different quarters either side of the quartiles, so 2 bins must see "
        f"nothing and 4 must see it; got {seen}. An implementation that ignores n_bins and "
        "uses the module default answers both the same way"
    )
    win, hist = lesson.LIVE["regime"], lesson.HISTORY["regime"]
    strict = lesson.respond_to_drift(win, hist, lesson.REFERENCE, lesson.BASELINE,
                                     lesson.KICKOFF, band(), explained_fraction=0.0005)
    want = lesson.diagnose_drift(lesson.drift_signature(lesson.REFERENCE, win), band(),
                                 0.0005)
    assert strict.diagnosis == want, (
        f"explained_fraction must reach diagnose_drift: the composition gives {want!r} and "
        f"respond_to_drift gave {strict.diagnosis!r}"
    )
    assert strict.diagnosis != "regime", (
        "with a fraction this small almost nothing counts as explained by duty, so the "
        "regime window must stop being called one; if it does not, respond_to_drift is "
        "using the module constant instead of the argument"
    )
    # thresholds and lead_hours have the same trap: they are only observable through
    # apply_fix, and only on a window whose diagnosis re-derives. The ageing window does.
    aged_win, aged_hist = lesson.LIVE["ageing"], lesson.HISTORY["ageing"]
    grid = np.array([1.42, 3.77])
    on_grid = lesson.respond_to_drift(aged_win, aged_hist, lesson.REFERENCE, lesson.BASELINE,
                                      lesson.KICKOFF, band(), thresholds=grid)
    assert on_grid.fix == "reprice" and float(on_grid.result.threshold) in (1.42, 3.77), (
        f"the ageing window is re-priced, so the threshold it comes back with has to be a "
        f"point of the grid you passed ({grid.tolist()}); got "
        f"{on_grid.result.threshold}. A threshold off that grid means respond_to_drift swept "
        "the module's THRESHOLDS instead of forwarding its own argument to apply_fix"
    )
    prompt = lesson.respond_to_drift(aged_win, aged_hist, lesson.REFERENCE, lesson.BASELINE,
                                     lesson.KICKOFF, band(), lead_hours=1)
    patient = lesson.respond_to_drift(aged_win, aged_hist, lesson.REFERENCE, lesson.BASELINE,
                                      lesson.KICKOFF, band(), lead_hours=240)
    assert prompt.result.counts != patient.result.counts, (
        f"an alarm that needs one hour of warning and one that needs 240 cannot score the "
        f"same fleet identically; both gave {prompt.result.counts}. lead_hours must be "
        "forwarded to apply_fix, which forwards it to alarm_counts"
    )

# --- the claim the lesson is built on, measured on the generated fleet --------------------

def _matrix(prices):
    return {name: {f: lesson.apply_fix(f, lesson.LIVE[name], lesson.HISTORY[name],
                                       lesson.BASELINE, prices) for f in FIXES}
            for name in EXPECTED}


def test_the_recommended_fix_is_the_cheapest_response():
    audited = lesson.estimate_prices(lesson.AUDIT, lesson.KICKOFF,
                                     lesson.MIN_OBSERVATIONS).prices
    matrix = _matrix(audited)
    for name, (_, want_f) in EXPECTED.items():
        row = matrix[name]
        best = min(row.values(), key=lambda r: r.cost)
        assert np.isclose(row[want_f].cost, best.cost), (
            f"on the {name} window the {want_f!r} response costs {row[want_f].cost:,.0f} and "
            f"the cheapest cell is {best.fix!r} at {best.cost:,.0f}. The diagonal of this "
            "matrix is the lesson; if it does not hold, one of adjust_feature, apply_fix or "
            "the threshold rule is not doing what its docstring says"
        )


def test_the_wrong_fix_is_measurably_worse():
    audited = lesson.estimate_prices(lesson.AUDIT, lesson.KICKOFF,
                                     lesson.MIN_OBSERVATIONS).prices
    matrix = _matrix(audited)
    aged = matrix["ageing"]
    assert aged["condition"].cost > 2.0 * aged["reprice"].cost, (
        f"conditioning on duty when the fleet is simply older costs "
        f"{aged['condition'].cost:,.0f} against re-pricing's {aged['reprice'].cost:,.0f}. It "
        "should be more than twice as expensive — the duty offsets are subtracted from real "
        "degradation and the failures go unalarmed"
    )
    assert aged["condition"].counts.fn > aged["reprice"].counts.fn + 10, (
        f"the extra cost is missed failures, not paperwork: {aged['condition'].counts.fn} "
        f"against {aged['reprice'].counts.fn}"
    )
    recal = matrix["recalibration"]
    assert recal["condition"].cost > 1.5 * recal["rebaseline"].cost, (
        f"conditioning on duty does nothing for a transmitter that reads high: "
        f"{recal['condition'].cost:,.0f} against re-baselining's "
        f"{recal['rebaseline'].cost:,.0f}"
    )
    regime = matrix["regime"]
    for name in ("regime", "ageing"):
        row = matrix[name]
        assert np.isclose(row["rebaseline"].cost, row["none"].cost), (
            f"on the {name} window the instruments never moved, so re-baselining against "
            f"them must cost EXACTLY what doing nothing costs: {row['rebaseline'].cost:,.0f} "
            f"against {row['none'].cost:,.0f}. A difference means adjust_feature's "
            "'rebaseline' branch is reading the feature instead of the self-test"
        )
    assert regime["condition"].cost < regime["none"].cost, (
        f"conditioning must beat doing nothing on a duty change: "
        f"{regime['condition'].cost:,.0f} against {regime['none'].cost:,.0f}"
    )
    assert regime["none"].counts.fp > 10 * max(1, regime["condition"].counts.fp), (
        f"the damage a duty change does is a false-alarm storm, and conditioning is what "
        f"stops it: {regime['none'].counts.fp} false alarms against "
        f"{regime['condition'].counts.fp}"
    )


RUBRIC = [
    (test_reference_bins_shape_and_infinite_edges, 3,
     "n_bins + 1 edges, -inf first and +inf last, strictly increasing, input flattened"),
    (test_reference_bins_cuts_the_mass_not_the_range, 4,
     "np.quantile of the reference at the interior cut points, not a linspace over its range"),
    (test_reference_bins_collapses_ties_and_validates, 3,
     "np.unique the interior cuts; empty values, a non-finite value and n_bins < 2 all raise"),
    (test_psi_value_and_zero, 5,
     "sum((c - r) * log(c / r)) over PROPORTIONS of each sample, not counts"),
    (test_psi_is_symmetric_and_scale_free, 4,
     "divide each histogram by its own sample size before the difference and the log"),
    (test_psi_floors_an_empty_bin, 4,
     "np.maximum(p, floor) on BOTH proportion arrays before the ratio — never drop a bin"),
    (test_psi_bins_are_half_open_and_validated, 3,
     "bin i is [edges[i], edges[i+1]); fewer than 3 edges, a non-increasing edge, an empty "
     "sample and a non-positive floor all raise"),
    (test_conditional_psi_explains_a_pure_regime_change, 5,
     "loop over the regimes in the reference and build each one's bins from that regime's "
     "own reference readings"),
    (test_conditional_psi_weights_by_the_reference_mix, 5,
     "weight each regime's PSI by its share of the REFERENCE readings; the current mix is "
     "the thing being conditioned away and cannot also be the weights"),
    (test_conditional_psi_keeps_a_within_regime_shift, 4,
     "a shift inside a regime must survive; if everything vanishes you are comparing each "
     "regime with itself"),
    (test_conditional_psi_rejects_an_unseen_regime_and_validates, 4,
     "a regime in the current window and not in the reference is a ValueError, and both "
     "value/regime pairs must match in shape"),
    (test_drift_signature_is_zero_against_itself, 3,
     "marginal, conditional and check are all PSI against the reference's own bins, and "
     "median_shift is current minus reference"),
    (test_drift_signature_bins_each_stream_on_its_own, 5,
     "the check component uses reference_bins built from reference.check, not from "
     "reference.health"),
    (test_drift_signature_separates_regime_from_ageing, 4,
     "pass each window its OWN regime array to conditional_psi"),
    (test_diagnose_quiet_comes_first, 3,
     "`marginal < psi_alert` -> 'none', and it is tested before everything else"),
    (test_diagnose_checks_the_instrument_before_duty, 5,
     "the instrument test runs BEFORE the duty test: a transmitter that reads high makes "
     "every other statistic a statistic about the transmitter"),
    (test_diagnose_duty_then_fleet_with_their_boundaries, 4,
     "`conditional <= explained_fraction * marginal` -> 'regime', `check >= psi_alert` -> "
     "'recalibration', otherwise 'ageing'"),
    (test_diagnose_validates, 2,
     "a negative psi_alert, an explained_fraction outside (0, 1) and a negative PSI "
     "component all raise ValueError"),
    (test_adjust_feature_inert_fixes_copy, 3,
     "'none' and 'reprice' return health.copy(); nothing is ever modified in place"),
    (test_adjust_feature_conditions_per_reading, 4,
     "np.asarray(baseline.regime_offset)[regime] — one offset per READING, not per unit"),
    (test_adjust_feature_rebaselines_against_the_check, 5,
     "subtract np.median(check, axis=1) - baseline.check_level, per unit; a mean is not "
     "robust to one absurd self-test reading"),
    (test_adjust_feature_leaves_a_clean_channel_alone, 5,
     "re-baselining reads the INSTRUMENT, never the feature's own quiet level: a unit whose "
     "self-test never moved comes back unchanged"),
    (test_adjust_feature_validates, 2,
     "an unknown fix, a 1-D health, a mismatched regime, an out-of-range regime index and a "
     "check with the wrong number of rows all raise ValueError"),
    (test_estimate_prices_averages_what_it_has, 4,
     "mean of the costs per outcome, zipped against Prices in OUTCOMES order"),
    (test_estimate_prices_keeps_the_guess_for_an_empty_class, 5,
     "a class below min_observations keeps the FALLBACK price; np.mean of an empty list is "
     "nan and a 0.0 prices a missed failure at nothing"),
    (test_estimate_prices_reports_what_is_still_a_guess, 4,
     "still_guessed lists the fallen-back outcomes in OUTCOMES order; counts report what the "
     "audit holds, not what was used"),
    (test_estimate_prices_validates, 2,
     "an unknown outcome, a negative or non-finite cost, and min_observations < 1 all raise"),
    (test_apply_fix_scores_the_matrix, 4,
     "adjust the feature, count with alarm_counts at lead_hours, price with expected_cost at "
     "the PRICES PASSED IN"),
    (test_apply_fix_feature_fixes_keep_the_threshold, 5,
     "only 'reprice' re-derives; the other three use baseline.threshold unchanged"),
    (test_apply_fix_reprice_moves_with_the_prices, 4,
     "sweep at `prices`, not at baseline.prices — the audited prices are the whole reason "
     "this function takes them"),
    (test_apply_fix_reprice_reads_the_history_not_the_window, 5,
     "cost_optimal_threshold on `history`, then count on `window`; re-deriving in-sample "
     "flatters every fix equally"),
    (test_apply_fix_validates, 2,
     "an unknown fix and an empty threshold grid both raise ValueError"),
    (test_respond_maps_each_diagnosis_to_its_own_fix, 6,
     "none->none, recalibration->rebaseline, regime->condition, ageing->reprice"),
    (test_respond_moves_the_threshold_only_when_it_should, 4,
     "pass psi_alert through to diagnose_drift, and the fix through to apply_fix"),
    (test_respond_passes_its_parameters_through, 4,
     "n_bins goes to drift_signature and explained_fraction to diagnose_drift; a module "
     "constant used in place of either is invisible until someone passes a different one"),
    (test_the_recommended_fix_is_the_cheapest_response, 5,
     "if a row's diagonal is not the cheapest cell, re-read adjust_feature's three branches "
     "and apply_fix's threshold rule"),
    (test_the_wrong_fix_is_measurably_worse, 5,
     "the gaps come from adjust_feature: conditioning must subtract the duty offsets and "
     "re-baselining must subtract the instrument shift, and neither may do the other's job"),
]
