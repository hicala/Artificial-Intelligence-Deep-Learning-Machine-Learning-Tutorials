"""Autograder rubric for P01-L06. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Expected intervals are re-derived HERE, by `_py_bootstrap`, which draws the same documented
  `(n_boot, n)` index matrix and then walks it with a Python loop and `np.quantile` instead of
  a vectorised gather and `np.percentile`. It is a different code path to the same contract,
  so it pins the number without copying a literal out of the notebook's output. The draw
  itself is part of the contract — the notebook says to take it in one `rng.integers` call —
  because an evidence pack whose intervals move between machines is not evidence.
* `lesson.TEST_FEATURES`, `lesson.TRAIN_FEATURES` and `lesson.RECORDED_LABELS` are never
  mutated. Tests that need different data build their own small arrays. Every expectation
  drawn from the held-out split is derived from `_PRISTINE_FEATURES` / `_PRISTINE_LABELS`,
  snapshots taken at import before any case has run, because a snapshot taken inside a test
  cannot detect an implementation that corrupted the fixture in an EARLIER case — mutation
  testing found exactly that hole.
* The crossing tests run mostly on HAND-BUILT curves whose lo/point columns were chosen so
  that the bound crossing and the point crossing are two grid steps apart, whose floor sits
  above, at and below the expected operating severity in turn, and whose ledger sequence
  numbers straddle the floor. The real sweep is a single integration case. A rubric that only
  ever saw the real sweep would be graded on one crossing, and one crossing cannot separate
  `lo < floor` from `point < floor` reliably at the margins this fixture has.
* `test_poison_support_floor_is_the_parameter` is what stops an implementation that
  special-cases `b09`, or that filters small batches out of `batches` entirely, from scoring
  full marks: lower `min_support` and the same batch must become a finding.
* Three tests exist because the fixture is one-sided. The real sweep fails, so a crossing
  report hard-wired to "fail" would pass; the real poison scan finds exactly one batch, so a
  findings list of length one hides every ordering bug; and the real extraction curve reaches
  its target, so `reached=False` was never exercised. Each now has a hand-built case.
"""
import importlib.util, math, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

L = lesson
FLOOR_RATIONALE = "a rationale long enough to be a rationale rather than a decoration"

# Snapshots taken at IMPORT, before any rubric case has run. Mutation testing found that
# comparing the fixture against a copy made inside the test is worthless: an earlier case has
# already called perturbation_curve, so an implementation that writes damage back into the
# features it was handed had corrupted the fixture before the test that hunts for it started
# looking. Every expectation derived from the held-out split is now derived from these.
_PRISTINE_FEATURES = {k: np.array(v, dtype=float) for k, v in lesson.TEST_FEATURES.items()}
_PRISTINE_LABELS = np.array(lesson.TEST_LABELS)


# --- independent re-derivations ----------------------------------------------------------

def _py_bootstrap(y_true, y_pred, metric, n_boot, level=0.95, seed=None):
    """The documented draw, walked by a Python loop instead of a vectorised gather."""
    seed = L.BOOT_SEED if seed is None else seed
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    reps = np.array([float(metric(y_true[row], y_pred[row])) for row in idx])
    a = (1.0 - level) / 2.0
    return float(np.quantile(reps, a)), float(np.quantile(reps, 1.0 - a))


def _py_accuracy(y_true, y_pred) -> float:
    hits = sum(1 for t, p in zip(list(y_true), list(y_pred)) if int(t) == int(p))
    return hits / len(list(y_true))


def _py_rate(flags) -> float:
    flags = [int(f) for f in flags]
    return sum(flags) / len(flags)


def _fresh_floor(ledger=None, metric="accuracy", floor=0.86):
    led = L.EvidenceLedger() if ledger is None else ledger
    return led, L.seal_floor(led, metric, floor, FLOOR_RATIONALE)


def _hand_curve(points, family="missing_field", metric="accuracy",
                expected=0.20, first_seq=1):
    """A curve with invented numbers: (severity, point, lo) triples, seqs from `first_seq`."""
    return {"family": family, "metric": metric, "feature": "dti",
            "severity_unit": "share of applications", "expected_operating_severity": expected,
            "points": [{"severity": float(s), "point": float(p), "lo": float(lo),
                        "hi": float(lo) + 0.02, "level": 0.95, "n": 100, "n_boot": 50,
                        "method": "percentile bootstrap", "metric": metric,
                        "label": f"{family}@{s:g}", "seq": first_seq + i}
                       for i, (s, p, lo) in enumerate(points)]}


def _sealed(floor=0.86, seq=0, metric="accuracy", rationale=FLOOR_RATIONALE):
    payload = {"metric": metric, "floor": float(floor), "rationale": rationale,
               "as_of": L.AS_OF.isoformat()}
    return {"seq": seq, **payload, "digest": L.digest_of(payload)}


_TINY_SIZES = (200, 200, 200, 200, 80, 80)


def _separable_population(sizes=(60, 60, 60, 40), seed=11):
    """Two far-apart clusters, labelled by cluster: every neighbourhood agrees perfectly.

    `neighbour_agreement` is 1.0 for every row, so NOTHING is flagged and every batch rate
    and every baseline interval collapses to exactly [0.0, 0.0]. That is the only way to put
    a batch's `lo` exactly level with its baseline's `hi` on a real scan, which is the
    boundary `lo > baseline_hi` turns on.
    """
    rng = np.random.default_rng(seed)
    total = sum(sizes)
    side = np.tile([0, 1], total // 2 + 1)[:total]
    features = {"income": 52000.0 + side * 400000.0 + rng.normal(0.0, 500.0, total),
                "dti": 0.34 + side * 4.0 + rng.normal(0.0, 0.005, total),
                "months_employed": 62.0 + side * 600.0 + rng.normal(0.0, 1.0, total),
                "prior_defaults": 0.35 + side * 8.0 + rng.normal(0.0, 0.01, total)}
    batches = np.concatenate([np.full(size, f"c{i}") for i, size in enumerate(sizes)])
    return features, side.astype(int), batches


def _tiny_population(flip_shares, sizes=_TINY_SIZES, seed=7):
    """A small labelled set whose labels follow a neighbourhood-consistent rule, then flipped.

    The rule uses all four standardised features, as the lesson's own generator does, so a
    row's nearest neighbours genuinely tend to share its label and a flipped label stands out.
    A rule that read one feature would give the detector nothing to find.
    """
    rng = np.random.default_rng(seed)
    total = sum(sizes)
    features = {"income": rng.normal(52000.0, 21000.0, total),
                "dti": rng.normal(0.34, 0.14, total),
                "months_employed": rng.normal(62.0, 46.0, total),
                "prior_defaults": rng.normal(0.35, 0.70, total)}
    labels = (L.standardise(features) @ np.array([1.0, -1.0, 1.0, -1.0]) > 0.0).astype(int)
    batches = np.concatenate([np.full(size, f"b{i}") for i, size in enumerate(sizes)])
    for i, share in enumerate(flip_shares):
        rows = np.flatnonzero(batches == f"b{i}")
        chosen = rows[rng.random(len(rows)) < share]
        labels[chosen] = 1 - labels[chosen]
    return features, labels, batches


# --- exercise 1: the paired bootstrap -----------------------------------------------------

def test_bootstrap_pairs_labels_with_predictions():
    t = np.tile([1, 0], 60)
    perfect = L.bootstrap_ci(t, t, n_boot=200)
    assert perfect["lo"] == 1.0 and perfect["hi"] == 1.0 and perfect["point"] == 1.0, (
        f"a model right about every row has a degenerate interval [1.0, 1.0]; you returned "
        f"[{perfect['lo']}, {perfect['hi']}]. Separate index draws for y_true and y_pred put "
        "this near 0.5"
    )
    worst = L.bootstrap_ci(t, 1 - t, n_boot=200)
    assert (worst["point"], worst["lo"], worst["hi"]) == (0.0, 0.0, 0.0), (
        "a model wrong about every row is 0.0 with a degenerate interval; unpaired resampling "
        f"puts it near 0.5 as well. You returned {worst['point']}, "
        f"[{worst['lo']}, {worst['hi']}]"
    )
    rng = np.random.default_rng(11)
    truth = (rng.random(300) < 0.5).astype(int)
    guess = np.where(rng.random(300) < 0.3, 1 - truth, truth)
    ci = L.bootstrap_ci(truth, guess, n_boot=400)
    lo, hi = _py_bootstrap(truth, guess, L.accuracy, 400)
    assert abs(ci["lo"] - lo) < 1e-12 and abs(ci["hi"] - hi) < 1e-12, (
        f"expected [{lo:.6f}, {hi:.6f}] and got [{ci['lo']:.6f}, {ci['hi']:.6f}] — index both "
        "arrays with the SAME (n_boot, n) matrix"
    )


def test_bootstrap_draw_is_the_documented_one():
    rng = np.random.default_rng(3)
    truth = (rng.random(250) < 0.4).astype(int)
    guess = np.where(rng.random(250) < 0.2, 1 - truth, truth)
    for n_boot in (120, 350):
        ci = L.bootstrap_ci(truth, guess, n_boot=n_boot)
        lo, hi = _py_bootstrap(truth, guess, L.accuracy, n_boot)
        assert abs(ci["lo"] - lo) < 1e-12 and abs(ci["hi"] - hi) < 1e-12, (
            f"at B={n_boot} expected [{lo:.6f}, {hi:.6f}], got "
            f"[{ci['lo']:.6f}, {ci['hi']:.6f}]. Take the whole index matrix in ONE "
            "rng.integers call with size=(n_boot, n): an evidence pack whose intervals move "
            "between machines is not evidence"
        )
    again = L.bootstrap_ci(truth, guess, n_boot=350)
    assert (again["lo"], again["hi"]) == (ci["lo"], ci["hi"]), (
        "two calls with the same seed must agree exactly — build the generator inside the "
        "function from `seed`, not from numpy's global state"
    )
    shifted = L.bootstrap_ci(truth, guess, n_boot=350, seed=L.BOOT_SEED + 1)
    assert (shifted["lo"], shifted["hi"]) != (ci["lo"], ci["hi"]), (
        "the seed argument must reach the generator; yours appears to be ignored"
    )


def test_bootstrap_point_is_the_full_sample_statistic():
    rng = np.random.default_rng(5)
    truth = (rng.random(200) < 0.35).astype(int)
    guess = np.where(rng.random(200) < 0.15, 1 - truth, truth)
    ci = L.bootstrap_ci(truth, guess, n_boot=300)
    assert ci["point"] == _py_accuracy(truth, guess), (
        f"point must be the metric on the FULL sample ({_py_accuracy(truth, guess):.10f}); "
        f"you returned {ci['point']:.10f}. The mean of the replicates is a different number "
        "and it is not the one that goes into the instructions for use"
    )
    r_ci = L.bootstrap_ci(truth, guess, metric=L.recall, n_boot=200)
    assert r_ci["point"] == float(L.recall(truth, guess)), (
        "the `metric` argument must be used rather than accuracy being hard-coded"
    )
    lo, hi = _py_bootstrap(truth, guess, L.recall, 200)
    assert abs(r_ci["lo"] - lo) < 1e-12 and abs(r_ci["hi"] - hi) < 1e-12, (
        "the interval must come from the SAME metric as the point"
    )


def test_bootstrap_level_and_reported_metadata():
    rng = np.random.default_rng(9)
    truth = (rng.random(240) < 0.5).astype(int)
    guess = np.where(rng.random(240) < 0.25, 1 - truth, truth)
    wide = L.bootstrap_ci(truth, guess, n_boot=300, level=0.95)
    narrow = L.bootstrap_ci(truth, guess, n_boot=300, level=0.50)
    lo, hi = _py_bootstrap(truth, guess, L.accuracy, 300, level=0.50)
    assert abs(narrow["lo"] - lo) < 1e-12 and abs(narrow["hi"] - hi) < 1e-12, (
        f"a 50% interval runs from the 25th to the 75th percentile: expected "
        f"[{lo:.6f}, {hi:.6f}], got [{narrow['lo']:.6f}, {narrow['hi']:.6f}]. Place the "
        "percentiles from `level` rather than hard-coding 2.5 and 97.5"
    )
    assert narrow["hi"] - narrow["lo"] < wide["hi"] - wide["lo"], (
        "a 50% interval has to be narrower than a 95% one"
    )
    assert (wide["n"], wide["n_boot"], wide["level"]) == (240, 300, 0.95), (
        f"echo back n, n_boot and level; you returned {wide['n']}, {wide['n_boot']}, "
        f"{wide['level']}. An interval nobody can reproduce is decoration"
    )
    assert wide["method"] == "percentile bootstrap", (
        f"method must say what was done: 'percentile bootstrap', not {wide['method']!r}"
    )


def test_bootstrap_edge_cases():
    empty = L.bootstrap_ci(np.array([]), np.array([]), n_boot=50)
    assert math.isnan(empty["point"]) and (empty["lo"], empty["hi"]) == (0.0, 1.0), (
        f"no rows means no measurement: nan and [0.0, 1.0]. You returned {empty['point']!r} "
        f"[{empty['lo']}, {empty['hi']}]; a zero-width interval there claims certainty about "
        "a measurement nobody made"
    )
    none_boot = L.bootstrap_ci(np.array([1, 0, 1]), np.array([1, 0, 1]), n_boot=0)
    assert math.isnan(none_boot["point"]) and (none_boot["lo"], none_boot["hi"]) == (0.0, 1.0), (
        "no resamples is the same blind answer as no rows"
    )
    flags = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    single = L.bootstrap_ci(flags, None, metric=L.METRICS["rate"], n_boot=100)
    assert single["point"] == 0.3 and single["n"] == 10, (
        f"y_pred=None means set y_pred = y_true, not raise; you returned {single['point']!r}"
    )
    try:
        L.bootstrap_ci(np.array([1, 0, 1]), np.array([1, 0]), n_boot=10)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a pairing of 3 labels with 2 predictions was accepted; a paired bootstrap needs "
            "one prediction per label and must say so rather than broadcasting"
        )


# --- exercise 2: declaring the bound ------------------------------------------------------

def _good_ci(**over):
    base = {"point": 0.89, "lo": 0.87, "hi": 0.91, "level": 0.95, "n": 1200,
            "n_boot": 2000, "method": "percentile bootstrap"}
    base.update(over)
    return base


def test_declare_picks_the_defensible_end():
    up = L.declare("accuracy", _good_ci())
    assert up["declared"] == 0.87, (
        f"higher-is-better declares the LOWER bound 0.87; you declared {up['declared']}"
    )
    down = L.declare("false_positive_rate", _good_ci(), higher_is_better=False)
    assert down["declared"] == 0.91, (
        f"lower-is-better declares the UPPER bound 0.91; you declared {down['declared']}. "
        "Using lo for both publishes the flattering end of your own uncertainty twice"
    )
    assert up["direction"] == "higher_is_better", f"got {up['direction']!r}"
    assert down["direction"] == "lower_is_better", f"got {down['direction']!r}"
    assert abs(up["width"] - 0.04) < 1e-9, (
        f"width is hi - lo = 0.04; you returned {up['width']}"
    )
    narrow = L.declare("accuracy", _good_ci(lo=0.885, hi=0.895))
    assert narrow["declared"] == 0.885 and narrow["width"] < up["width"], (
        "the declared value and the width both follow the interval you were handed"
    )
    # A ZERO-WIDTH interval is not a pathology here: it is what this lesson's own bootstrap
    # returns for a model that is right about every row, and section 4's pairing demo prints
    # exactly that. `point` sitting ON a bound is inside its interval, and `hi == lo` is not
    # an inversion. Mutation testing found both refusals scoring full marks.
    exact = _good_ci(point=1.0, lo=1.0, hi=1.0)
    flat = L.declare("accuracy", exact)
    assert flat["declared"] == 1.0 and flat["width"] == 0.0, (
        f"a zero-width interval declares its own value with width 0.0; you returned "
        f"declared={flat['declared']} width={flat['width']}. This is the interval "
        "bootstrap_ci() hands you for a model that is right about every row — refusing it "
        "with `lo < point < hi` or `hi <= lo` rejects a measurement the lesson itself makes"
    )
    edge = L.declare("false_positive_rate", _good_ci(point=0.87, lo=0.87, hi=0.91),
                     higher_is_better=False)
    assert edge["declared"] == 0.91, (
        f"a point sitting exactly ON the lower bound is inside [lo, hi], so this interval is "
        f"declarable and the lower-is-better end is 0.91; you returned {edge['declared']}"
    )


def test_declare_carries_the_provenance_and_leaves_the_input_alone():
    ci = _good_ci()
    snapshot = dict(ci)
    out = L.declare("accuracy", ci)
    assert ci == snapshot, (
        "declare() must not write into the interval it was handed; a measurement edited by "
        "the thing that reports it is not a measurement"
    )
    assert out["name"] == "accuracy", f"the name travels with the declaration; got {out['name']!r}"
    assert (out["n"], out["n_boot"], out["method"], out["level"]) == \
        (1200, 2000, "percentile bootstrap", 0.95), (
        "the sample size, the resample count, the method and the level are part of the claim"
    )
    assert (out["point"], out["lo"], out["hi"]) == (0.89, 0.87, 0.91), (
        "keep the point and both ends: the reader needs all three to check the declaration"
    )
    # The defensible end again, on a second interval, so this distinction is not carried by
    # one hidden case alone — mutation testing found it resting on a single test.
    wide = _good_ci(point=0.40, lo=0.10, hi=0.70)
    assert L.declare("recall", wide)["declared"] == 0.10, (
        "higher-is-better still declares lo on a wide interval"
    )
    assert L.declare("false_positive_rate", wide, higher_is_better=False)["declared"] == 0.70, (
        "and lower-is-better still declares hi. A declaration that always publishes lo reads "
        "a 10%-to-70% false positive rate as a promise of 10%"
    )


def test_declare_refuses_anything_that_is_not_an_interval():
    bad_cases = {
        "a bare float": 0.89,
        "a string": "0.89",
        "no lo and hi": {"point": 0.89, "level": 0.95, "n": 5, "n_boot": 5, "method": "b"},
        "no hi": _shorn("hi"),
        "no level": _shorn("level"),
        "no n": _shorn("n"),
        "no n_boot": _shorn("n_boot"),
        "no method": _shorn("method"),
        "a nan lo": _good_ci(lo=float("nan")),
        "a nan point": _good_ci(point=float("nan")),
        "a None hi": _good_ci(hi=None),
        "an inverted interval": _good_ci(lo=0.95, hi=0.80),
        "a point above hi": _good_ci(point=0.99),
        "a point below lo": _good_ci(point=0.10),
    }
    for label, bad in bad_cases.items():
        try:
            L.declare("accuracy", bad)
        except L.UndeclarableMetricError:
            continue
        except NotImplementedError:
            raise                       # an unfilled stub is a TODO, not a wrong answer
        except Exception as exc:
            raise AssertionError(
                f"declare() raised {type(exc).__name__} for {label}; it must raise "
                "UndeclarableMetricError so a caller can tell a refusal from a crash"
            )
        raise AssertionError(
            f"declare() accepted {label}. Every one of these is a number somebody could "
            "publish and nobody could check, which is the whole point of the refusal"
        )


def _shorn(key):
    ci = _good_ci()
    ci.pop(key)
    return ci


# --- exercise 3: sealing the floor --------------------------------------------------------

def test_seal_floor_records_payload_and_digest():
    led = L.EvidenceLedger()
    sealed = L.seal_floor(led, "accuracy", 0.86, FLOOR_RATIONALE)
    payload = {"metric": "accuracy", "floor": 0.86, "rationale": FLOOR_RATIONALE,
               "as_of": L.AS_OF.isoformat()}
    assert sealed["digest"] == L.digest_of(payload), (
        "the digest must be digest_of({metric, floor, rationale, as_of}) and nothing else, or "
        "the crossing report cannot recompute it"
    )
    assert {k: sealed[k] for k in payload} == payload, (
        f"return the payload fields themselves; you returned {sealed!r}"
    )
    assert sealed["seq"] == 0, f"the first entry is seq 0; you returned {sealed['seq']}"
    entries = led.find(kind="floor")
    assert len(entries) == 1 and entries[0]["payload"] == payload, (
        f"the seal must APPEND to the ledger with exactly that payload; the ledger holds "
        f"{[e['payload'] for e in entries]}"
    )
    second = L.EvidenceLedger()
    second.append("note", {"metric": "accuracy", "text": "unrelated"})
    later = L.seal_floor(second, "accuracy", 0.80, FLOOR_RATIONALE)
    assert later["seq"] == 1, (
        "an entry that is not a measurement must not block the seal, and the seq is the "
        "ledger's own, not always zero"
    )


def test_seal_floor_refuses_a_floor_chosen_after_the_fact():
    led = L.EvidenceLedger()
    led.append("measurement", {"metric": "accuracy", "label": "baseline", "point": 0.89})
    try:
        L.seal_floor(led, "accuracy", 0.80, FLOOR_RATIONALE)
    except L.FloorAfterTheFactError:
        pass
    else:
        raise AssertionError(
            "a floor was sealed for a metric the ledger had already measured. That is the one "
            "thing this function exists to refuse, and it is the heart of the module"
        )
    other = L.EvidenceLedger()
    other.append("measurement", {"metric": "recall", "label": "baseline", "point": 0.90})
    ok = L.seal_floor(other, "accuracy", 0.86, FLOOR_RATIONALE)
    assert ok["seq"] == 1, (
        "a measurement of a DIFFERENT metric must not block this floor — match on the metric "
        "inside the payload, not merely on the entry kind"
    )
    twice = L.EvidenceLedger()
    L.seal_floor(twice, "accuracy", 0.86, FLOOR_RATIONALE)
    try:
        L.seal_floor(twice, "accuracy", 0.70, FLOOR_RATIONALE)
    except L.FloorAfterTheFactError:
        pass
    else:
        raise AssertionError(
            "a SECOND floor for the same metric was accepted. Seal 0.86, look at the sweep, "
            "seal 0.70: the ordering check is defeated one entry later unless you refuse this"
        )
    assert len(twice.find(kind="floor")) == 1, (
        "the refused second seal must not have been appended before it was refused"
    )


def test_seal_floor_validates_its_arguments():
    for label, rationale in (("empty", ""), ("whitespace", "    "),
                             ("too short", "x" * (L.MIN_RATIONALE_CHARS - 1)),
                             ("not a string", None), ("a number", 3)):
        try:
            L.seal_floor(L.EvidenceLedger(), "accuracy", 0.86, rationale)
        except ValueError:
            continue
        except L.FloorAfterTheFactError:
            raise AssertionError(f"a {label} rationale raised the wrong error; use ValueError")
        raise AssertionError(
            f"a floor with a {label} rationale was accepted; a bare number is exactly what "
            "later gets described as principled"
        )
    padded = "  " + "x" * (L.MIN_RATIONALE_CHARS - 4) + " " * 8
    assert len(padded) > L.MIN_RATIONALE_CHARS > len(padded.strip()), "fixture sanity"
    try:
        L.seal_floor(L.EvidenceLedger(), "accuracy", 0.86, padded)
        raise AssertionError(
            f"a rationale of {len(padded)} characters that is only {len(padded.strip())} once "
            "stripped was accepted. The length check is on the STRIPPED text — padding a short "
            "reason out to length is the same bare number with whitespace in front of it"
        )
    except ValueError:
        pass
    at_the_limit = L.seal_floor(L.EvidenceLedger(), "accuracy", 0.86,
                                "y" * L.MIN_RATIONALE_CHARS)
    assert at_the_limit["floor"] == 0.86, (
        f"a rationale of exactly MIN_RATIONALE_CHARS ({L.MIN_RATIONALE_CHARS}) is long "
        "enough; the floor is inclusive"
    )
    for bad in (1.4, -0.1, 2.0):
        try:
            L.seal_floor(L.EvidenceLedger(), "accuracy", bad, FLOOR_RATIONALE)
        except ValueError:
            continue
        raise AssertionError(f"a floor of {bad} was accepted; an accuracy floor lives in [0, 1]")
    for fine in (0.0, 1.0):
        assert L.seal_floor(L.EvidenceLedger(), "accuracy", fine,
                            FLOOR_RATIONALE)["floor"] == fine, (
            f"{fine} is inside [0, 1] and must be accepted; the bounds are inclusive"
        )


# --- exercise 4: the sweep ----------------------------------------------------------------

def test_curve_measures_every_declared_severity_in_order():
    led, floor = _fresh_floor()
    curve = L.perturbation_curve(led, "unit_change", L.TEST_FEATURES, L.TEST_LABELS, n_boot=40)
    spec = L.PERTURBATIONS["unit_change"]
    assert [p["severity"] for p in curve["points"]] == [float(s) for s in spec["severities"]], (
        f"one point per declared severity, in the declared order; you returned "
        f"{[p['severity'] for p in curve['points']]}"
    )
    for key, expected in (("family", "unit_change"), ("metric", "accuracy"),
                          ("feature", spec["feature"]), ("severity_unit", spec["unit"]),
                          ("expected_operating_severity",
                           float(spec["expected_operating_severity"]))):
        assert curve[key] == expected, (
            f"curve[{key!r}] should be {expected!r} and is {curve.get(key)!r}; severities "
            "mean nothing without their unit and the severity the system expects to meet"
        )
    other = L.perturbation_curve(led, "sensor_noise", L.TEST_FEATURES, L.TEST_LABELS,
                                 n_boot=40, metric_name="recall")
    assert other["metric"] == "recall" and other["family"] == "sensor_noise", (
        "the family and the metric are arguments, not constants"
    )


def test_curve_perturbs_the_inputs_and_leaves_the_fixture_alone():
    led, floor = _fresh_floor()
    curve = L.perturbation_curve(led, "missing_field", L.TEST_FEATURES, L.TEST_LABELS, n_boot=40)
    for key, column in _PRISTINE_FEATURES.items():
        assert np.array_equal(np.asarray(L.TEST_FEATURES[key], dtype=float), column), (
            f"TEST_FEATURES[{key!r}] no longer matches the values section 3 generated. The "
            "apply functions hand you a fresh dict; writing into the one you were given "
            "leaves every later severity measured on damage the earlier ones did — and "
            "leaves every later cell in the notebook running on a table that is not the one "
            "it prints"
        )
    assert np.array_equal(np.asarray(L.TEST_LABELS), _PRISTINE_LABELS), (
        "the outcomes those applicants actually had do not change when their inputs are "
        "damaged; y_true is constant across the sweep"
    )
    spec = L.PERTURBATIONS["missing_field"]
    assert abs(curve["points"][0]["point"]
               - _py_accuracy(_PRISTINE_LABELS, L.predict(_PRISTINE_FEATURES))) < 1e-12, (
        "severity 0.0 is the control and must reproduce the undamaged baseline exactly"
    )
    worst = spec["severities"][-1]
    expected = _py_accuracy(_PRISTINE_LABELS,
                            L.predict(spec["apply"](_PRISTINE_FEATURES, worst)))
    assert abs(curve["points"][-1]["point"] - expected) < 1e-12, (
        f"at severity {worst:g} the accuracy is {expected:.6f} and you reported "
        f"{curve['points'][-1]['point']:.6f}. Apply the family's own function and re-run "
        "predict() on the damaged inputs"
    )
    # `metric_name` is a parameter and two of METRICS are ASYMMETRIC, so the order the
    # measurement is handed (y_true, y_pred) is observable. accuracy alone cannot see it:
    # a curve that measures metric(y_pred, y_true) scores full marks on every accuracy case.
    led2, _ = _fresh_floor()
    rec = L.perturbation_curve(led2, "missing_field", L.TEST_FEATURES, L.TEST_LABELS,
                               metric_name="recall", n_boot=40)
    damaged = spec["apply"](_PRISTINE_FEATURES, worst)
    want = float(L.METRICS["recall"](_PRISTINE_LABELS, L.predict(damaged)))
    swapped = float(L.METRICS["recall"](L.predict(damaged), _PRISTINE_LABELS))
    assert abs(rec["points"][-1]["point"] - want) < 1e-12, (
        f"at severity {worst:g} the RECALL is {want:.6f} and you reported "
        f"{rec['points'][-1]['point']:.6f}"
        + (f" — which is recall(y_pred, y_true) = {swapped:.6f}, the arguments the wrong way "
           "round. Recall is not symmetric: the labels go first, the predictions second"
           if abs(rec["points"][-1]["point"] - swapped) < 1e-12 else "")
    )
    assert rec["metric"] == "recall", (
        f"the curve reports the metric it was asked for; you returned {rec['metric']!r}"
    )


def test_curve_records_every_point_through_the_ledger():
    led, floor = _fresh_floor()
    before = len(led.entries)
    curve = L.perturbation_curve(led, "stale_feature", L.TEST_FEATURES, L.TEST_LABELS, n_boot=30)
    grown = len(led.entries) - before
    assert grown == len(curve["points"]), (
        f"the ledger grew by {grown} for {len(curve['points'])} points. Go through measure(): "
        "calling bootstrap_ci directly leaves no sequence number and no ordering evidence, "
        "which is the only thing that makes the floor checkable"
    )
    seqs = [p["seq"] for p in curve["points"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) and min(seqs) > floor["seq"], (
        f"each point carries its own ascending ledger seq, all after the floor at "
        f"{floor['seq']}; you returned {seqs}"
    )
    labels = [e["payload"]["label"] for e in led.find(kind="measurement")]
    assert labels == [f"stale_feature@{s:g}" for s in L.PERTURBATIONS["stale_feature"]["severities"]], (
        f"label each measurement f'{{family}}@{{severity:g}}' so the ledger reads as a sweep; "
        f"the ledger holds {labels}"
    )
    lo, hi = _py_bootstrap(_PRISTINE_LABELS,
                           L.predict(L.PERTURBATIONS["stale_feature"]["apply"](
                               _PRISTINE_FEATURES, 0.25)), L.accuracy, 30)
    at_025 = next(p for p in curve["points"] if p["severity"] == 0.25)
    assert abs(at_025["lo"] - lo) < 1e-12 and abs(at_025["hi"] - hi) < 1e-12, (
        "each point carries the interval for its own severity; pass n_boot through to "
        "measure() rather than falling back on the default"
    )


# --- exercise 5: the crossing -------------------------------------------------------------

def test_crossing_uses_the_lower_bound_not_the_point():
    curve = _hand_curve([(0.0, 0.92, 0.90), (0.10, 0.90, 0.88), (0.20, 0.89, 0.85),
                         (0.40, 0.88, 0.82), (0.80, 0.84, 0.79)], expected=0.10)
    rep = L.crossing_severity(curve, _sealed(0.86))
    assert rep["crossing_severity"] == 0.20, (
        f"the lower bound first falls below 0.86 at severity 0.20 (0.85 < 0.86); you reported "
        f"{rep['crossing_severity']}"
    )
    assert rep["point_crossing_severity"] == 0.80, (
        f"the POINT first falls below 0.86 at severity 0.80 (0.84 < 0.86); you reported "
        f"{rep['point_crossing_severity']}. Both are reported, and the gap between them is "
        "the size of the claim the point reading would have been making"
    )
    assert rep["floor"] == 0.86 and rep["floor_seq"] == 0 and rep["first_measurement_seq"] == 1, (
        f"carry the floor and both sequence numbers: they are what a reviewer checks. You "
        f"returned floor={rep['floor']} floor_seq={rep['floor_seq']} "
        f"first={rep['first_measurement_seq']}"
    )
    assert rep["severity_unit"] == curve["severity_unit"] and rep["family"] == "missing_field", (
        "the unit and the family travel with the crossing, because 0.20 of one family is not "
        "0.20 of another"
    )
    # A bound that lands exactly ON the floor has not fallen through it. Nothing on the real
    # sweep ties, so only a hand-built curve can pin this; mutation testing found `lo <= floor`
    # scoring full marks.
    level = _hand_curve([(0.0, 0.92, 0.90), (0.10, 0.89, 0.86), (0.20, 0.87, 0.84)],
                        expected=0.10)
    on_the_floor = L.crossing_severity(level, _sealed(0.86))
    assert on_the_floor["crossing_severity"] == 0.20, (
        f"at severity 0.10 the lower bound is 0.86, which is the floor exactly and is not "
        f"BELOW it; the first bound that falls through is 0.84 at severity 0.20. You reported "
        f"{on_the_floor['crossing_severity']} — a floor of 0.86 that a measured 0.86 breaches "
        "declares a failure the evidence does not show"
    )
    never = _hand_curve([(0.0, 0.92, 0.90), (0.10, 0.91, 0.89), (0.20, 0.90, 0.88)],
                        expected=0.10)
    clean = L.crossing_severity(never, _sealed(0.86))
    assert clean["crossing_severity"] is None and clean["point_crossing_severity"] is None, (
        "a curve that never falls through the floor reports None, not the last severity"
    )
    assert clean["steps_of_headroom"] is None and clean["verdict"] == "pass", (
        "no crossing means no headroom figure and a pass; a report hard-wired to 'fail' "
        "carries no information at all"
    )


def test_crossing_refuses_a_floor_that_is_not_prior():
    points = [(0.0, 0.92, 0.90), (0.10, 0.88, 0.85), (0.20, 0.84, 0.81)]
    curve = _hand_curve(points, expected=0.10, first_seq=4)
    assert L.crossing_severity(curve, _sealed(0.86, seq=3))["crossing_severity"] == 0.10, (
        "a floor sealed at seq 3 before a curve starting at seq 4 is prior and must be "
        "reported on"
    )
    for label, seq in (("at the first measurement", 4), ("in the middle", 5),
                       ("after the whole curve", 9)):
        try:
            L.crossing_severity(curve, _sealed(0.86, seq=seq))
        except L.FloorAfterTheFactError:
            continue
        raise AssertionError(
            f"a floor sealed {label} (seq {seq}) was accepted against a curve measured at "
            "seqs 4-6. Compare against the MINIMUM seq in the curve: a floor that is merely "
            "before the last point was still chosen after the first results came in"
        )


def test_crossing_refuses_a_tampered_floor_or_the_wrong_metric():
    curve = _hand_curve([(0.0, 0.92, 0.90), (0.10, 0.88, 0.85)], expected=0.10)
    honest = _sealed(0.86)
    for label, tampered in (("the floor value", {**honest, "floor": 0.70}),
                            ("the rationale", {**honest, "rationale": "revised after review"}),
                            ("the date", {**honest, "as_of": "2027-01-01"}),
                            ("the metric name", {**honest, "metric": "accuracy "})):
        try:
            L.crossing_severity(curve, tampered)
        except L.TamperedFloorError:
            continue
        except ValueError:
            if label == "the metric name":
                continue
            raise AssertionError(f"editing {label} raised ValueError, not TamperedFloorError")
        raise AssertionError(
            f"a floor whose {label} was edited after sealing was accepted. Recompute "
            "digest_of({metric, floor, rationale, as_of}) and compare it with the digest you "
            "were handed"
        )
    try:
        L.crossing_severity({**curve, "metric": "recall"}, honest)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "an accuracy floor was applied to a recall curve; a floor governs one metric"
        )


def test_crossing_headroom_and_verdict_against_the_operating_point():
    points = [(0.0, 0.92, 0.91), (0.10, 0.90, 0.89), (0.20, 0.89, 0.87),
              (0.40, 0.87, 0.85), (0.80, 0.82, 0.80)]
    beyond = L.crossing_severity(_hand_curve(points, expected=0.10), _sealed(0.86))
    assert beyond["crossing_severity"] == 0.40 and beyond["steps_of_headroom"] == 2, (
        f"the crossing at 0.40 is two grid steps beyond an expected 0.10; you reported "
        f"{beyond['steps_of_headroom']} steps. Count grid positions, not the arithmetic "
        "difference of the severities"
    )
    assert beyond["verdict"] == "pass", "a crossing beyond the operating point passes"
    at = L.crossing_severity(_hand_curve(points, expected=0.40), _sealed(0.86))
    assert at["steps_of_headroom"] == 0 and at["verdict"] == "fail", (
        f"crossing AT the severity the system expects to meet is zero headroom and a fail; "
        f"you reported {at['steps_of_headroom']} and {at['verdict']!r}"
    )
    inside = L.crossing_severity(_hand_curve(points, expected=0.80), _sealed(0.86))
    assert inside["steps_of_headroom"] == -1 and inside["verdict"] == "fail", (
        f"a crossing one step BELOW the operating point is negative headroom and a fail; you "
        f"reported {inside['steps_of_headroom']} and {inside['verdict']!r}"
    )
    lower = L.crossing_severity(_hand_curve(points, expected=0.10), _sealed(0.81))
    assert lower["crossing_severity"] == 0.80 and lower["floor"] == 0.81, (
        f"with a floor of 0.81 the only lower bound below it is 0.80, at severity 0.80; you "
        f"reported a crossing at {lower['crossing_severity']} against a floor of "
        f"{lower['floor']}. The floor is read from the sealed record, not from a constant in "
        "your code"
    )


def test_crossing_on_the_real_sweep():
    led, floor = _fresh_floor(floor=L.ACCURACY_FLOOR)
    curve = L.perturbation_curve(led, "missing_field", L.TEST_FEATURES, L.TEST_LABELS,
                                 n_boot=L.N_BOOT_SWEEP)
    rep = L.crossing_severity(curve, floor)
    spec = L.PERTURBATIONS["missing_field"]
    expected_cross = expected_point = None
    for severity in spec["severities"]:
        y_pred = L.predict(spec["apply"](_PRISTINE_FEATURES, severity))
        lo, _hi = _py_bootstrap(_PRISTINE_LABELS, y_pred, L.accuracy, L.N_BOOT_SWEEP)
        if expected_cross is None and lo < L.ACCURACY_FLOOR:
            expected_cross = float(severity)
        if expected_point is None and _py_accuracy(_PRISTINE_LABELS, y_pred) < L.ACCURACY_FLOOR:
            expected_point = float(severity)
    assert rep["crossing_severity"] == expected_cross, (
        f"on the real sweep the bound falls through at {expected_cross}; you reported "
        f"{rep['crossing_severity']}"
    )
    assert rep["point_crossing_severity"] == expected_point, (
        f"and the point falls through at {expected_point}; you reported "
        f"{rep['point_crossing_severity']}"
    )
    assert rep["crossing_severity"] < rep["point_crossing_severity"], (
        "on this fixture the bound gives way a full grid step before the point does, which is "
        "the whole reason the lesson declares the bound"
    )
    # The ordering refusal on a REAL curve as well as a hand-built one. Mutation testing found
    # the "compare against the minimum seq" rule resting on a single hidden case, and it is the
    # rule the whole module turns on.
    late_ledger = L.EvidenceLedger()
    late_curve = L.perturbation_curve(late_ledger, "unit_change", L.TEST_FEATURES,
                                      L.TEST_LABELS, n_boot=30)
    mid = late_curve["points"][1]["seq"]
    try:
        L.crossing_severity(late_curve, _sealed(L.ACCURACY_FLOOR, seq=mid))
    except L.FloorAfterTheFactError:
        pass
    else:
        raise AssertionError(
            f"a floor sealed at seq {mid}, after the first point of the sweep it governs, was "
            "accepted on the real curve. Comparing against the LAST measurement instead of "
            "the first lets a floor be chosen once most of the results are in, which is "
            "exactly how it happens"
        )


# --- exercise 6: the poisoning scan -------------------------------------------------------

def test_poison_rates_match_an_independent_derivation():
    rep = L.poison_scan(L.TRAIN_FEATURES, L.RECORDED_LABELS, L.TRAIN_BATCHES, n_boot=200)
    flags = (L.neighbour_agreement(L.TRAIN_FEATURES, L.RECORDED_LABELS, L.K_NEIGHBOURS)
             < L.NEIGHBOUR_AGREEMENT_FLOOR).astype(int)
    assert rep["flagged_rows"] == int(flags.sum()), (
        f"{int(flags.sum())} rows sit under the agreement floor; you counted "
        f"{rep['flagged_rows']}"
    )
    assert abs(rep["overall"]["point"] - _py_rate(flags)) < 1e-12, (
        "the overall rate is the flag rate over every row"
    )
    assert set(rep["batches"]) == set(L.BATCH_IDS), (
        f"every batch gets a row, including the ones too small to judge; you reported "
        f"{sorted(rep['batches'])}"
    )
    for batch_id in ("b00", "b06", "b09"):
        rows = np.asarray(L.TRAIN_BATCHES) == batch_id
        cell = rep["batches"][batch_id]
        assert cell["n"] == int(rows.sum()), (
            f"{batch_id} holds {int(rows.sum())} rows; you reported {cell['n']}"
        )
        assert abs(cell["rate"]["point"] - _py_rate(flags[rows])) < 1e-12, (
            f"{batch_id}'s flag rate should be {_py_rate(flags[rows]):.6f}"
        )
    assert rep["k"] == L.K_NEIGHBOURS and rep["min_support"] == L.MIN_SUPPORT, (
        "report the k and the support floor the scan was run at: another k is another scan"
    )


def test_poison_support_floor_is_the_parameter_not_a_special_case():
    rep = L.poison_scan(L.TRAIN_FEATURES, L.RECORDED_LABELS, L.TRAIN_BATCHES, n_boot=200)
    tiny = rep["batches"]["b09"]
    assert tiny["n"] == 20 and tiny["status"] == "insufficient_support", (
        "b09 has 20 rows and the loudest flag rate in the table. The support floor is applied "
        "BEFORE any comparison, exactly as the subgroup floor was in P01-L04"
    )
    assert tiny["supported"] is False and "b09" in rep["insufficient"], (
        "it must still be NAMED: 'this batch cannot be examined' is itself a finding about "
        "the data, and dropping it reads as a clean bill of health"
    )
    assert "b09" not in rep["findings"], "the 20-row batch is not a finding"
    lowered = L.poison_scan(L.TRAIN_FEATURES, L.RECORDED_LABELS, L.TRAIN_BATCHES,
                            min_support=10, n_boot=200)
    assert lowered["batches"]["b09"]["status"] == "poisoned" and "b09" in lowered["findings"], (
        "with min_support=10 the same 20-row batch clears the floor and becomes a finding. If "
        "it does not, min_support is not really a parameter — you have special-cased b09 or "
        "dropped small batches from the report"
    )
    raised = L.poison_scan(L.TRAIN_FEATURES, L.RECORDED_LABELS, L.TRAIN_BATCHES,
                           min_support=200, n_boot=200)
    assert raised["batches"]["b06"]["status"] == "insufficient_support", (
        "with a 200-row floor the 120-row poisoned batch also becomes unexaminable; the floor "
        "is inclusive and applies to every batch alike"
    )
    assert raised["batches"]["b04"]["supported"] is True, (
        "a batch of exactly min_support rows clears the floor: 200 >= 200"
    )
    assert raised["batches"]["b04"]["status"] != "insufficient_support", (
        f"b04 has exactly min_support (200) rows, so it IS examinable and must be judged — "
        f"you reported status {raised['batches']['b04']['status']!r}. `supported` and `status` "
        "have to agree at the boundary: deciding one with >= and the other with <= makes a "
        "batch that the report calls supported and then declines to examine"
    )


def test_poison_compares_intervals_against_a_leave_one_out_baseline():
    rep = L.poison_scan(L.TRAIN_FEATURES, L.RECORDED_LABELS, L.TRAIN_BATCHES, n_boot=300)
    flags = (L.neighbour_agreement(L.TRAIN_FEATURES, L.RECORDED_LABELS, L.K_NEIGHBOURS)
             < L.NEIGHBOUR_AGREEMENT_FLOOR).astype(int)
    for batch_id in ("b02", "b06"):
        rows = np.asarray(L.TRAIN_BATCHES) == batch_id
        cell = rep["batches"][batch_id]
        assert cell["baseline"]["n"] == int((~rows).sum()), (
            f"{batch_id}'s baseline must leave that batch out: "
            f"{int((~rows).sum())} rows, not {cell['baseline']['n']}. A batch inside its own "
            "baseline drags the baseline towards itself and hides exactly what you are hunting"
        )
        assert abs(cell["baseline"]["point"] - _py_rate(flags[~rows])) < 1e-12, (
            f"{batch_id}'s baseline rate should be {_py_rate(flags[~rows]):.6f}"
        )
        expected_mde = ((cell["rate"]["hi"] - cell["rate"]["lo"])
                        + (cell["baseline"]["hi"] - cell["baseline"]["lo"])) / 2.0
        assert abs(cell["min_detectable_excess"] - expected_mde) < 1e-12, (
            f"{batch_id}'s minimum detectable excess is half this batch's interval width plus "
            f"half the baseline's = {expected_mde:.6f}; you reported "
            f"{cell['min_detectable_excess']:.6f}"
        )
    noisy = rep["batches"]["b08"]
    assert noisy["rate"]["point"] > noisy["baseline"]["point"], (
        "b08's flag rate really is above its baseline — this fixture would be pointless if it "
        "were not"
    )
    assert noisy["status"] == "ok" and "b08" not in rep["findings"], (
        "and it is still not a finding, because 90 rows cannot separate it. Comparing point "
        "estimates accuses a vendor on the strength of noise; compare the intervals"
    )
    assert rep["batches"]["b00"]["min_detectable_excess"] < \
        rep["batches"]["b07"]["min_detectable_excess"], (
        "a bigger batch detects a smaller excess; if yours does not, the detection limit is "
        "not coming from the interval widths"
    )


def test_poison_passes_a_clean_set_and_orders_its_findings():
    features, labels, batches = _tiny_population((0.0,) * 6)
    clean = L.poison_scan(features, labels, batches, n_boot=200)
    assert clean["verdict"] == "pass" and clean["findings"] == [], (
        f"nothing is planted in this set and the scan reports {clean['findings']}. A scan "
        "that always fails carries no information at all"
    )
    assert len(clean["batches"]) == len(_TINY_SIZES) and clean["insufficient"] == [], (
        "every batch still gets a row, and none of these is under the support floor"
    )
    features, labels, batches = _tiny_population((0.0, 0.0, 0.0, 0.0, 0.85, 0.55))
    dirty = L.poison_scan(features, labels, batches, n_boot=200)
    assert dirty["findings"] == ["b4", "b5"], (
        f"b4 is flipped at 0.85 and b5 at 0.55, so the findings run worst rate first: "
        f"['b4', 'b5']. You returned {dirty['findings']} — sorting by rate ascending, or by "
        "batch id, puts the smaller problem at the top of a remediation list"
    )
    assert dirty["verdict"] == "fail", "any finding fails the scan"
    assert dirty["batches"]["b4"]["rate"]["lo"] > dirty["batches"]["b4"]["baseline"]["hi"], (
        "a batch is poisoned when its interval CLEARS the baseline's, not when its point "
        "estimate is merely higher"
    )
    # The boundary the real fixture never reaches: a set in which NO row is flagged, so every
    # batch interval and every baseline interval is [0.0, 0.0] and `lo` is level with the
    # baseline's `hi` rather than above it. `>=` here accuses every batch in a clean set.
    sep_f, sep_lab, sep_b = _separable_population()
    spotless = L.poison_scan(sep_f, sep_lab, sep_b, n_boot=100)
    assert spotless["flagged_rows"] == 0, (
        f"fixture sanity: this set is two far-apart clusters labelled by cluster, so every "
        f"row agrees with all {L.K_NEIGHBOURS} of its neighbours; got "
        f"{spotless['flagged_rows']} flagged"
    )
    assert spotless["findings"] == [] and spotless["verdict"] == "pass", (
        f"no row in this set is flagged, so every batch rate and every baseline is exactly "
        f"[0.0, 0.0] and no interval CLEARS another. You reported {spotless['findings']} — a "
        "batch is poisoned when its lo is strictly ABOVE the baseline's hi; level with it is "
        "not above it, and `>=` accuses every batch in a spotless training set"
    )
    assert all(cell["status"] == "ok" for cell in spotless["batches"].values()), (
        "every batch in a set with nothing to find is 'ok', not 'poisoned'"
    )


# --- exercise 7: the extraction budget ----------------------------------------------------

def _hand_extraction(rows):
    return {"target_agreement": 0.95, "repeats": 4, "evaluation_rows": 100,
            "points": [{"queries": q, "point": p, "lo": lo, "hi": hi, "level": 0.95,
                        "n": 100, "n_boot": 50, "method": "percentile bootstrap"}
                       for q, p, lo, hi in rows]}


def test_extraction_budget_takes_the_attacker_favourable_bound():
    curve = _hand_extraction([(10, 0.80, 0.78, 0.82), (50, 0.90, 0.88, 0.92),
                              (100, 0.940, 0.930, 0.952), (200, 0.955, 0.945, 0.965),
                              (400, 0.970, 0.952, 0.980)])
    rep = L.extraction_risk(curve)
    q = rep["queries_to_extract"]
    assert q["lo"] == 100, (
        f"100 queries is the first budget whose UPPER bound (0.952) reaches 0.95 — the "
        f"attacker got lucky, and that is the budget a control has to survive. You reported "
        f"{q['lo']}"
    )
    assert q["point"] == 200, (
        f"200 is the first budget whose POINT reaches 0.95; you reported {q['point']}"
    )
    assert q["hi"] == 400, (
        f"400 is the first budget whose LOWER bound reaches 0.95 — the one you could be "
        f"confident about. You reported {q['hi']}. Reading lo and hi the other way round "
        "inverts the threat: for somebody else's attack you plan against their luck, not yours"
    )
    assert rep["reached"] is True and rep["target_agreement"] == 0.95, (
        "say whether the grid reached the target, and at what target"
    )
    assert q["lo"] <= q["point"] <= q["hi"], "lo <= point <= hi holds by construction"
    # Landing EXACTLY on the target reaches it. No bound on the real curve ties with 0.95, so
    # only a hand-built curve pins this; `>` scored full marks in mutation testing.
    on_target = _hand_extraction([(10, 0.80, 0.78, 0.82), (50, 0.90, 0.88, 0.95),
                                  (100, 0.95, 0.94, 0.96), (200, 0.97, 0.95, 0.98)])
    exact = L.extraction_risk(on_target)["queries_to_extract"]
    assert (exact["lo"], exact["point"], exact["hi"]) == (50, 100, 200), (
        f"a bound that equals the target has reached it: 50 is the first budget whose hi is "
        f"0.95, 100 the first whose point is, 200 the first whose lo is. You reported "
        f"lo={exact['lo']} point={exact['point']} hi={exact['hi']} — testing with `>` instead "
        "of `>=` silently buys the attacker one more grid step than the measurement gives them"
    )


def test_extraction_days_limit_and_client_traffic():
    curve = _hand_extraction([(10, 0.80, 0.78, 0.82), (100, 0.940, 0.930, 0.952),
                              (200, 0.955, 0.945, 0.965), (400, 0.970, 0.952, 0.980)])
    rep = L.extraction_risk(curve, current_daily_limit=50.0, target_days=30)
    days = rep["days_to_extract"]
    assert abs(days["lo"] - 100 / 50.0) < 1e-12 and abs(days["hi"] - 400 / 50.0) < 1e-12, (
        f"days are the query budget over the enforced limit, keeping the ordering: the "
        f"soonest extraction uses the SMALLEST budget. Expected lo={100 / 50.0} "
        f"hi={400 / 50.0}, got lo={days['lo']} hi={days['hi']}"
    )
    assert abs(days["point"] - 200 / 50.0) < 1e-12, "the point budget over the same limit"
    assert rep["recommended_daily_limit"] == math.floor(100 / 30), (
        f"the recommendation is floor(lo / target_days) = {math.floor(100 / 30)}; you said "
        f"{rep['recommended_daily_limit']}. Sizing it off the confident budget sets a limit "
        "the lucky attacker walks straight through"
    )
    assert rep["verdict"] == "fail", (
        f"the soonest extraction takes {100 / 50.0} days against a {30}-day target, so this "
        "fails"
    )
    assert (rep["current_daily_limit"], rep["target_days"]) == (50.0, 30), (
        "carry the limit and the target that were actually used, not the module defaults"
    )
    # The verdict boundary: an extraction that takes EXACTLY the target time has not landed
    # inside it. 100 queries at 1 a day is 100 days against a 100-day target.
    on_time = L.extraction_risk(curve, current_daily_limit=1.0, target_days=100)
    assert abs(on_time["days_to_extract"]["lo"] - 100.0) < 1e-12, "fixture sanity: 100 / 1 = 100"
    assert on_time["verdict"] == "pass", (
        "the soonest extraction takes exactly the 100-day target, which is not INSIDE it; you "
        "reported a fail. `<=` here fails a system that met its own target exactly"
    )
    # floor, not round: the two agree on 100/30 above, so that case cannot tell them apart.
    rounding = L.extraction_risk(curve, current_daily_limit=50.0, target_days=60)
    assert rounding["recommended_daily_limit"] == math.floor(100 / 60), (
        f"floor(100 / 60) = {math.floor(100 / 60)}; you said "
        f"{rounding['recommended_daily_limit']} (round(100 / 60) = {round(100 / 60)}). "
        "Rounding a recommended limit UP hands the lucky attacker back the queries the floor "
        "just took away"
    )
    rates = {"a": 10.0, "b": 500.0, "c": 51.0, "d": 50.0, "e": 300.0}
    listed = L.extraction_risk(curve, current_daily_limit=50.0, target_days=30,
                               client_rates=rates)["clients_over_current_limit"]
    assert listed == ["b", "e", "c"], (
        f"clients strictly above the enforced limit, busiest first: ['b', 'e', 'c']. You "
        f"returned {listed} — note that 'd' sits exactly on the limit and is not over it"
    )
    default_clients = L.extraction_risk(curve)["clients_over_current_limit"]
    assert default_clients == ["partner-epsilon", "broker-beta"], (
        f"with no client_rates given, use CLIENT_QUERY_RATES; you returned {default_clients}"
    )


def test_extraction_reports_a_target_the_grid_never_reaches():
    curve = _hand_extraction([(10, 0.80, 0.78, 0.82), (100, 0.90, 0.88, 0.92)])
    rep = L.extraction_risk(curve, target=0.99)
    assert rep["reached"] is False, (
        "0.99 agreement is never reached on this grid and the report must say so"
    )
    assert rep["queries_to_extract"] is None and rep["days_to_extract"] is None, (
        "no budget may be invented for a target the measurement never reached; the honest "
        "answer is None, not the largest grid point"
    )
    assert rep["recommended_daily_limit"] is None and rep["verdict"] == "pass", (
        "with no measured budget there is no recommendation to make and nothing to fail on — "
        "the finding is that the grid was too short, which `reached` already says"
    )
    partial = L.extraction_risk(
        _hand_extraction([(10, 0.80, 0.78, 0.82), (100, 0.94, 0.93, 0.96)]), target=0.95)
    assert partial["reached"] is True and partial["queries_to_extract"]["lo"] == 100, (
        "reaching the target on the upper bound alone still counts as reached: that is the "
        "attacker-favourable reading"
    )
    assert partial["queries_to_extract"]["point"] is None, (
        "the point never reaches 0.95 on this grid, so the point budget is None while the "
        "lucky one is 100 — do not fall back to the lucky figure for all three"
    )


# --- exercise 8: the residual-risk statement ----------------------------------------------

def test_residual_resolves_paths_and_keeps_the_unsupported():
    ci = {"point": 0.89, "lo": 0.87, "hi": 0.91, "level": 0.95, "n": 10, "n_boot": 10,
          "method": "percentile bootstrap"}
    deep = {"point": 0.2, "lo": 0.1, "hi": 0.3}
    report = {"accuracy": {"headline": ci}, "a": {"b": {"c": deep}}}
    catalogue = (
        {"id": "one", "evidence": "accuracy.headline", "risk": "r1", "blind_spot": ""},
        {"id": "two", "evidence": "a.b.c", "risk": "r2", "blind_spot": ""},
        {"id": "three", "evidence": "a.b.missing", "risk": "r3", "blind_spot": ""},
        {"id": "four", "evidence": "nowhere.at.all", "risk": "r4", "blind_spot": ""},
        {"id": "five", "evidence": None, "risk": "r5", "blind_spot": "nothing bounds this"},
    )
    out = L.residual_risk(report, catalogue)
    assert [s["id"] for s in out["statements"]] == ["one", "two", "three", "four", "five"], (
        "one statement per catalogued risk, in catalogue order. Dropping the ones that do not "
        "resolve is what a hand-written residual-risk section does, and it is the failure "
        "this generator exists to prevent"
    )
    assert out["statements"][1]["bound"] is deep, (
        "a dotted path is walked all the way down, and the resolved interval itself is the "
        "bound"
    )
    assert out["n_supported"] == 2 and out["unsupported"] == ["three", "four", "five"], (
        f"two of the five resolve to an interval; you reported n_supported="
        f"{out['n_supported']} unsupported={out['unsupported']}"
    )
    assert out["statements"][0]["supported"] is True and out["statements"][0]["because"] == "", (
        "a supported statement has nothing to explain away"
    )
    assert "a.b.missing" in out["statements"][2]["because"], (
        "when a path is not in the report, say which path — a reader has to be able to find "
        "the evidence that is missing"
    )
    assert out["statements"][4]["because"] == "nothing bounds this", (
        "an entry with no evidence at all carries its catalogue blind_spot as the reason"
    )
    assert out["statements"][2]["bound"] is None and out["statements"][4]["bound"] is None, (
        "an unsupported statement carries no bound"
    )
    assert out["statements"][0]["risk"] == "r1", "the risk text travels with the statement"


def test_residual_refuses_evidence_that_carries_no_interval():
    catalogue = ({"id": "x", "evidence": "a.b", "risk": "r", "blind_spot": "bs"},)
    for label, value in (("a bare float", 0.89), ("an int", 3),
                         ("a dict with only a point", {"point": 0.89}),
                         ("a dict missing hi", {"point": 0.89, "lo": 0.87}),
                         ("a string", "0.89"), ("a list", [0.87, 0.91])):
        try:
            L.residual_risk({"a": {"b": value}}, catalogue)
        except L.UndeclarableMetricError:
            continue
        except NotImplementedError:
            raise                       # an unfilled stub is a TODO, not a wrong answer
        except Exception as exc:
            raise AssertionError(
                f"{label} as evidence raised {type(exc).__name__}; raise "
                "UndeclarableMetricError so the caller can tell a refusal from a crash"
            )
        raise AssertionError(
            f"{label} was accepted as evidence bounding a residual risk. A number with no "
            "interval cannot bound anything, and this is the third place the lesson refuses "
            "one"
        )


# --- exercise 9: the audit ----------------------------------------------------------------

def test_audit_finds_naked_numbers_and_spares_the_exact_ones():
    assert L.unbounded_numbers({"accuracy": 0.89}) == [".accuracy"], (
        f"a bare 0.89 under 'accuracy' is what the audit exists to find; you returned "
        f"{L.unbounded_numbers({'accuracy': 0.89})}"
    )
    counted = {"n": 1200, "n_boot": 2000, "seq": 3, "level": 0.95, "rows": 10,
               "flagged_rows": 7, "min_support": 30, "floor": 0.86}
    assert L.unbounded_numbers(counted) == [], (
        f"counts, declared policies and grid readings are in EXACT_QUANTITIES and are not "
        f"reported; you returned {L.unbounded_numbers(counted)}"
    )
    mixed = {"cell": {"n": 4, "drift": 0.1, "floor": 0.8, "excess": 0.02}}
    assert L.unbounded_numbers(mixed) == [".cell.drift", ".cell.excess"], (
        f"only the estimated leaves are unbounded; you returned {L.unbounded_numbers(mixed)}. "
        "Test the KEY against EXACT_QUANTITIES, not the value"
    )
    assert L.unbounded_numbers({"z": 0.1, "m": 0.2, "a": 0.3}) == [".a", ".m", ".z"], (
        "return the paths sorted, so two runs of the audit are comparable"
    )


def test_audit_walks_lists_and_nesting():
    nested = {"a": {"b": {"c": 0.5, "n": 2}}, "d": [[{"e": 0.1}], {"f": 0.2}]}
    assert L.unbounded_numbers(nested) == [".a.b.c", ".d[0][0].e", ".d[1].f"], (
        f"dicts inside lists inside lists all have to be walked; you returned "
        f"{L.unbounded_numbers(nested)}"
    )
    keyed = {"n": [1, 2, 3], "widths": [0.1, 0.2]}
    assert L.unbounded_numbers(keyed) == [".widths[0]", ".widths[1]"], (
        f"a list keeps its parent key, so a list under 'n' is still counted and a list under "
        f"'widths' is not; you returned {L.unbounded_numbers(keyed)}"
    )
    assert L.unbounded_numbers({"t": (0.1, {"u": 0.2})}) == [".t[0]", ".t[1].u"], (
        "tuples are walked like lists"
    )


def test_audit_stops_at_intervals_and_ignores_non_numbers():
    interval = {"point": 0.89, "lo": 0.87, "hi": 0.91, "width": 0.04, "declared": 0.87,
                "strange": 1.2345}
    assert L.unbounded_numbers({"accuracy": interval}) == [], (
        f"everything inside an interval is bounded by construction, so do not descend into "
        f"one; you returned {L.unbounded_numbers({'accuracy': interval})}"
    )
    assert L.unbounded_numbers(interval) == [], "the interval itself is bounded too"
    assert L.unbounded_numbers({"ok": True, "off": False, "name": "x", "gone": None}) == [], (
        "True is not a number here. isinstance(True, int) is True in Python, so a bare int "
        "check reports every boolean in the report as an unbounded number"
    )
    near = {"point": 0.89, "lo": 0.87}
    assert L.unbounded_numbers({"half": near}) == [".half.lo", ".half.point"], (
        f"a dict with a point and a lo but no hi is not an interval — it is two naked numbers "
        f"and must be reported as such; you returned {L.unbounded_numbers({'half': near})}"
    )


# --- the artefact -------------------------------------------------------------------------

def test_the_report_carries_uncertainty_everywhere():
    report = L.article_15_report()
    naked = L.unbounded_numbers(report)
    assert naked == [], (
        f"the assembled Article 15 report still has {len(naked)} numeric leaf/leaves with no "
        f"interval and no licence to be bare: {naked[:6]}. Every one of them is a number an "
        "inspector could ask about and nobody could answer"
    )
    assert report["robustness"]["floor"]["floor"] == L.ACCURACY_FLOOR, (
        "the sealed floor travels into the artefact"
    )
    assert report["residual_risk"]["unsupported"] == [
        "degradation_under_untested_perturbation", "poisoning_below_the_detection_floor",
        "extraction_by_colluding_clients"], (
        f"three of the seven catalogued risks can never resolve to an interval and must be "
        f"reported as unbounded; you have {report['residual_risk']['unsupported']}"
    )
    assert report["verdict"] == "fail", (
        "stale_feature crosses at the severity it expects to operate at and extraction lands "
        "inside the 90-day target, so the overall verdict is a fail"
    )


RUBRIC = [
    (test_bootstrap_pairs_labels_with_predictions, 5,
     "index y_true and y_pred with the SAME (n_boot, n) matrix"),
    (test_bootstrap_draw_is_the_documented_one, 4,
     "one rng.integers call with size=(n_boot, n), from default_rng(seed)"),
    (test_bootstrap_point_is_the_full_sample_statistic, 3,
     "point is the metric on the full sample, not the mean of the replicates"),
    (test_bootstrap_level_and_reported_metadata, 3,
     "place the percentiles from `level`; echo n, n_boot, level and method"),
    (test_bootstrap_edge_cases, 2,
     "no rows or no resamples is nan with [0.0, 1.0]; y_pred=None means y_true"),
    (test_declare_picks_the_defensible_end, 4,
     "lo when higher is better, hi when lower is better"),
    (test_declare_carries_the_provenance_and_leaves_the_input_alone, 2,
     "carry name, level, n, n_boot and method; do not write into the interval"),
    (test_declare_refuses_anything_that_is_not_an_interval, 5,
     "raise UndeclarableMetricError on a missing key, a nan, an inversion or a point outside"),
    (test_seal_floor_records_payload_and_digest, 4,
     "append one floor entry; digest_of({metric, floor, rationale, as_of})"),
    (test_seal_floor_refuses_a_floor_chosen_after_the_fact, 5,
     "refuse when that metric has a measurement, or already has a floor"),
    (test_seal_floor_validates_its_arguments, 4,
     "ValueError for a short rationale or a floor outside [0, 1]; both bounds inclusive"),
    (test_curve_measures_every_declared_severity_in_order, 4,
     "one point per declared severity, with the family's unit and operating point"),
    (test_curve_perturbs_the_inputs_and_leaves_the_fixture_alone, 4,
     "apply the family's function, re-predict, and never write into the features you were given"),
    (test_curve_records_every_point_through_the_ledger, 4,
     "go through measure() so every point carries an ascending ledger seq"),
    (test_crossing_uses_the_lower_bound_not_the_point, 5,
     "the crossing is the first severity whose LOWER bound falls below the floor"),
    (test_crossing_refuses_a_floor_that_is_not_prior, 5,
     "compare the floor's seq against the MINIMUM seq in the curve"),
    (test_crossing_refuses_a_tampered_floor_or_the_wrong_metric, 4,
     "recompute the digest from the four payload fields before reporting anything"),
    (test_crossing_headroom_and_verdict_against_the_operating_point, 4,
     "headroom is grid steps from the operating point; at or below it is a fail"),
    (test_crossing_on_the_real_sweep, 4,
     "the bound crosses a grid step before the point on this fixture"),
    (test_poison_rates_match_an_independent_derivation, 4,
     "flag under NEIGHBOUR_AGREEMENT_FLOOR; one row per batch; report k and min_support"),
    (test_poison_support_floor_is_the_parameter_not_a_special_case, 5,
     "min_support is an argument, applied before any comparison, and inclusive"),
    (test_poison_compares_intervals_against_a_leave_one_out_baseline, 5,
     "baseline excludes the batch; poisoned means lo above the baseline's hi"),
    (test_poison_passes_a_clean_set_and_orders_its_findings, 4,
     "a clean set must pass; findings run worst rate first"),
    (test_extraction_budget_takes_the_attacker_favourable_bound, 5,
     "lo is the first budget whose hi reaches the target; hi is the first whose lo does"),
    (test_extraction_days_limit_and_client_traffic, 5,
     "days = budget / limit keeping the ordering; recommend floor(lo / target_days)"),
    (test_extraction_reports_a_target_the_grid_never_reaches, 3,
     "None rather than an invented budget, and reached=False"),
    (test_residual_resolves_paths_and_keeps_the_unsupported, 5,
     "one statement per catalogued risk, in order, unsupported ones kept and explained"),
    (test_residual_refuses_evidence_that_carries_no_interval, 4,
     "evidence without point, lo and hi raises UndeclarableMetricError"),
    (test_audit_finds_naked_numbers_and_spares_the_exact_ones, 5,
     "test the KEY against EXACT_QUANTITIES and return the paths sorted"),
    (test_audit_walks_lists_and_nesting, 4,
     "walk dicts by key and lists by index, keeping the parent key for list items"),
    (test_audit_stops_at_intervals_and_ignores_non_numbers, 3,
     "do not descend into a dict carrying point, lo and hi; booleans are not numbers"),
    (test_the_report_carries_uncertainty_everywhere, 5,
     "the assembled report must audit clean, and three residual risks stay unbounded"),
]
