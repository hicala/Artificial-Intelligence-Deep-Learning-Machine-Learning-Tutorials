"""Autograder rubric for P01-L05. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Every expected value here is re-derived from `lesson.EVENT_STREAM` by code in THIS file, in
  a different shape from the notebook's own (a plain `enumerate` over the raw stream and plain
  dicts, rather than a sorted walk of the chained log). Nothing is a literal copied out of the
  notebook's output, and nothing calls the student's function to decide what the student's
  function should have returned.
* Exercises 2 to 6 are graded against `VIEW`, this file's OWN reference fold, and not against
  the student's `decision_view`. A broken exercise 1 must not cost a student the other five;
  that is what partial credit means here.
* `VIEW` is never mutated. Any test that needs a different view builds one with `_record(...)`,
  which returns a complete record with every key present. A mutated fixture would leak into
  the tests that run after it and make a failure depend on rubric order.
* Several tests deliberately run a function on a HAND-BUILT view rather than on the lesson's,
  because the lesson's log has exactly one instance of each defect and an implementation can
  pass it by hard-coding. The four-eyes reason ladder, the bias status ladder, the signed
  sensitivity, the findings order and every "this is a parameter, not a constant" test are all
  of that kind.
* `test_bias_separates_collapse_from_decline` is what stops an implementation that special-
  cases `u:marlies`, or that compares |sensitivity| instead of the signed one, from scoring
  full marks. A reviewer who overrides MORE as confidence rises has a different problem, and
  an absolute comparison files it as automation bias.
* `test_gate_clock_anomaly_precedes_reviewed_too_fast` exists because -40 < 10 is true. The
  naive comparison accuses a named person of rushing a case they never saw, and buries the
  clock bug that produced the number.
"""
import importlib.util
import math
import os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

KEYS = ("decision_id", "first_seq", "last_seq", "model_version", "confidence",
        "recommendation", "identification_basis", "submitted_by", "verifier_ids",
        "outcome", "use_start", "use_end", "duration_seconds")
SCALARS = ("model_version", "submitted_by", "identification_basis", "outcome",
           "use_start", "use_end")


def _record(decision_id="D-X", **kw) -> dict:
    """A complete view record, so a hand-built case can never be missing a key."""
    base = {k: None for k in KEYS}
    base.update({"decision_id": decision_id, "first_seq": 0, "last_seq": 1,
                 "verifier_ids": [], "identification_basis": "credit_file",
                 "submitted_by": "u:intake"})
    base.update(kw)
    return base


def _view(records: list) -> dict:
    return {r["decision_id"]: r for r in records}


def _reference_view() -> dict:
    """This file's own fold, straight off EVENT_STREAM with enumerate as the sequence."""
    view: dict = {}
    for seq, event in enumerate(lesson.EVENT_STREAM):
        did = event.get("decision_id")
        if did is None:
            continue
        if did not in view:
            view[did] = _record(did, first_seq=seq, identification_basis=None,
                                submitted_by=None, verifier_ids=[])
        rec = view[did]
        rec["last_seq"] = seq
        payload = event["payload"]
        for field in SCALARS:
            if payload.get(field) not in (None, "", []):
                rec[field] = payload[field]
        if payload.get("match_score") is not None:
            rec["confidence"] = float(payload["match_score"])
        if payload.get("recommendation") not in (None, ""):
            rec["recommendation"] = payload["recommendation"]
        rec["verifier_ids"] = rec["verifier_ids"] + [v for v in payload.get("verifier_ids", [])
                                                     if v]
    for rec in view.values():
        if rec["use_start"] and rec["use_end"]:
            rec["duration_seconds"] = (lesson.parse_ts(rec["use_end"])
                                       - lesson.parse_ts(rec["use_start"])).total_seconds()
    return view


VIEW = _reference_view()
IDS = sorted(VIEW)


def _pct(values: list, q: float) -> float:
    """numpy's linear percentile, written out here so the rubric does not borrow numpy's."""
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[int(pos)])
    return float(ordered[low] * (high - pos) + ordered[high] * (pos - low))


def _distinct(ids: list) -> list:
    return list(dict.fromkeys(ids))


# ----------------------------------------------------------------- exercise 1
def test_view_shape_and_coverage():
    got = lesson.decision_view(lesson.LOG)
    assert isinstance(got, dict), f"decision_view returned {type(got).__name__}, want a dict"
    assert set(got) == set(VIEW), (
        f"the view holds {len(got)} decisions, expected {len(VIEW)}. A view with one extra "
        "key usually means the system-level model_deployed event — which has no decision_id "
        "at all — was folded in as a decision"
    )
    sample = got[IDS[0]]
    assert set(sample) == set(KEYS), (
        f"a record's keys came back {sorted(sample)}, expected {sorted(KEYS)}"
    )
    for did in IDS:
        assert got[did]["first_seq"] == VIEW[did]["first_seq"], (
            f"{did} first_seq is {got[did]['first_seq']}, expected {VIEW[did]['first_seq']}"
        )
        assert got[did]["last_seq"] == VIEW[did]["last_seq"], (
            f"{did} last_seq is {got[did]['last_seq']}, expected {VIEW[did]['last_seq']} — "
            "take the seq of the LAST entry that mentions the decision, walking in seq order"
        )


def test_view_folds_scalar_fields_and_never_overwrites_with_nothing():
    got = lesson.decision_view(lesson.LOG)
    for did in IDS:
        for field in SCALARS + ("confidence", "recommendation"):
            assert got[did][field] == VIEW[did][field], (
                f"{did} {field} came back {got[did][field]!r}, expected "
                f"{VIEW[did][field]!r}. identification_basis and submitted_by sit in the "
                "inference_requested payload; confidence is match_score from match_found"
            )
    # A later entry carrying an empty value must not erase a real one. All THREE empties are
    # checked: None, "" and []. The lesson's own log contains none of them, so an
    # implementation that guards only `is not None` passes the whole fixture and still throws
    # away a real submitted_by the moment a half-built upstream writes a blank string.
    log = []
    lesson.append_event(log, {"ts": "2026-01-01T00:00:00Z", "event_type": "decision_recorded",
                              "actor": "svc", "system_id": "s", "decision_id": "D-1",
                              "payload": {"outcome": "approve", "use_end":
                                          "2026-01-01T00:00:00Z",
                                          "submitted_by": "u:intake",
                                          "identification_basis": "remote_biometric"}})
    lesson.append_event(log, {"ts": "2026-01-01T00:01:00Z", "event_type": "note",
                              "actor": "svc", "system_id": "s", "decision_id": "D-1",
                              "payload": {"outcome": None, "submitted_by": "",
                                          "identification_basis": []}})
    rec = lesson.decision_view(log)["D-1"]
    assert rec["outcome"] == "approve", (
        f"outcome came back {rec['outcome']!r} — a payload value of None, '' or [] is what a "
        "half-built pipeline writes when it has nothing to say, and it must not overwrite a "
        "real value recorded earlier"
    )
    assert rec["submitted_by"] == "u:intake", (
        f"submitted_by came back {rec['submitted_by']!r} — a later payload carried the EMPTY "
        "STRING and it overwrote a real value. Guard on None, '' and [] together; guarding "
        "only `is not None` passes every decision in this lesson's log and still loses the "
        "submitter, which is the field the self-verification check reads"
    )
    assert rec["identification_basis"] == "remote_biometric", (
        f"identification_basis came back {rec['identification_basis']!r} — a later payload "
        "carried an EMPTY LIST and it overwrote a real value. Losing the basis silently moves "
        "a decision out of Article 14(5) scope, which is the expensive direction to be wrong in"
    )


def test_view_keeps_duplicate_verifiers_in_sequence_order():
    got = lesson.decision_view(lesson.LOG)
    repeated = [d for d in IDS
                if len(VIEW[d]["verifier_ids"]) != len(set(VIEW[d]["verifier_ids"]))]
    assert len(repeated) == 12, "the reference fold itself is wrong; this is a rubric bug"
    for did in IDS:
        assert got[did]["verifier_ids"] == VIEW[did]["verifier_ids"], (
            f"{did} verifier_ids came back {got[did]['verifier_ids']}, expected "
            f"{VIEW[did]['verifier_ids']}. Keep every entry, in seq order: deduplicating here "
            "destroys the evidence that one person signed twice, and sorting by ts reverses "
            "the pairs whose clock stepped backwards"
        )
    # A hand-built pair in reverse timestamp order: the fold must follow seq, not ts. The
    # third entry carries a blank id alongside a real one — no decision in this lesson's log
    # does, so nothing else pins down the "empty strings dropped" half of the contract.
    log = []
    for stamp, vid in (("2026-01-01T00:05:00Z", "u:a"), ("2026-01-01T00:01:00Z", "u:b")):
        lesson.append_event(log, {"ts": stamp, "event_type": "human_verification",
                                  "actor": "ui", "system_id": "s", "decision_id": "D-1",
                                  "payload": {"verifier_ids": [vid]}})
    lesson.append_event(log, {"ts": "2026-01-01T00:09:00Z", "event_type": "human_verification",
                              "actor": "ui", "system_id": "s", "decision_id": "D-1",
                              "payload": {"verifier_ids": ["", "u:a"]}})
    blanks = lesson.decision_view(log)["D-1"]["verifier_ids"]
    assert blanks == ["u:a", "u:b", "u:a"], (
        f"verifier_ids came back {blanks}, expected ['u:a', 'u:b', 'u:a'] — keep every real "
        "id including the repeat, in seq order, and DROP the empty string. A blank id is not "
        "a natural person, and counting it as one is how a single-signer decision passes a "
        "two-person rule"
    )
    log = log[:2]
    assert lesson.decision_view(log)["D-1"]["verifier_ids"] == ["u:a", "u:b"], (
        "two verification events whose timestamps run backwards came back in timestamp "
        "order. Order by seq; the clock is the thing under suspicion here"
    )


def test_view_durations_are_signed_and_may_be_missing():
    got = lesson.decision_view(lesson.LOG)
    for did in IDS:
        expected = VIEW[did]["duration_seconds"]
        actual = got[did]["duration_seconds"]
        if expected is None:
            assert actual is None, f"{did} duration should be None, came back {actual!r}"
        else:
            # 1e-6 rather than 1e-9: subtracting two POSIX timestamps near 1.8e9 loses about
            # 5e-8 of a second, so a 1e-9 window rejects that perfectly honest route and
            # accepts only datetime subtraction. Every way of getting the SIGN wrong is out by
            # at least twice the duration, so nothing that matters slips through this window.
            assert actual is not None and abs(actual - expected) < 1e-6, (
                f"{did} duration came back {actual!r}, expected {expected!r}"
            )
    negative = [d for d in IDS if (got[d]["duration_seconds"] or 0) < 0]
    assert len(negative) == 3, (
        f"{len(negative)} decisions came back with a negative duration, expected 3. abs() on "
        "a backwards clock invents a review that never happened and hides the bug for good"
    )


# ----------------------------------------------------------------- exercise 2
def _expected_overrides(view: dict):
    overall = [0, 0]
    reviewers: dict = {}
    outcomes: dict = {}
    undecided = unscored = unverified = 0
    for rec in view.values():
        if rec["outcome"] is None:
            undecided += 1
            continue
        if rec["recommendation"] is None:
            unscored += 1
            continue
        if not rec["verifier_ids"]:
            unverified += 1
            continue
        over = 1 if rec["outcome"] != rec["recommendation"] else 0
        overall[0] += 1
        overall[1] += over
        cell = outcomes.setdefault(rec["recommendation"], [0, 0])
        cell[0] += 1
        cell[1] += over
        for vid in _distinct(rec["verifier_ids"]):
            r = reviewers.setdefault(vid, [0, 0])
            r[0] += 1
            r[1] += over
    return overall, reviewers, outcomes, undecided, unscored, unverified


def test_override_exclusions_and_overall_rate():
    rep = lesson.override_rates(VIEW)
    overall, _, _, undecided, unscored, unverified = _expected_overrides(VIEW)
    assert set(rep) == {"overall", "by_reviewer", "by_outcome", "unverified_decided",
                        "undecided", "unscored"}, f"keys came back {sorted(rep)}"
    assert (rep["undecided"], rep["unscored"], rep["unverified_decided"]) == \
        (undecided, unscored, unverified), (
        f"the exclusion counts came back "
        f"{(rep['undecided'], rep['unscored'], rep['unverified_decided'])}, expected "
        f"{(undecided, unscored, unverified)}"
    )
    assert rep["overall"]["n"] == overall[0] and rep["overall"]["n_overrides"] == overall[1], (
        f"overall came back n={rep['overall']['n']}, overrides={rep['overall']['n_overrides']}, "
        f"expected n={overall[0]}, overrides={overall[1]}"
    )
    assert abs(rep["overall"]["rate"] - overall[1] / overall[0]) < 1e-12
    # The three exclusions are ordered. A decision with none of outcome, recommendation or a
    # verifier is counted ONCE, as undecided.
    tiny = lesson.override_rates(_view([_record("D-1")]))
    assert (tiny["undecided"], tiny["unscored"], tiny["unverified_decided"]) == (1, 0, 0), (
        f"a record with no outcome, no recommendation and no verifier came back "
        f"{(tiny['undecided'], tiny['unscored'], tiny['unverified_decided'])}; it is one "
        "decision and it is counted once, as undecided"
    )
    assert tiny["overall"]["rate"] is None, "an empty denominator is None, not 0.0"


def test_override_by_reviewer_counts_distinct_persons():
    rep = lesson.override_rates(VIEW)
    _, reviewers, _, _, _, _ = _expected_overrides(VIEW)
    assert set(rep["by_reviewer"]) == set(reviewers), (
        f"by_reviewer holds {sorted(rep['by_reviewer'])}, expected {sorted(reviewers)}"
    )
    for vid, (n, overrides) in reviewers.items():
        cell = rep["by_reviewer"][vid]
        assert (cell["n"], cell["n_overrides"]) == (n, overrides), (
            f"{vid} came back n={cell['n']}, overrides={cell['n_overrides']}, expected "
            f"n={n}, overrides={overrides}"
        )
        assert abs(cell["rate"] - overrides / n) < 1e-12
    doubled = lesson.override_rates(_view([
        _record("D-1", outcome="decline", recommendation="approve",
                verifier_ids=["u:a", "u:a", "u:a"])]))
    assert doubled["by_reviewer"]["u:a"]["n"] == 1, (
        f"one decision signed three times by one person came back as "
        f"{doubled['by_reviewer']['u:a']['n']} decisions for that person. Count distinct "
        "verifiers: a repeat is the same person, and counting it again inflates their volume"
    )
    assert doubled["overall"]["n"] == 1


def test_override_by_outcome_is_keyed_on_the_model_recommendation():
    rep = lesson.override_rates(VIEW)
    _, _, outcomes, _, _, _ = _expected_overrides(VIEW)
    assert set(rep["by_outcome"]) == set(outcomes), (
        f"by_outcome holds {sorted(rep['by_outcome'])}, expected {sorted(outcomes)}"
    )
    for rec, (n, overrides) in outcomes.items():
        cell = rep["by_outcome"][rec]
        assert (cell["n"], cell["n_overrides"]) == (n, overrides), (
            f"by_outcome[{rec!r}] came back n={cell['n']}, overrides={cell['n_overrides']}, "
            f"expected n={n}, overrides={overrides}"
        )
    # Keyed on the recorded outcome instead, this table says something else entirely.
    flipped = lesson.override_rates(_view([
        _record("D-1", outcome="decline", recommendation="approve", verifier_ids=["u:a"]),
        _record("D-2", outcome="approve", recommendation="approve", verifier_ids=["u:a"])]))
    assert flipped["by_outcome"]["approve"]["n"] == 2, (
        f"by_outcome['approve'] came back n={flipped['by_outcome']['approve']['n']}, expected "
        "2 — both decisions were RECOMMENDED approve. Keying on the recorded outcome gives "
        "the rate at which people override INTO approve, which is a different question"
    )
    assert flipped["by_outcome"]["approve"]["n_overrides"] == 1
    assert "decline" not in flipped["by_outcome"]


def test_override_support_floor_is_the_parameter():
    strict = lesson.override_rates(VIEW, min_support=10 ** 6)
    loose = lesson.override_rates(VIEW, min_support=1)
    assert set(strict["by_reviewer"]) == set(loose["by_reviewer"]), (
        "the set of reviewers changed with min_support — an unsupported reviewer is flagged, "
        "never dropped. A reviewer nobody can evaluate is itself the finding"
    )
    assert all(not c["supported"] for c in strict["by_reviewer"].values()), (
        "with min_support at a million, every reviewer must come back unsupported; "
        "min_support is an argument, not the constant MIN_SUPPORT"
    )
    assert all(c["supported"] for c in loose["by_reviewer"].values())
    exact = lesson.override_rates(_view([
        _record(f"D-{k}", outcome="approve", recommendation="approve", verifier_ids=["u:a"])
        for k in range(30)]), min_support=30)
    assert exact["by_reviewer"]["u:a"]["supported"] is True, (
        "exactly min_support decisions must count as supported — the floor is inclusive"
    )


# ----------------------------------------------------------------- exercise 3
def test_time_anomalies_never_enter_the_distribution():
    # Three honest decisions and one backwards clock big enough to be unmissable.
    rows = [_record(f"D-{k}", use_start="2026-01-01T00:00:00Z",
                    use_end="2026-01-01T00:01:40Z", duration_seconds=100.0,
                    verifier_ids=["u:a"]) for k in range(3)]
    rows.append(_record("D-9", use_start="2026-01-02T00:00:00Z",
                        use_end="2026-01-01T00:00:00Z", duration_seconds=-86400.0,
                        verifier_ids=["u:a"]))
    rep = lesson.time_to_decision(_view(rows))
    assert rep["n"] == 3 and rep["anomalous"] == ["D-9"], (
        f"n={rep['n']}, anomalous={rep['anomalous']}; the backwards-clock decision must leave "
        "the distribution entirely, not be counted and not be made positive"
    )
    assert rep["median"] == 100.0 and rep["p10"] == 100.0, (
        f"median came back {rep['median']} over three 100-second decisions and one clock "
        "anomaly. Including -86400 drags it; abs() turns it into a 24-hour review"
    )
    assert rep["by_reviewer"]["u:a"]["n"] == 3, (
        "the anomaly reached a reviewer's own distribution even though it left the overall one"
    )
    got = lesson.time_to_decision(VIEW)
    real = [VIEW[d]["duration_seconds"] for d in IDS
            if VIEW[d]["duration_seconds"] is not None and VIEW[d]["duration_seconds"] >= 0]
    assert got["n"] == len(real), f"n came back {got['n']}, expected {len(real)}"
    for name, q in (("median", 50), ("p10", 10), ("p90", 90)):
        assert abs(got[name] - _pct(real, q)) < 1e-9, (
            f"{name} came back {got[name]}, expected {_pct(real, q)} — numpy's default linear "
            "percentile over the non-negative durations only"
        )


def test_time_lists_are_exact_and_sorted():
    got = lesson.time_to_decision(VIEW)
    anomalous = sorted(d for d in IDS if (VIEW[d]["duration_seconds"] or 0) < 0)
    fast = sorted(d for d in IDS
                  if VIEW[d]["duration_seconds"] is not None
                  and 0 <= VIEW[d]["duration_seconds"] < lesson.MIN_PLAUSIBLE_SECONDS)
    incomplete = sum(1 for d in IDS if VIEW[d]["duration_seconds"] is None)
    assert got["anomalous"] == anomalous, (
        f"anomalous came back {got['anomalous']}, expected {anomalous}"
    )
    assert got["fast"] == fast, (
        f"fast came back {len(got['fast'])} ids, expected {len(fast)} and sorted. A decision "
        "with a negative duration is not fast; it has already left"
    )
    assert got["incomplete"] == incomplete, (
        f"incomplete came back {got['incomplete']}, expected {incomplete} — the decisions "
        "that never closed, so they have no duration at all"
    )


def test_time_floor_is_the_parameter():
    got = lesson.time_to_decision(VIEW, floor_seconds=0)
    assert got["fast"] == [], (
        f"with floor_seconds at 0, {len(got['fast'])} decisions were still called fast — the "
        "comparison is strict and floor_seconds is an argument, not MIN_PLAUSIBLE_SECONDS"
    )
    wide = lesson.time_to_decision(VIEW, floor_seconds=10 ** 9)
    assert len(wide["fast"]) == wide["n"], (
        f"with an enormous floor, {len(wide['fast'])} of {wide['n']} real durations were "
        "called fast; every one of them should be"
    )
    # No decision in the log lands exactly on the floor, so > and >= agree there. One here does.
    edge = lesson.time_to_decision(_view([
        _record("D-1", use_start="2026-01-01T00:00:00Z", use_end="2026-01-01T00:00:10Z",
                duration_seconds=10.0, verifier_ids=["u:a"]),
        _record("D-2", use_start="2026-01-01T00:00:00Z", use_end="2026-01-01T00:00:09Z",
                duration_seconds=9.0, verifier_ids=["u:a"])]), floor_seconds=10)
    assert edge["fast"] == ["D-2"], (
        f"fast came back {edge['fast']} for one 10-second and one 9-second decision against a "
        "10-second floor. The comparison is STRICTLY below: a decision that took exactly the "
        "floor met the policy, and <= turns every boundary case into an accusation"
    )


def test_time_by_reviewer_medians_and_distinct_verifiers():
    got = lesson.time_to_decision(VIEW)
    expected: dict = {}
    for did in IDS:
        rec = VIEW[did]
        seconds = rec["duration_seconds"]
        if seconds is None or seconds < 0:
            continue
        for vid in _distinct(rec["verifier_ids"]):
            expected.setdefault(vid, []).append(seconds)
    assert set(got["by_reviewer"]) == set(expected), (
        f"by_reviewer holds {sorted(got['by_reviewer'])}, expected {sorted(expected)}"
    )
    for vid, values in expected.items():
        cell = got["by_reviewer"][vid]
        assert cell["n"] == len(values), (
            f"{vid} n came back {cell['n']}, expected {len(values)} — one entry per decision "
            "per DISTINCT verifier"
        )
        assert abs(cell["median"] - _pct(values, 50)) < 1e-9, (
            f"{vid} median came back {cell['median']}, expected {_pct(values, 50)}"
        )


# ----------------------------------------------------------------- exercise 4
def test_four_eyes_scope_is_the_basis_parameter():
    got = lesson.four_eyes_report(VIEW)
    in_scope = [d for d in IDS if VIEW[d]["identification_basis"] == "remote_biometric"]
    assert got["in_scope"] == len(in_scope), (
        f"in_scope came back {got['in_scope']}, expected {len(in_scope)}"
    )
    assert got["out_of_scope"] == len(IDS) - len(in_scope), (
        f"out_of_scope came back {got['out_of_scope']}, expected "
        f"{len(IDS) - len(in_scope)} — Article 14(5) reaches the Annex III point 1(a) "
        "decisions only, and the rest are out of scope rather than compliant"
    )
    flipped = lesson.four_eyes_report(VIEW, basis="credit_file")
    assert flipped["in_scope"] == len(IDS) - len(in_scope), (
        "passing basis='credit_file' did not move the scope — basis is a parameter, not the "
        "hard-coded string 'remote_biometric'"
    )
    # Every decision in this lesson's log records a basis, so the fixture alone cannot say
    # what happens to one that does not. A decision whose inference_requested event never
    # arrived has identification_basis None, and None is not point 1(a): it is out of scope.
    missing = lesson.four_eyes_report(_view([
        _record("D-1", identification_basis="remote_biometric", verifier_ids=["u:a", "u:b"]),
        _record("D-2", identification_basis="credit_file", verifier_ids=["u:a"]),
        _record("D-3", identification_basis=None, verifier_ids=[]),
    ]))
    assert (missing["in_scope"], missing["out_of_scope"]) == (1, 2), (
        f"a view holding one remote_biometric, one credit_file and one record with NO "
        f"identification_basis came back in_scope={missing['in_scope']}, "
        f"out_of_scope={missing['out_of_scope']}, expected 1 and 2. Compare the basis to "
        "`basis` and nothing else: sweeping a missing basis into scope reports Article 14(5) "
        "violations against decisions the article never reached"
    )
    assert missing["violations"] == [], (
        f"violations came back {missing['violations']} — the only in-scope decision there has "
        "two distinct verifiers and complies; D-3 has no basis and is not in scope at all"
    )


def test_four_eyes_counts_persons_not_entries():
    got = lesson.four_eyes_report(VIEW)
    reasons = dict(got["violations"])
    duplicates = sorted(d for d in IDS
                        if VIEW[d]["identification_basis"] == "remote_biometric"
                        and len(set(VIEW[d]["verifier_ids"])) == 1
                        and len(VIEW[d]["verifier_ids"]) >= 2)
    assert len(duplicates) == 12, "the reference fold itself is wrong; this is a rubric bug"
    for did in duplicates:
        assert reasons.get(did) == "duplicate_verifier", (
            f"{did} was reported {reasons.get(did)!r}. Its verifier list has two entries and "
            "one person in it. `len(verifier_ids) >= 2` passes all twelve of these, and "
            "Article 14(5) asks for two natural PERSONS"
        )
    assert set(got["compliant"]) & set(duplicates) == set()
    singles = sorted(d for d in IDS
                     if VIEW[d]["identification_basis"] == "remote_biometric"
                     and len(VIEW[d]["verifier_ids"]) == 1)
    empties = sorted(d for d in IDS
                     if VIEW[d]["identification_basis"] == "remote_biometric"
                     and not VIEW[d]["verifier_ids"])
    assert [d for d, r in got["violations"] if r == "single_verifier"] == singles
    assert [d for d, r in got["violations"] if r == "no_verification"] == empties


def test_four_eyes_reason_ladder():
    rows = [_record("D-1", identification_basis="remote_biometric", verifier_ids=[]),
            _record("D-2", identification_basis="remote_biometric", verifier_ids=["u:a"]),
            _record("D-3", identification_basis="remote_biometric",
                    verifier_ids=["u:a", "u:a", "u:a"]),
            _record("D-4", identification_basis="remote_biometric",
                    verifier_ids=["u:a", "u:b"]),
            _record("D-5", identification_basis="remote_biometric",
                    verifier_ids=["u:a", "u:b", "u:a"])]
    got = lesson.four_eyes_report(_view(rows))
    assert got["compliant"] == ["D-4", "D-5"], (
        f"compliant came back {got['compliant']}, expected ['D-4', 'D-5'] — D-5 has three "
        "entries and two distinct people, which satisfies the rule"
    )
    assert got["violations"] == [("D-1", "no_verification"), ("D-2", "single_verifier"),
                                 ("D-3", "duplicate_verifier")], (
        f"violations came back {got['violations']}. The three reasons are not "
        "interchangeable: nothing recorded, one person once, one person repeatedly — three "
        "different remedies, and D-3 is the one a length test lets through"
    )


def test_four_eyes_rate_verdict_and_a_clean_log():
    got = lesson.four_eyes_report(VIEW)
    in_scope = [d for d in IDS if VIEW[d]["identification_basis"] == "remote_biometric"]
    compliant = [d for d in in_scope if len(set(VIEW[d]["verifier_ids"])) >= 2]
    assert abs(got["rate"] - len(compliant) / len(in_scope)) < 1e-12, (
        f"rate came back {got['rate']}, expected {len(compliant) / len(in_scope)} — compliant "
        "over IN-SCOPE decisions, not over the whole log"
    )
    assert got["verdict"] == "fail"
    clean = lesson.four_eyes_report(_view([
        _record("D-1", identification_basis="remote_biometric", verifier_ids=["u:a", "u:b"]),
        _record("D-2", identification_basis="credit_file", verifier_ids=[])]))
    assert clean["verdict"] == "pass" and clean["violations"] == [], (
        f"a log whose only in-scope decision has two distinct verifiers came back "
        f"{clean['verdict']!r}. A verdict that is always 'fail' carries no information"
    )
    assert clean["rate"] == 1.0 and clean["out_of_scope"] == 1
    empty = lesson.four_eyes_report(_view([_record("D-1", identification_basis="credit_file")]))
    assert empty["rate"] is None, "with no in-scope decisions the rate is None, not 0.0 or 1.0"


# ----------------------------------------------------------------- exercise 5
def _expected_bias(view: dict, min_support=None, split=None, floor=None):
    min_support = lesson.MIN_SUPPORT if min_support is None else min_support
    split = lesson.CONFIDENCE_SPLIT if split is None else split
    floor = lesson.HIGH_CONFIDENCE_FLOOR if floor is None else floor
    n_bins = len(lesson.CONFIDENCE_BINS) - 1
    tally: dict = {}
    for rec in view.values():
        if rec["outcome"] is None or rec["recommendation"] is None \
                or rec["confidence"] is None:
            continue
        over = 1 if rec["outcome"] != rec["recommendation"] else 0
        index = lesson.confidence_bin(rec["confidence"], lesson.CONFIDENCE_BINS)
        for vid in _distinct(rec["verifier_ids"]):
            cell = tally.setdefault(vid, {"bins": [[0, 0] for _ in range(n_bins)],
                                          "low": [0, 0], "high": [0, 0]})
            cell["bins"][index][0] += 1
            cell["bins"][index][1] += over
            half = cell["high"] if rec["confidence"] >= split else cell["low"]
            half[0] += 1
            half[1] += over
    out: dict = {}
    for vid, cell in tally.items():
        n_low, o_low = cell["low"]
        n_high, o_high = cell["high"]
        n = n_low + n_high
        low = o_low / n_low if n_low else None
        high = o_high / n_high if n_high else None
        if min(n_low, n_high) < min_support:
            out[vid] = {"n": n, "n_low": n_low, "n_high": n_high, "low_rate": low,
                        "high_rate": high, "sensitivity": None, "min_detectable": None,
                        "status": "insufficient_support", "bins": cell["bins"]}
            continue
        sens = low - high
        mdd = lesson.min_detectable_difference((o_low + o_high) / n, n_low, n_high)
        status = ("confidence_independent" if sens < mdd
                  else "automation_bias" if high <= floor else "confidence_sensitive")
        out[vid] = {"n": n, "n_low": n_low, "n_high": n_high, "low_rate": low,
                    "high_rate": high, "sensitivity": sens, "min_detectable": mdd,
                    "status": status, "bins": cell["bins"]}
    return out


def test_bias_per_bin_counts_match_an_independent_derivation():
    got = lesson.automation_bias(VIEW)
    expected = _expected_bias(VIEW)
    assert set(got["by_reviewer"]) == set(expected), (
        f"by_reviewer holds {sorted(got['by_reviewer'])}, expected {sorted(expected)}"
    )
    for vid, want in expected.items():
        bins = got["by_reviewer"][vid]["per_bin"]
        assert len(bins) == len(want["bins"]), (
            f"{vid} has {len(bins)} bins, expected {len(want['bins'])}"
        )
        for k, (cell, (n, overrides)) in enumerate(zip(bins, want["bins"])):
            assert (cell["n"], cell["n_overrides"]) == (n, overrides), (
                f"{vid} bin {k} came back n={cell['n']}, overrides={cell['n_overrides']}, "
                f"expected n={n}, overrides={overrides}. A decision with no confidence, no "
                "recommendation or no outcome contributes to no bin at all"
            )
            assert (cell["rate"] is None) == (n == 0)
            if n:
                assert abs(cell["rate"] - overrides / n) < 1e-12


def test_bias_halves_sensitivity_and_detectable_floor():
    got = lesson.automation_bias(VIEW)
    expected = _expected_bias(VIEW)
    for vid, want in expected.items():
        cell = got["by_reviewer"][vid]
        assert (cell["n"], cell["n_low"], cell["n_high"]) == \
            (want["n"], want["n_low"], want["n_high"]), (
            f"{vid} counts came back {(cell['n'], cell['n_low'], cell['n_high'])}, expected "
            f"{(want['n'], want['n_low'], want['n_high'])} — the split is at "
            f"CONFIDENCE_SPLIT and the high half is confidence >= split"
        )
        for field in ("low_rate", "high_rate", "sensitivity", "min_detectable"):
            if want[field] is None:
                assert cell[field] is None, (
                    f"{vid} {field} came back {cell[field]!r}, expected None"
                )
            else:
                assert abs(cell[field] - want[field]) < 1e-12, (
                    f"{vid} {field} came back {cell[field]!r}, expected {want[field]!r}"
                )
    # The boundary itself. No decision in this lesson's log carries a confidence exactly equal
    # to CONFIDENCE_SPLIT, so the fixture cannot tell `>= split` from `> split` — and a real
    # scorer that emits two-decimal scores puts a pile of decisions on precisely that line.
    # The split is also a bin edge: confidence_bin(split) is a HIGH bin, so a record sitting on
    # it must be in the high half or the per-bin table and the halves contradict each other.
    split = lesson.CONFIDENCE_SPLIT
    edge = lesson.automation_bias(_view(
        [_record(f"E-{i}", confidence=split, recommendation="approve", outcome="approve",
                 verifier_ids=["u:edge"]) for i in range(lesson.MIN_SUPPORT)]
        + [_record(f"L-{i}", confidence=split - 0.10, recommendation="approve",
                   outcome="decline", verifier_ids=["u:edge"])
           for i in range(lesson.MIN_SUPPORT)]))["by_reviewer"]["u:edge"]
    assert (edge["n_low"], edge["n_high"]) == (lesson.MIN_SUPPORT, lesson.MIN_SUPPORT), (
        f"with {lesson.MIN_SUPPORT} decisions scored exactly at CONFIDENCE_SPLIT and "
        f"{lesson.MIN_SUPPORT} below it, the halves came back n_low={edge['n_low']}, "
        f"n_high={edge['n_high']}, expected {lesson.MIN_SUPPORT} and {lesson.MIN_SUPPORT}. "
        "The high half is `confidence >= split`, inclusive. Using `>` drops every decision "
        "sitting on the line into the low half, where it makes a rubber stamp look sensitive"
    )
    high_bin = lesson.confidence_bin(split, lesson.CONFIDENCE_BINS)
    assert sum(b["n"] for b in edge["per_bin"][high_bin:]) == edge["n_high"], (
        "the per-bin table and the high half disagree about the decisions sitting exactly on "
        "CONFIDENCE_SPLIT. The split is a bin edge: every record from that bin upwards is in "
        "the high half"
    )


def test_bias_support_floor_comes_first_and_is_the_parameter():
    got = lesson.automation_bias(VIEW)
    expected = _expected_bias(VIEW)
    unsupported = sorted(v for v, c in expected.items()
                         if c["status"] == "insufficient_support")
    assert got["unsupported"] == unsupported, (
        f"unsupported came back {got['unsupported']}, expected {unsupported}"
    )
    loose = lesson.automation_bias(VIEW, min_support=5)
    assert set(loose["by_reviewer"]) == set(got["by_reviewer"]), (
        "the set of reviewers moved with min_support — an unsupported reviewer is reported "
        "with a status, never filtered out of the table"
    )
    for vid in unsupported:
        assert loose["by_reviewer"][vid]["status"] != "insufficient_support", (
            f"{vid} is still insufficient_support at min_support=5 — min_support is an "
            "argument, not the constant MIN_SUPPORT"
        )
        assert loose["by_reviewer"][vid]["sensitivity"] is not None
    # The floor is on the SMALLER half, not on the total: 500 low and 29 high is not enough.
    rows = [_record(f"L-{k}", confidence=0.6, recommendation="approve",
                    outcome="approve", verifier_ids=["u:a"]) for k in range(500)]
    rows += [_record(f"H-{k}", confidence=0.95, recommendation="approve",
                     outcome="decline", verifier_ids=["u:a"]) for k in range(29)]
    lopsided = lesson.automation_bias(_view(rows), min_support=30)
    assert lopsided["by_reviewer"]["u:a"]["status"] == "insufficient_support", (
        "529 decisions, 29 of them above the split, came back with a status. The comparison "
        "is between two halves, so the floor applies to the smaller of them"
    )


def test_bias_separates_collapse_from_decline_and_is_signed():
    got = lesson.automation_bias(VIEW)
    expected = _expected_bias(VIEW)
    for vid, want in expected.items():
        assert got["by_reviewer"][vid]["status"] == want["status"], (
            f"{vid} came back {got['by_reviewer'][vid]['status']!r}, expected "
            f"{want['status']!r}"
        )
    collapsing = [v for v, c in expected.items() if c["status"] == "automation_bias"]
    declining = [v for v, c in expected.items() if c["status"] == "confidence_sensitive"]
    assert collapsing and declining, "the fixture must show both shapes; this is a rubric bug"
    # Two reviewers, same n, both declining hard. Only the level they land on separates them.
    def curve(vid, low_rate, high_rate):
        rows = []
        for k in range(200):
            rows.append(_record(f"{vid}-L{k}", confidence=0.6, recommendation="approve",
                                outcome="decline" if k < 200 * low_rate else "approve",
                                verifier_ids=[vid]))
        for k in range(200):
            rows.append(_record(f"{vid}-H{k}", confidence=0.95, recommendation="approve",
                                outcome="decline" if k < 200 * high_rate else "approve",
                                verifier_ids=[vid]))
        return rows
    pair = lesson.automation_bias(_view(curve("u:stamp", 0.40, 0.02)
                                        + curve("u:real", 0.40, 0.20)))
    assert pair["by_reviewer"]["u:stamp"]["status"] == "automation_bias", (
        "a reviewer who falls from 40% to 2% is the one the module exists to find"
    )
    assert pair["by_reviewer"]["u:real"]["status"] == "confidence_sensitive", (
        f"u:real came back {pair['by_reviewer']['u:real']['status']!r}. She declines by the "
        "same 20 points as u:stamp falls by 38, and she lands at 20%, where she is still "
        "disagreeing with the model. Declining is not the pathology; collapsing is"
    )
    # The floor is inclusive: "at or under this is a stamp". No reviewer in this lesson's log
    # lands exactly on it, so the fixture cannot tell `<=` from `<`, and the difference is a
    # reviewer who is waved through for being precisely as bad as the threshold allows.
    n_side = 200
    at_floor = lesson.automation_bias(_view(
        [_record(f"F-L{k}", confidence=0.6, recommendation="approve",
                 outcome="decline" if k < round(n_side * 0.40) else "approve",
                 verifier_ids=["u:exact"]) for k in range(n_side)]
        + [_record(f"F-H{k}", confidence=0.95, recommendation="approve",
                   outcome="decline"
                   if k < round(n_side * lesson.HIGH_CONFIDENCE_FLOOR) else "approve",
                   verifier_ids=["u:exact"]) for k in range(n_side)]))["by_reviewer"]["u:exact"]
    assert at_floor["high_rate"] == lesson.HIGH_CONFIDENCE_FLOOR, (
        f"this rubric case is built to land exactly on the floor and came back "
        f"{at_floor['high_rate']!r}; this is a rubric bug, not a student one"
    )
    assert at_floor["status"] == "automation_bias", (
        f"a reviewer whose high-confidence override rate is EXACTLY "
        f"{lesson.HIGH_CONFIDENCE_FLOOR} came back {at_floor['status']!r}. The ladder reads "
        "`high_rate <= high_confidence_floor` — at or under the floor, not strictly under it. "
        "A strict `<` clears the one reviewer sitting precisely on the line the policy drew"
    )
    # The sensitivity is SIGNED. Overriding MORE as confidence rises is a different problem.
    inverted = lesson.automation_bias(_view(curve("u:inv", 0.02, 0.40)))
    assert inverted["by_reviewer"]["u:inv"]["sensitivity"] < 0
    assert inverted["by_reviewer"]["u:inv"]["status"] == "confidence_independent", (
        f"a reviewer who overrides 2% of low-confidence scores and 40% of high-confidence "
        f"ones came back {inverted['by_reviewer']['u:inv']['status']!r}. Comparing "
        "|sensitivity| files that as automation bias, which it is the opposite of"
    )
    assert inverted["findings"] == []
    # The floor is COMPUTED from the sample, and a fixed ten points is wrong in both
    # directions. Small sample: a 15-point decline this many decisions could not have found.
    thin = lesson.automation_bias(_view(curve("u:thin", 0.45, 0.30)[:0] + [
        _record(f"T-L{k}", confidence=0.6, recommendation="approve",
                outcome="decline" if k < 18 else "approve", verifier_ids=["u:thin"])
        for k in range(40)] + [
        _record(f"T-H{k}", confidence=0.95, recommendation="approve",
                outcome="decline" if k < 12 else "approve", verifier_ids=["u:thin"])
        for k in range(40)]))
    assert thin["by_reviewer"]["u:thin"]["min_detectable"] > 0.10
    assert thin["by_reviewer"]["u:thin"]["status"] == "confidence_independent", (
        f"u:thin came back {thin['by_reviewer']['u:thin']['status']!r}. Her rate drops 15 "
        f"points, but on 40 decisions a side this test could only have found "
        f"{thin['by_reviewer']['u:thin']['min_detectable']:.3f}. A fixed ten-point threshold "
        "publishes that 15 points as a finding the data does not support"
    )
    # Large sample: a 7-point decline that a fixed ten-point threshold would throw away.
    wide = lesson.automation_bias(_view([
        _record(f"W-L{k}", confidence=0.6, recommendation="approve",
                outcome="decline" if k < 160 else "approve", verifier_ids=["u:wide"])
        for k in range(1000)] + [
        _record(f"W-H{k}", confidence=0.95, recommendation="approve",
                outcome="decline" if k < 90 else "approve", verifier_ids=["u:wide"])
        for k in range(1000)]))
    assert wide["by_reviewer"]["u:wide"]["min_detectable"] < 0.10
    assert wide["by_reviewer"]["u:wide"]["status"] == "confidence_sensitive", (
        f"u:wide came back {wide['by_reviewer']['u:wide']['status']!r}. Her rate drops 7 "
        f"points over 2000 decisions, where the detectable floor is "
        f"{wide['by_reviewer']['u:wide']['min_detectable']:.3f}. A fixed ten-point threshold "
        "discards a real decline because the constant was picked for a different sample size"
    )
    # high_confidence_floor is an argument too.
    raised = lesson.automation_bias(_view(curve("u:real", 0.40, 0.20)),
                                    high_confidence_floor=0.25)
    assert raised["by_reviewer"]["u:real"]["status"] == "automation_bias", (
        "raising high_confidence_floor above this reviewer's high-confidence rate did not "
        "change her status — the floor is an argument, not HIGH_CONFIDENCE_FLOOR"
    )


def test_bias_findings_order_and_a_clean_verdict():
    got = lesson.automation_bias(VIEW)
    expected = _expected_bias(VIEW)
    want = sorted([v for v, c in expected.items() if c["status"] == "automation_bias"],
                  key=lambda v: (-expected[v]["sensitivity"], v))
    assert got["findings"] == want, f"findings came back {got['findings']}, expected {want}"
    assert got["verdict"] == "fail"

    def curve(vid, low_rate, high_rate, n=200):
        rows = []
        for k in range(n):
            rows.append(_record(f"{vid}-L{k}", confidence=0.6, recommendation="approve",
                                outcome="decline" if k < n * low_rate else "approve",
                                verifier_ids=[vid]))
        for k in range(n):
            rows.append(_record(f"{vid}-H{k}", confidence=0.95, recommendation="approve",
                                outcome="decline" if k < n * high_rate else "approve",
                                verifier_ids=[vid]))
        return rows
    two = lesson.automation_bias(_view(curve("u:aaa", 0.25, 0.01) + curve("u:zzz", 0.55, 0.01)))
    assert two["findings"] == ["u:zzz", "u:aaa"], (
        f"findings came back {two['findings']}. Two reviewers collapse here, by 54 points and "
        "by 24. They are ordered by sensitivity DESCENDING, so the worst is what an inspector "
        "reads first; alphabetical order would put the smaller harm at the top"
    )
    clean = lesson.automation_bias(_view(curve("u:flat", 0.30, 0.30)))
    assert clean["findings"] == [] and clean["verdict"] == "pass", (
        f"a reviewer whose rate does not move came back {clean['verdict']!r} with findings "
        f"{clean['findings']}. A verdict hard-wired to 'fail' carries no information"
    )
    assert clean["by_reviewer"]["u:flat"]["status"] == "confidence_independent"
    # A reviewer who overrides almost nothing at EVERY confidence level has a high-confidence
    # rate under the floor and a sensitivity under their own detectable difference. Their rate
    # does not respond to the model, so this is not automation bias — it is a finding about
    # the LEVEL, which override_rates reports. Testing the level first files one as the other.
    stamp = lesson.automation_bias(_view([
        _record(f"S-L{k}", confidence=0.6, recommendation="approve",
                outcome="decline" if k < 12 else "approve", verifier_ids=["u:stamp"])
        for k in range(400)] + [
        _record(f"S-H{k}", confidence=0.95, recommendation="approve",
                outcome="decline" if k < 8 else "approve", verifier_ids=["u:stamp"])
        for k in range(400)]))
    cell = stamp["by_reviewer"]["u:stamp"]
    assert cell["high_rate"] <= lesson.HIGH_CONFIDENCE_FLOOR and cell["sensitivity"] > 0
    assert cell["status"] == "confidence_independent" and stamp["findings"] == [], (
        f"a reviewer overriding 3% of low-confidence and 2% of high-confidence scores came "
        f"back {cell['status']!r} with findings {stamp['findings']}. Her high-confidence rate "
        "is under the floor, but her sensitivity is under her own detectable difference: the "
        "rate does not respond to the model at all, so it is not automation bias. Test the "
        "sensitivity BEFORE the level"
    )


# ----------------------------------------------------------------- exercise 6
def test_gate_verifier_requirement_depends_on_the_basis():
    policy = lesson.POLICY
    one = lesson.oversight_gate(_record("D-1", identification_basis="credit_file",
                                        outcome="approve", duration_seconds=90.0,
                                        verifier_ids=["u:a"]), policy)
    assert one["allowed"] is True and one["reasons"] == [], (
        f"a credit_file decision with one verifier came back {one['reasons']} — the policy "
        "asks for one there"
    )
    two = lesson.oversight_gate(_record("D-2", identification_basis="remote_biometric",
                                        outcome="approve", duration_seconds=90.0,
                                        verifier_ids=["u:a"]), policy)
    assert two["reasons"] == ["insufficient_verifiers"], (
        f"a remote_biometric decision with one verifier came back {two['reasons']} — Article "
        "14(5) asks for two there, and the requirement is looked up by identification_basis"
    )
    dup = lesson.oversight_gate(_record("D-3", identification_basis="remote_biometric",
                                        outcome="approve", duration_seconds=90.0,
                                        verifier_ids=["u:a", "u:a"]), policy)
    assert dup["reasons"] == ["insufficient_verifiers"], (
        "two entries and one person passed the gate. Count distinct verifier ids"
    )
    unknown = lesson.oversight_gate(_record("D-4", identification_basis="something_new",
                                            outcome="approve", duration_seconds=90.0,
                                            verifier_ids=["u:a"]), policy)
    assert unknown["allowed"] is True, (
        f"a basis the policy does not mention came back {unknown['reasons']} — default to one "
        "verifier rather than raising a KeyError on a new product line"
    )
    # The default itself, which the case above cannot see: with one verifier present, a
    # default of 1 and a default of 0 agree. With NONE present they do not, and a default of
    # 0 makes the gate fail OPEN on exactly the decisions it has never been told about.
    nobody = lesson.oversight_gate(_record("D-5", identification_basis="something_new",
                                           outcome="approve", duration_seconds=90.0,
                                           verifier_ids=[]), policy)
    assert nobody["reasons"] == ["insufficient_verifiers"], (
        f"an unrecognised basis with NO verifiers came back {nobody['reasons']}, expected "
        "['insufficient_verifiers'] — the default for a basis the policy does not mention is "
        "ONE verifier, not zero. Defaulting to zero lets a brand-new product line through the "
        "gate unreviewed, which is the one thing the gate exists to stop"
    )


def test_gate_clock_anomaly_precedes_reviewed_too_fast():
    policy = lesson.POLICY
    back = lesson.oversight_gate(_record("D-1", outcome="approve", duration_seconds=-40.0,
                                         verifier_ids=["u:a"]), policy)
    assert back["reasons"] == ["clock_anomaly"], (
        f"a decision whose duration is -40 seconds came back {back['reasons']}. -40 < 10 is "
        "true, so a naive comparison also reports reviewed_too_fast — which accuses a named "
        "person of rushing a case, and buries the clock bug that produced the number"
    )
    fast = lesson.oversight_gate(_record("D-2", outcome="approve", duration_seconds=4.0,
                                         verifier_ids=["u:a"]), policy)
    assert fast["reasons"] == ["reviewed_too_fast"]
    edge = lesson.oversight_gate(_record("D-3", outcome="approve",
                                         duration_seconds=float(policy["min_seconds"]),
                                         verifier_ids=["u:a"]), policy)
    assert edge["allowed"] is True, (
        "a decision taking exactly min_seconds was blocked — the comparison is strict"
    )
    missing = lesson.oversight_gate(_record("D-4", outcome="approve", duration_seconds=None,
                                            verifier_ids=["u:a"]), policy)
    assert missing["reasons"] == [], (
        f"a decision with no duration at all came back {missing['reasons']} — None is not a "
        "fast review, and comparing it to a number raises"
    )


def test_gate_reads_its_flags_out_of_the_policy():
    strict = dict(lesson.POLICY)
    record = _record("D-1", outcome=None, duration_seconds=90.0,
                     submitted_by="u:a", verifier_ids=["u:a"])
    got = lesson.oversight_gate(record, strict)
    assert got["reasons"] == ["no_outcome", "self_verification"], (
        f"came back {got['reasons']}, expected both no_outcome and self_verification — the "
        "submitter is the only verifier, and the decision never closed"
    )
    relaxed = lesson.oversight_gate(record, {**strict, "require_outcome": False,
                                             "forbid_self_verification": False})
    assert relaxed["reasons"] == [], (
        f"with both flags off the same record came back {relaxed['reasons']} — read the flags "
        "out of the policy argument rather than hard-coding them"
    )
    slow = lesson.oversight_gate(_record("D-2", outcome="approve", duration_seconds=90.0,
                                         verifier_ids=["u:a"]),
                                 {**strict, "min_seconds": 600})
    assert slow["reasons"] == ["reviewed_too_fast"], (
        "min_seconds is a policy field; a 90-second review is too fast under a 600-second one"
    )
    co_signed = lesson.oversight_gate(_record("D-3", identification_basis="remote_biometric",
                                              outcome="approve", duration_seconds=90.0,
                                              submitted_by="u:a",
                                              verifier_ids=["u:a", "u:b"]), strict)
    assert co_signed["reasons"] == ["self_verification"], (
        f"came back {co_signed['reasons']} — two distinct people signed, so the count is "
        "satisfied, but one of them submitted the case"
    )


def test_gate_reasons_are_sorted_and_allowed_is_derived():
    record = _record("D-1", identification_basis="remote_biometric", outcome=None,
                     duration_seconds=-5.0, submitted_by="u:a", verifier_ids=["u:a"])
    got = lesson.oversight_gate(record, lesson.POLICY)
    assert got["decision_id"] == "D-1"
    assert got["reasons"] == sorted(got["reasons"]), (
        f"reasons came back {got['reasons']} — sort them, so two runs of the same report diff "
        "cleanly"
    )
    assert set(got["reasons"]) == {"insufficient_verifiers", "no_outcome", "clock_anomaly",
                                   "self_verification"}, (
        f"came back {got['reasons']}; this record fails the policy four different ways"
    )
    assert got["allowed"] is False
    clean = lesson.oversight_gate(_record("D-2", outcome="approve", duration_seconds=90.0,
                                          verifier_ids=["u:a"]), lesson.POLICY)
    assert clean["allowed"] is True and clean["reasons"] == [], (
        "a decision that satisfies the policy must come back allowed; a gate that blocks "
        "everything is not a gate"
    )


def test_gate_report_totals_reasons_and_scope():
    got = lesson.gate_report(VIEW, lesson.POLICY)
    blocked: list = []
    by_reason: dict = {}
    in_scope = 0
    for did in IDS:
        verdict = lesson.oversight_gate(VIEW[did], lesson.POLICY)
        if verdict["allowed"]:
            continue
        blocked.append(did)
        if VIEW[did]["identification_basis"] == "remote_biometric":
            in_scope += 1
        for reason in verdict["reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    assert got["n"] == len(IDS) and got["blocked"] == len(blocked), (
        f"n/blocked came back {(got['n'], got['blocked'])}, expected "
        f"{(len(IDS), len(blocked))}"
    )
    assert got["allowed"] == len(IDS) - len(blocked)
    assert got["blocked_ids"] == sorted(blocked), "blocked_ids must be the sorted ids"
    assert abs(got["blocked_rate"] - len(blocked) / len(IDS)) < 1e-12
    assert got["by_reason"] == by_reason, (
        f"by_reason came back {got['by_reason']}, expected {by_reason}"
    )
    assert sum(got["by_reason"].values()) > got["blocked"], (
        "the reason counts sum to no more than the blocked count, so a decision that fails "
        "the policy two ways is being counted once. Seven decisions here fail twice"
    )
    assert got["in_scope_blocked"] == in_scope, (
        f"in_scope_blocked came back {got['in_scope_blocked']}, expected {in_scope} — the "
        "blocked decisions whose basis is remote_biometric"
    )


RUBRIC = [
    (test_view_shape_and_coverage, 4,
     "walk the log in seq order; skip events with no decision_id; first and last seq"),
    (test_view_folds_scalar_fields_and_never_overwrites_with_nothing, 4,
     "fold the payload fields; None, '' and [] never overwrite a real value"),
    (test_view_keeps_duplicate_verifiers_in_sequence_order, 4,
     "keep every verifier entry, in seq order — do not deduplicate and do not sort by ts"),
    (test_view_durations_are_signed_and_may_be_missing, 3,
     "use_end minus use_start, signed; None when either is missing"),
    (test_override_exclusions_and_overall_rate, 4,
     "exclude undecided, then unscored, then unverified — in that order, counted once each"),
    (test_override_by_reviewer_counts_distinct_persons, 4,
     "one decision counts once per DISTINCT verifier on it"),
    (test_override_by_outcome_is_keyed_on_the_model_recommendation, 4,
     "key by_outcome on recommendation, not on the recorded outcome"),
    (test_override_support_floor_is_the_parameter, 3,
     "min_support is an argument, the floor is inclusive, and nobody is dropped"),
    (test_time_anomalies_never_enter_the_distribution, 4,
     "a negative duration leaves entirely: no percentile, no fast list, no reviewer median"),
    (test_time_lists_are_exact_and_sorted, 4,
     "fast is strictly below the floor and sorted; incomplete counts the missing durations"),
    (test_time_floor_is_the_parameter, 2,
     "floor_seconds is an argument, not MIN_PLAUSIBLE_SECONDS"),
    (test_time_by_reviewer_medians_and_distinct_verifiers, 3,
     "one entry per decision per distinct verifier; numpy's linear median"),
    (test_four_eyes_scope_is_the_basis_parameter, 4,
     "only the records whose identification_basis equals basis are in scope"),
    (test_four_eyes_counts_persons_not_entries, 5,
     "len(set(verifier_ids)) >= 2; a length test passes every duplicate in this log"),
    (test_four_eyes_reason_ladder, 4,
     "no_verification, then duplicate_verifier, then single_verifier — in that order"),
    (test_four_eyes_rate_verdict_and_a_clean_log, 3,
     "rate is over in-scope decisions; a clean log must come back pass"),
    (test_bias_per_bin_counts_match_an_independent_derivation, 4,
     "confidence_bin per record; skip anything with no outcome, recommendation or confidence"),
    (test_bias_halves_sensitivity_and_detectable_floor, 4,
     "the high half is confidence >= split; the floor is pooled over both halves"),
    (test_bias_support_floor_comes_first_and_is_the_parameter, 5,
     "the floor applies to min(n_low, n_high), before any comparison, and is an argument"),
    (test_bias_separates_collapse_from_decline_and_is_signed, 5,
     "signed sensitivity against the reviewer's own floor, then the high-confidence level"),
    (test_bias_findings_order_and_a_clean_verdict, 4,
     "findings are the automation_bias reviewers, sensitivity descending then id"),
    (test_gate_verifier_requirement_depends_on_the_basis, 4,
     "look the requirement up by identification_basis; default to 1; count distinct ids"),
    (test_gate_clock_anomaly_precedes_reviewed_too_fast, 4,
     "test the sign before the floor; the floor comparison is strict; None never compares"),
    (test_gate_reads_its_flags_out_of_the_policy, 4,
     "require_outcome, forbid_self_verification and min_seconds all come from the policy"),
    (test_gate_reasons_are_sorted_and_allowed_is_derived, 3,
     "sort the reasons; allowed is exactly the empty-reasons case"),
    (test_gate_report_totals_reasons_and_scope, 4,
     "count every reason on every blocked decision, so the counts need not sum to blocked"),
]
