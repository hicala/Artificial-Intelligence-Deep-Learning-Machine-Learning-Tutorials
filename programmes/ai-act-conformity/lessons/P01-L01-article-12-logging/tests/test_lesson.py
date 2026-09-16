"""Autograder rubric for P01-L01. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Every expected value here is computed from the event stream by code in THIS file, never by
  calling the student's own functions. `_expected_trace_seqs` and `_expected_missing` are
  independent re-derivations; if they ever disagree with the lesson, the lesson is wrong.
* `build_log` stores the event dicts themselves, so any test that tampers with an entry works
  on a `copy.deepcopy` of the stream. Mutating the module-level fixture would leak into every
  later test in the same process and make failures depend on rubric order.
* The chain tests never assert a literal digest. The digest depends on the canonical form, and
  a student who chooses `ensure_ascii=True` has an equally correct log. What is graded is the
  behaviour a chain has to have: an edit, a deletion, a relink and a whole-tail rewrite are
  each detected, and the digest demonstrably covers both the event and the history.
* `test_verify_chain_anchor` is what stops a verifier that only checks internal consistency
  from scoring full marks. The rewritten log it builds is flawless by every other test in the
  file; only the published head catches it. That is the point of the exercise.
* `test_retention_scans_for_extremes` and `test_retention_partition_and_boundary` use a log
  whose ages run young, ancient, young. On the shipped stream the ages happen to descend, so
  an implementation that sorts and splits at an index, or reads `log[0]` and `log[-1]`, passes
  by accident. On this one it does not.
* `test_completeness_rejects_empty_values` checks 0 and False as well as None and "". A
  `if not value` test rejects a recorded outcome of 0, which is a real answer in a scoring
  system; the exercise asks for emptiness, not falsiness.
"""
import copy
import importlib.util, os
from datetime import date
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

STREAM = lesson.EVENT_STREAM
AS_OF = lesson.AS_OF
HEXDIGITS = set("0123456789abcdef")


def _fresh_stream() -> list:
    """A private copy of the event stream, safe to tamper with."""
    return copy.deepcopy(STREAM)


def _expected_trace_seqs(decision_id: str) -> list:
    """Re-derive a decision's trace independently of the student's reconstruct_decision."""
    own = [i for i, e in enumerate(STREAM) if e.get("decision_id") == decision_id]
    versions = {STREAM[i].get("payload", {}).get("model_version") for i in own}
    versions.discard(None)
    context = [i for i, e in enumerate(STREAM)
               if e.get("event_type") == "model_deployed"
               and e.get("payload", {}).get("model_version") in versions]
    return sorted(set(own) | set(context)) if own else []


def _expected_missing(decision_id: str) -> list:
    """Re-derive which required fields a decision lacks, without calling the lesson."""
    def real(value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, dict, set)):
            return bool(value)
        return True

    seen = set()
    for i in _expected_trace_seqs(decision_id):
        event = STREAM[i]
        for key, value in list(event.items()) + list(event.get("payload", {}).items()):
            if real(value):
                seen.add(key)
    return sorted(set(lesson.REQUIRED_FIELDS) - seen)


def _ages() -> list:
    return [(AS_OF - lesson.parse_ts(e["ts"]).date()).days for e in STREAM]


# --- exercise 1: canonical bytes and the chain -----------------------------------------

def test_canonical_bytes_is_order_independent():
    cb = lesson.canonical_bytes
    assert isinstance(cb({"a": 1}), bytes), (
        f"canonical_bytes returned {type(cb({'a': 1})).__name__}, not bytes — encode the "
        "dumped string as UTF-8"
    )
    assert cb({"b": 1, "a": 2}) == cb({"a": 2, "b": 1}), (
        "two dicts with the same items in a different insertion order must serialise "
        "identically: json.dumps(..., sort_keys=True)"
    )
    assert cb({"x": {"b": 1, "a": 2}, "y": [1, {"d": 4, "c": 3}]}) == \
           cb({"y": [1, {"c": 3, "d": 4}], "x": {"a": 2, "b": 1}}), (
        "sort_keys sorts every level of nesting; sorting only the top level leaves the chain "
        "dependent on how a nested payload happened to be built"
    )
    assert cb({"a": 1}) != cb({"a": 2}), "different values must serialise differently"
    assert cb({"a": 1}) != cb({"b": 1}), "different keys must serialise differently"


def test_canonical_bytes_round_trips():
    import json
    for obj in ({"a": [1, 2], "b": "été"},
                {"z": None, "n": 0, "f": False, "nested": {"k": ["v", {"q": 1}]}},
                {"ts": "2026-09-16T09:00:00Z", "payload": {"score": 0.8231}}):
        raw = lesson.canonical_bytes(obj)
        assert json.loads(raw.decode("utf-8")) == obj, (
            f"canonical_bytes({obj!r}) did not round-trip through json.loads — the canonical "
            "form must stay JSON, not become repr(), str(sorted(...)) or pickle"
        )


def test_append_event_shape_and_links():
    log = []
    first = lesson.append_event(log, {"ts": "2026-01-01T00:00:00Z", "event_type": "boot"})
    assert set(first) == {"seq", "event", "prev_hash", "entry_hash"}, (
        f"an entry carries exactly those four keys; yours has {sorted(first)}"
    )
    assert first["seq"] == 0, f"the first entry is seq 0, got {first['seq']}"
    assert first["prev_hash"] == lesson.GENESIS_HASH, (
        f"the first entry's prev_hash is GENESIS_HASH, got {first['prev_hash']!r}"
    )
    assert len(log) == 1 and log[0] is first, (
        "append_event must append to `log` in place AND return the entry it appended"
    )
    digest = first["entry_hash"]
    assert isinstance(digest, str) and len(digest) == 64 and set(digest) <= HEXDIGITS, (
        f"entry_hash must be the 64-character lowercase hex digest from .hexdigest(), "
        f"got {digest!r}"
    )
    second = lesson.append_event(log, {"ts": "2026-01-01T00:00:01Z", "event_type": "tick"})
    assert second["seq"] == 1, f"the second entry is seq 1, got {second['seq']}"
    assert second["prev_hash"] == first["entry_hash"], (
        "each entry's prev_hash is the PREVIOUS entry's entry_hash — not its own, not genesis"
    )
    chained = lesson.build_log(STREAM)
    assert [e["seq"] for e in chained] == list(range(len(STREAM))), (
        "a chained stream has contiguous sequence numbers from 0"
    )
    assert all(chained[i]["prev_hash"] == chained[i - 1]["entry_hash"]
               for i in range(1, len(chained))), "the chain must link every consecutive pair"
    assert all(e["event"] is s for e, s in zip(chained, STREAM)), (
        "the entry stores the event unchanged; do not copy, re-key or re-serialise it"
    )


def test_append_event_digest_covers_event_and_history():
    same_head = {"ts": "2026-01-01T00:00:00Z", "event_type": "boot"}
    other_head = {"ts": "2026-01-01T00:00:00Z", "event_type": "different"}
    tail_a = {"ts": "2026-01-01T00:00:01Z", "event_type": "tick"}
    tail_b = {"ts": "2026-01-01T00:00:01Z", "event_type": "tock"}

    baseline = lesson.build_log([same_head, tail_a])[1]["entry_hash"]
    changed_tail = lesson.build_log([same_head, tail_b])[1]["entry_hash"]
    changed_head = lesson.build_log([other_head, tail_a])[1]["entry_hash"]

    assert baseline != changed_tail, (
        "two logs with identical first events and different second events produced the same "
        "second hash — the digest does not cover the event, so no edit is ever detectable"
    )
    assert baseline != changed_head, (
        "two logs with identical second events and different first events produced the same "
        "second hash — the digest does not cover prev_hash, so the chain proves nothing about "
        "what came before it and entries can be reordered freely"
    )
    # Two identical events at different positions must still differ, via seq or prev_hash.
    repeated = lesson.build_log([same_head, same_head])
    assert repeated[0]["entry_hash"] != repeated[1]["entry_hash"], (
        "the same event logged twice produced the same digest twice — include seq and "
        "prev_hash in the digest so position is part of what is signed"
    )


# --- exercise 2: verify_chain ----------------------------------------------------------

def test_verify_chain_accepts_a_clean_log():
    verdict = lesson.verify_chain(lesson.build_log(STREAM))
    assert isinstance(verdict, lesson.ChainVerdict), (
        f"return a ChainVerdict, got {type(verdict).__name__}"
    )
    assert verdict.intact is True, f"the freshly built log must verify, got {verdict!r}"
    assert verdict.first_bad_seq is None and verdict.reason == "", (
        f"an intact verdict carries no sequence number and no reason, got {verdict!r}"
    )
    empty = lesson.verify_chain([])
    assert empty.intact and empty.first_bad_seq is None, (
        f"an empty log is vacuously intact, got {empty!r}"
    )


def test_verify_chain_catches_an_edit():
    for target in (0, 4, len(STREAM) - 1):
        edited = lesson.build_log(_fresh_stream())
        edited[target]["event"]["event_type"] = "quietly-rewritten"
        verdict = lesson.verify_chain(edited)
        assert not verdict.intact, (
            f"editing the event at seq {target} went undetected — recompute each entry's "
            "digest from its own contents instead of trusting the stored value"
        )
        assert verdict.first_bad_seq == target, (
            f"the edit at seq {target} was reported at seq {verdict.first_bad_seq} — return at "
            "the FIRST break, walking the entries in order"
        )
        assert "entry_hash" in verdict.reason and "sequence" not in verdict.reason, (
            f"the reason for an edited event must name entry_hash and nothing else, got "
            f"{verdict.reason!r} — the numbering and the links are intact here"
        )
    # A payload edit, not just an envelope one: the digest covers the whole event.
    deep = lesson.build_log(_fresh_stream())
    deep[4]["event"]["payload"]["outcome"] = "approve"
    assert lesson.verify_chain(deep).first_bad_seq == 4, (
        "an edit buried in a nested payload must be caught too — hash the whole event, not a "
        "handful of its top-level fields"
    )


def test_verify_chain_catches_a_deletion():
    for target in (0, 6, len(STREAM) - 1):
        pruned = lesson.build_log(_fresh_stream())
        del pruned[target]
        verdict = lesson.verify_chain(pruned)
        if target == len(STREAM) - 1:
            # Removing the LAST entry leaves a chain that is internally perfect. Only a
            # published head can catch it, which is the next test's job; here we simply make
            # sure the verifier does not invent a break that is not there.
            assert verdict.intact, (
                "dropping the final entry leaves seq numbers contiguous and every link valid, "
                f"so the chain alone cannot object: got {verdict!r}"
            )
            continue
        assert not verdict.intact and verdict.first_bad_seq == target, (
            f"deleting the entry at index {target} leaves a gap in the sequence there, "
            f"got {verdict!r}"
        )
        # A deletion breaks the link as well as the numbering, so a verifier with no
        # contiguity check still fails the entry — for the wrong reason, and it sends an
        # inspector hunting a forgery instead of a purge job. Check the numbering first.
        assert "sequence" in verdict.reason and "prev_hash" not in verdict.reason, (
            f"a gap at index {target} was reported as {verdict.reason!r} — check "
            "entry['seq'] == i BEFORE the link, and say 'sequence' in the reason"
        )


def test_verify_chain_catches_a_relink():
    relinked = lesson.build_log(_fresh_stream())
    relinked[3]["prev_hash"] = lesson.GENESIS_HASH
    verdict = lesson.verify_chain(relinked)
    assert not verdict.intact and verdict.first_bad_seq == 3, (
        f"re-pointing seq 3 at genesis must break the link check there, got {verdict!r}"
    )
    assert "prev_hash" in verdict.reason and "sequence" not in verdict.reason, (
        f"the reason must name prev_hash and only prev_hash, got {verdict.reason!r} — the "
        "sequence numbers are intact here, so blaming them is a false finding"
    )
    swapped = lesson.build_log(_fresh_stream())
    swapped[5]["prev_hash"] = swapped[2]["entry_hash"]
    assert lesson.verify_chain(swapped).first_bad_seq == 5, (
        "an entry pointed at some OTHER real entry's hash is still a break — compare against "
        "the immediately preceding entry, not against the set of hashes in the log"
    )


def test_verify_chain_anchor():
    sealed = lesson.build_log(STREAM)
    head = sealed[-1]["entry_hash"]
    assert lesson.verify_chain(sealed, expected_head=head).intact, (
        "the log's own head hash must satisfy the anchor check"
    )
    assert not lesson.verify_chain(sealed, expected_head="f" * 64).intact, (
        "a head that does not match must fail, even though the chain is internally perfect"
    )
    assert lesson.verify_chain([], expected_head=lesson.GENESIS_HASH).intact, (
        "an empty log's head is GENESIS_HASH, so that anchor must be accepted"
    )
    assert not lesson.verify_chain([], expected_head="a" * 64).intact, (
        "an empty log does not satisfy an anchor pointing at anything but GENESIS_HASH"
    )

    forged_stream = _fresh_stream()
    forged_stream[4]["payload"]["outcome"] = "approve"
    forged = lesson.build_log(forged_stream)
    assert lesson.verify_chain(forged).intact, (
        "a log rewritten from the edit onwards is internally consistent by construction — if "
        "your verifier objects to it without an anchor, it is objecting to the wrong thing"
    )
    verdict = lesson.verify_chain(forged, expected_head=head)
    assert not verdict.intact, (
        "the forged log must fail against the head published before the forgery — this is the "
        "only check in the file that catches a whole-tail rewrite"
    )
    assert "head" in verdict.reason, (
        f"the reason for an anchor mismatch must name the head, got {verdict.reason!r}"
    )
    assert verdict.first_bad_seq is None, (
        f"first_bad_seq came back {verdict.first_bad_seq!r} — a head mismatch does not "
        "localise the tampering, because every entry re-hashed cleanly; report None"
    )


# --- exercise 3: retention -------------------------------------------------------------

def test_retention_partition_and_boundary():
    log = lesson.plain_log(STREAM)
    report = lesson.retention_status(log, AS_OF)
    assert set(report) == {"as_of", "minimum_days", "oldest_ts", "newest_ts", "retained",
                           "expired", "missing_seq", "compliant"}, (
        f"keys were {sorted(report)} — return exactly the eight documented names"
    )
    assert report["as_of"] == AS_OF.isoformat(), (
        f"as_of must be the ISO string, got {report['as_of']!r}"
    )
    floor = lesson.RETENTION_MINIMUM_DAYS
    assert report["minimum_days"] == floor, "echo back the floor that was used"
    ages = _ages()
    expected_expired = sorted(i for i, age in enumerate(ages) if age > floor)
    expected_retained = sorted(i for i, age in enumerate(ages) if age <= floor)
    assert report["expired"] == expected_expired, (
        f"expired={report['expired']}, expected {expected_expired}"
    )
    assert report["retained"] == expected_retained, (
        f"retained={report['retained']}, expected {expected_retained} — two entries sit "
        "EXACTLY on the floor and are still inside the window: compare with > , not >="
    )
    assert sorted(report["retained"] + report["expired"]) == list(range(len(log))), (
        "retained and expired must partition every entry exactly once, with no entry in both"
    )
    # Young, ancient, young: no index splits this list, and neither end is an extreme.
    mixed = lesson.plain_log([{"ts": lesson._ts(1), "event_type": "a", "payload": {}},
                              {"ts": lesson._ts(400), "event_type": "b", "payload": {}},
                              {"ts": lesson._ts(2), "event_type": "c", "payload": {}}])
    got = lesson.retention_status(mixed, AS_OF)
    assert got["expired"] == [1] and got["retained"] == [0, 2], (
        f"expired={got['expired']} retained={got['retained']} — decide per entry; sorting the "
        "log and splitting it at a cut-off index gets this wrong"
    )


def test_retention_scans_for_extremes():
    report = lesson.retention_status(lesson.plain_log(STREAM), AS_OF)
    assert report["oldest_ts"] == min(e["ts"] for e in STREAM), (
        f"oldest_ts={report['oldest_ts']!r} is the MINIMUM timestamp over the whole log"
    )
    assert report["newest_ts"] == max(e["ts"] for e in STREAM), (
        f"newest_ts={report['newest_ts']!r} is the MAXIMUM timestamp over the whole log"
    )
    mixed = lesson.plain_log([{"ts": lesson._ts(1), "event_type": "a", "payload": {}},
                              {"ts": lesson._ts(400), "event_type": "b", "payload": {}},
                              {"ts": lesson._ts(2), "event_type": "c", "payload": {}}])
    got = lesson.retention_status(mixed, AS_OF)
    assert got["oldest_ts"] == lesson._ts(400), (
        f"oldest_ts={got['oldest_ts']!r} — the oldest entry is in the middle of this log, so "
        "reading log[0] gives the wrong answer"
    )
    assert got["newest_ts"] == lesson._ts(1), (
        f"newest_ts={got['newest_ts']!r} — the newest entry is the FIRST in this log, so "
        "reading log[-1] gives the wrong answer"
    )


def test_retention_finds_gaps():
    log = lesson.plain_log(STREAM)
    clean = lesson.retention_status(log, AS_OF)
    assert clean["missing_seq"] == [] and clean["compliant"] is True, (
        f"the shipped stream is contiguous: got missing_seq={clean['missing_seq']}"
    )
    punched = [e for e in log if e["seq"] not in (2, 9)]
    holed = lesson.retention_status(punched, AS_OF)
    assert holed["missing_seq"] == [2, 9], (
        f"seqs 2 and 9 were removed, so both are missing: got {holed['missing_seq']} — the "
        "gaps are the seq numbers absent between 0 and the largest seq present, not "
        "the difference between len(log) and the largest seq"
    )
    assert holed["compliant"] is False, "a log with a hole punched in it is not compliant"
    # Removing two entries from NEAR the end shortens the list past the second gap, so
    # range(len(log)) stops one short of it and reports only the first.
    late = [e for e in log if e["seq"] not in (len(log) - 3, len(log) - 2)]
    want_late = [len(log) - 3, len(log) - 2]
    assert lesson.retention_status(late, AS_OF)["missing_seq"] == want_late, (
        f"got {lesson.retention_status(late, AS_OF)['missing_seq']}, expected {want_late} — "
        "the range runs to the LARGEST seq present, not to len(log)"
    )
    trailing = [e for e in log if e["seq"] < len(log) - 3]
    assert lesson.retention_status(trailing, AS_OF)["missing_seq"] == [], (
        "entries dropped from the END leave no gap: the range runs to the largest seq PRESENT, "
        "so a truncated log reports no missing sequence numbers"
    )
    empty = lesson.retention_status([], AS_OF)
    assert empty["oldest_ts"] is None and empty["newest_ts"] is None, (
        f"an empty log has no timestamps, got {empty['oldest_ts']!r}/{empty['newest_ts']!r}"
    )
    assert empty["missing_seq"] == [] and empty["compliant"] is True, (
        f"an empty log has no gaps and nothing to answer for, got {empty!r} — max() over an "
        "empty sequence is a ValueError, so guard it"
    )


# --- exercise 4: reconstructing one decision -------------------------------------------

def test_reconstruct_selects_and_orders():
    log = lesson.plain_log(STREAM)
    for decision_id in ("D-1001", "D-1002", "D-1003", "D-1004"):
        got = [e["seq"] for e in lesson.reconstruct_decision(log, decision_id)]
        want = _expected_trace_seqs(decision_id)
        assert got == want, f"{decision_id}: got seqs {got}, expected {want}"
    skewed = [e["seq"] for e in lesson.reconstruct_decision(log, "D-1003")]
    assert skewed == sorted(skewed), (
        f"D-1003 came back as {skewed} — its closing event carries a timestamp two days "
        "earlier than its opening one, so sorting by ts reverses the decision. Sort by seq"
    )


def test_reconstruct_pulls_in_deployment_context():
    log = lesson.plain_log(STREAM)
    for decision_id, version in (("D-1002", "loan-risk-2025.11"),
                                 ("D-1004", "loan-risk-2026.04")):
        trace = lesson.reconstruct_decision(log, decision_id)
        deploys = [e for e in trace if e["event"]["event_type"] == "model_deployed"]
        assert len(deploys) == 1, (
            f"{decision_id} must pull in exactly one model_deployed entry, got {len(deploys)} "
            "— a trace without the deployment cannot say what decided the case, and a trace "
            "with every deployment in the log is not a trace"
        )
        assert deploys[0]["event"]["payload"]["model_version"] == version, (
            f"{decision_id} names {version}, but the trace carries "
            f"{deploys[0]['event']['payload']['model_version']} — match on the model_version "
            "the decision's own events reference"
        )
    # A decision whose events name no model version pulls in no deployment at all.
    bare = lesson.plain_log([
        {"ts": lesson._ts(1), "event_type": "model_deployed", "system_id": "s",
         "payload": {"model_version": "v1"}},
        {"ts": lesson._ts(1), "event_type": "note", "system_id": "s", "decision_id": "D-Z",
         "payload": {}},
    ])
    got = [e["seq"] for e in lesson.reconstruct_decision(bare, "D-Z")]
    assert got == [1], (
        f"got {got} — with no model_version referenced there is no context to add; do not "
        "fall back to every deployment in the log"
    )


def test_reconstruct_returns_entries_and_handles_unknown():
    log = lesson.plain_log(STREAM)
    assert lesson.reconstruct_decision(log, "D-9999") == [], (
        "an unknown decision id returns an empty list, not the system-level events"
    )
    trace = lesson.reconstruct_decision(log, "D-1001")
    assert all(set(e) >= {"seq", "event", "entry_hash"} for e in trace), (
        "return the log ENTRIES, not the bare events — an auditor wants the sequence number "
        "and the hash alongside the payload"
    )
    chained = lesson.build_log(STREAM)
    assert all(e["entry_hash"] for e in lesson.reconstruct_decision(chained, "D-1001")), (
        "the entries handed back must be the log's own, hashes included, so the trace can be "
        "verified against the chain it came from"
    )


# --- exercise 5: completeness ----------------------------------------------------------

def test_completeness_shape_and_spans():
    log = lesson.plain_log(STREAM)
    report = lesson.completeness_report(lesson.reconstruct_decision(log, "D-1001"))
    assert set(report) == {"decision_id", "present", "missing", "coverage", "complete",
                           "first_seq", "last_seq"}, (
        f"keys were {sorted(report)} — return exactly the seven documented names"
    )
    want = _expected_trace_seqs("D-1001")
    assert (report["first_seq"], report["last_seq"]) == (min(want), max(want)), (
        f"D-1001's trace spans seqs {min(want)} to {max(want)}, got "
        f"{report['first_seq']} to {report['last_seq']}"
    )
    assert report["decision_id"] == "D-1001", (
        f"decision_id came back {report['decision_id']!r} — the trace opens with a deployment "
        "event that has no decision_id, so take the first entry that does have one"
    )
    empty = lesson.completeness_report([])
    assert empty["first_seq"] is None and empty["last_seq"] is None, (
        f"an empty trace has no span, got {empty['first_seq']!r}/{empty['last_seq']!r}"
    )
    assert empty["decision_id"] is None and empty["complete"] is False, (
        f"an empty trace names no decision and satisfies nothing, got {empty!r}"
    )


def test_completeness_envelope_and_payload():
    log = lesson.plain_log(STREAM)
    for decision_id in ("D-1001", "D-1002", "D-1003", "D-1004"):
        report = lesson.completeness_report(lesson.reconstruct_decision(log, decision_id))
        want_missing = _expected_missing(decision_id)
        assert report["missing"] == want_missing, (
            f"{decision_id}: missing={report['missing']}, expected {want_missing}"
        )
        want_present = sorted(set(lesson.REQUIRED_FIELDS) - set(want_missing))
        assert report["present"] == want_present, (
            f"{decision_id}: present={report['present']}, expected {want_present} — present "
            "and missing must be sorted and must partition the required fields"
        )
        assert report["complete"] is (not want_missing), (
            f"{decision_id}: complete={report['complete']} disagrees with its own missing list"
        )
    solo = lesson.plain_log([{"ts": lesson._ts(1), "event_type": "x", "system_id": "loan",
                              "decision_id": "D-Q", "payload": {"outcome": "approve"}}])
    report = lesson.completeness_report(solo, required=("system_id", "outcome"))
    assert report["present"] == ["outcome", "system_id"], (
        f"got {report['present']} — system_id sits in the event envelope and outcome in the "
        "payload; a checker that reads only one of the two can never see both"
    )


def test_completeness_rejects_empty_values():
    blank = lesson.plain_log([{"ts": lesson._ts(1), "event_type": "x", "decision_id": "D-X",
                               "system_id": "   ",
                               "payload": {"outcome": None, "verifier_ids": [],
                                           "use_start": "", "reference_database": {}}}])
    report = lesson.completeness_report(
        blank, required=("system_id", "outcome", "verifier_ids", "use_start",
                         "reference_database"))
    assert report["present"] == [], (
        f"got {report['present']} — a blank string, a None, an empty list and an empty dict "
        "are what a half-built pipeline writes when it has nothing to say; none of them is a "
        "recorded value"
    )
    assert report["coverage"] == 0.0 and report["complete"] is False, (
        f"nothing present means coverage 0.0 and complete False, got {report!r}"
    )
    falsy = lesson.plain_log([{"ts": lesson._ts(1), "event_type": "x", "decision_id": "D-Y",
                               "payload": {"outcome": 0, "verified": False}}])
    got = lesson.completeness_report(falsy, required=("outcome", "verified"))["present"]
    assert got == ["outcome", "verified"], (
        f"got {got} — an outcome of 0 and a verified flag of False are recorded answers. Test "
        "for None and emptiness, not for falsiness: `if not value` throws both away"
    )
    # An empty value in one entry must not stop a real one later in the trace counting.
    later = lesson.plain_log([
        {"ts": lesson._ts(2), "event_type": "a", "decision_id": "D-Z", "payload": {"outcome": ""}},
        {"ts": lesson._ts(1), "event_type": "b", "decision_id": "D-Z",
         "payload": {"outcome": "refer"}},
    ])
    assert lesson.completeness_report(later, required=("outcome",))["present"] == ["outcome"], (
        "the first entry carries an empty outcome and the second a real one — a field counts "
        "when SOME entry carries it, so do not stop at the first entry that mentions it"
    )


def test_completeness_coverage_and_vacuous_case():
    log = lesson.plain_log(STREAM)
    required = lesson.REQUIRED_FIELDS
    for decision_id in ("D-1001", "D-1002", "D-1003", "D-1004"):
        report = lesson.completeness_report(lesson.reconstruct_decision(log, decision_id))
        want = (len(required) - len(_expected_missing(decision_id))) / len(required)
        assert abs(report["coverage"] - want) < 1e-9, (
            f"{decision_id}: coverage={report['coverage']}, expected {want} — it is "
            "len(present) / len(required), a fraction, not a percentage and not a count"
        )
    vacuous = lesson.completeness_report([], required=())
    assert vacuous["coverage"] == 1.0, (
        f"coverage over an empty requirement list is 1.0, got {vacuous['coverage']!r} — "
        "nothing required is vacuously satisfied, and it must not be a ZeroDivisionError"
    )
    assert vacuous["complete"] is True, "with nothing required, the trace is complete"


RUBRIC = [
    (test_canonical_bytes_is_order_independent, 3,
     "json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')"),
    (test_canonical_bytes_round_trips, 2,
     "the canonical form must still be JSON: json.loads(canonical_bytes(x)) == x"),
    (test_append_event_shape_and_links, 4,
     "seq is len(log); prev_hash is GENESIS_HASH or the previous entry_hash; append in place "
     "and return the entry"),
    (test_append_event_digest_covers_event_and_history, 4,
     "hash canonical_bytes({'seq': ..., 'prev_hash': ..., 'event': ...}) — all three, so that "
     "an edit, a reorder and a rewrite each move the digest"),
    (test_verify_chain_accepts_a_clean_log, 2,
     "ChainVerdict(True, None, '') for a log you just built, and for an empty one"),
    (test_verify_chain_catches_an_edit, 3,
     "recompute each entry's digest from its own contents; never trust the stored value"),
    (test_verify_chain_catches_a_deletion, 3,
     "check entry['seq'] == i first: a gap is a deletion. Dropping the LAST entry leaves no "
     "gap, so that case must still verify"),
    (test_verify_chain_catches_a_relink, 2,
     "compare prev_hash against the IMMEDIATELY preceding entry's entry_hash"),
    (test_verify_chain_anchor, 4,
     "after every entry passes, compare the last entry_hash against expected_head and report "
     "first_bad_seq=None — a rewritten tail is internally perfect"),
    (test_retention_partition_and_boundary, 4,
     "age > minimum_days is expired, age <= minimum_days is retained; decide per entry"),
    (test_retention_scans_for_extremes, 3,
     "oldest_ts and newest_ts are min() and max() over the log, not log[0] and log[-1]"),
    (test_retention_finds_gaps, 3,
     "missing_seq is range(0, max(seq) + 1) minus the seqs present, and [] for an empty log"),
    (test_reconstruct_selects_and_orders, 4,
     "select on event['decision_id'], then sort by 'seq' — never by 'ts'"),
    (test_reconstruct_pulls_in_deployment_context, 3,
     "second pass: add the model_deployed entries whose model_version the decision names"),
    (test_reconstruct_returns_entries_and_handles_unknown, 2,
     "hand back the log entries themselves, and [] for a decision id that is not there"),
    (test_completeness_shape_and_spans, 3,
     "first_seq/last_seq are min/max of the trace's seqs; decision_id comes from the first "
     "entry that has one"),
    (test_completeness_envelope_and_payload, 3,
     "look in entry['event'] AND entry['event']['payload'] — system_id is never in a payload"),
    (test_completeness_rejects_empty_values, 3,
     "reject None, blank strings and empty containers; 0 and False are recorded values"),
    (test_completeness_coverage_and_vacuous_case, 3,
     "coverage is len(present)/len(required), and 1.0 when nothing is required"),
]
