"""Autograder rubric for P01-L09. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Every expected value here is re-derived by code in THIS file, in a different shape from
  the notebook's own: `_ref_results` walks `lesson.CHECKS` directly, `_ref_findings` builds
  the finding rows with a comprehension and a `sorted(key=...)`, and `_ref_delta` diffs two
  indices with set arithmetic. Nothing is a literal copied out of the notebook's output, and
  no test calls the student's function to decide what the student's function should return.

* Exercises 2 and 7 are graded against `REF_FINDINGS`, this file's own findings, and not
  against the student's `inspect_pack`. A broken exercise 2 must not cost a student the
  repair; that is what partial credit means here.

* Exercises 6 and 8 are graded almost entirely on HAND-BUILT inputs — four findings, a
  synthetic delta, a synthetic suite change — because the lesson's own pack has exactly one
  instance of each signal and an implementation can pass it by hard-coding. The verdict
  ladder, the reason order, the "a mutation inside the finding is not silencing" rule and
  the report's verdict precedence are all of that kind.

* `test_plan_repairs_on_a_variant_pack` is what stops a plan that hard-codes
  "export-2026-05-14", `"accuracy"` as the metric the document section is about, or the
  four hundred stamped verification ids. The variant purges a different window, exports it
  on a different night, makes thirty more reviews implausible, adds a second substantial
  modification and points the document section at a different metric. A plan that reads the
  findings passes it; a plan that remembers this notebook does not.

* `test_audit_ignores_mutations_inside_the_finding` exists because the honest repair to this
  pack mutates 805 evidence keys. An audit that treats every mutation as silencing fails
  every honest repair, which is the same defect as an audit that treats none of them as
  silencing, arriving from the other side.
"""
import copy
import importlib.util
import os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

PACK = lesson.PACK
SUITE = lesson.SUITE


# --------------------------------------------------------------- this file's own references
def _ref_results(pack: dict, suite: dict) -> dict:
    """Walk lesson.CHECKS directly: the rubric's own suite run, not the student's."""
    out = {}
    for cid in sorted(suite):
        row = suite[cid]
        if not row.get("enabled"):
            out[cid] = {"status": "not_run", "reason_code": "not_verified",
                        "affected": None, "population": None, "detail": {}}
        elif cid not in lesson.CHECKS:
            out[cid] = {"status": "not_run", "reason_code": "no_such_check",
                        "affected": None, "population": None, "detail": {}}
        else:
            answer = lesson.CHECKS[cid](pack, row.get("params", {}))
            out[cid] = {"status": "green" if answer["green"] else "red",
                        "reason_code": answer["reason_code"],
                        "affected": answer["affected"],
                        "population": answer["population"], "detail": answer["detail"]}
    return out


def _ref_run(pack: dict, suite: dict) -> dict:
    results = _ref_results(pack, suite)
    buckets = {kind: sorted(c for c, r in results.items() if r["status"] == kind)
               for kind in ("green", "red", "not_run")}
    return {"results": results, **buckets,
            "verdict": "pass" if not (buckets["red"] or buckets["not_run"]) else "fail"}


def _ref_findings(run: dict) -> list:
    rows = []
    for cid, row in run["results"].items():
        if row["status"] == "green" or cid not in lesson.CHECK_META:
            continue
        meta = lesson.CHECK_META[cid]
        if row["status"] == "not_run":
            share = 1.0
        elif row["affected"] is None or not row["population"]:
            share = 0.0
        else:
            share = row["affected"] / row["population"]
        rows.append({"finding_id": f"{cid}:{row['reason_code']}", "check": cid,
                     "evidence_id": meta["evidence_id"], "module": meta["module"],
                     "basis": meta["basis"], "reason_code": row["reason_code"],
                     "affected": row["affected"], "population": row["population"],
                     "share": share,
                     "evidence_keys": sorted(row["detail"].get("keys", [])),
                     "detail": row["detail"]})
    rows.sort(key=lambda row: (-row["share"], row["finding_id"]))
    return rows


def _ref_delta(before: dict, after: dict) -> dict:
    families = {}
    for family in set(before) | set(after):
        old, new = before.get(family, {}), after.get(family, {})
        families[family] = {
            "removed": sorted(set(old) - set(new)),
            "added": sorted(set(new) - set(old)),
            "mutated": sorted(key for key in old if key in new and old[key] != new[key])}
    return families


RUN = _ref_run(PACK, SUITE)
REF_FINDINGS = _ref_findings(RUN)
BY_CHECK = {f["check"]: f for f in REF_FINDINGS}


def _variant_pack() -> dict:
    """A second pack with the same six defects in different places and different sizes."""
    pack = lesson.build_pack()
    restored = [copy.deepcopy(e)
                for e in pack["cold_store"]["export-2026-05-14"]["entries"]]
    whole = sorted(pack["log"] + restored, key=lambda e: e["seq"])
    low, high = 301, 312
    pack["log"] = [e for e in whole if not (low <= e["seq"] <= high)]
    pack["cold_store"]["export-2026-06-02"] = {
        "segment_id": "export-2026-06-02", "covers": [low, high],
        "entries": [copy.deepcopy(e) for e in whole if low <= e["seq"] <= high]}
    for verification in pack["oversight"]["verifications"][900:930]:
        verification["seconds"] = 3.0
    for change in pack["conformity"]["changes"]:
        if change["change_id"] == "C-0033":
            change["modifies_intended_purpose"] = True
    pack["sources"]["metrics"]["payload"]["primary_metric"] = "recall"
    for section in pack["annex_iv"]["sections"]:
        if section["number"] == "4":
            section["source_digest"] = lesson.digest_of(
                pack["sources"]["metrics"]["payload"])
    return pack


# ------------------------------------------------------------- hand-built audit fixtures
def _finding(check: str, code: str = "defect", keys=()) -> dict:
    return {"finding_id": f"{check}:{code}", "check": check, "evidence_id": "evidence",
            "module": "P01-LXX", "basis": "basis", "reason_code": code, "affected": 1,
            "population": 4, "share": 0.25, "evidence_keys": sorted(keys), "detail": {}}


def _run_of(**statuses) -> dict:
    results = {}
    for check, value in statuses.items():
        status, code = value if isinstance(value, tuple) else (value, "ok")
        results[check] = {"status": status, "reason_code": code, "affected": None,
                          "population": None, "detail": {}}
    buckets = {kind: sorted(c for c, r in results.items() if r["status"] == kind)
               for kind in ("green", "red", "not_run")}
    return {"results": results, **buckets,
            "verdict": "pass" if not (buckets["red"] or buckets["not_run"]) else "fail"}


def _delta_of(**families) -> dict:
    built = {}
    for family, spec in families.items():
        built[family] = {"removed": list(spec.get("removed", [])),
                         "added": list(spec.get("added", [])),
                         "mutated": list(spec.get("mutated", [])),
                         "n_before": 10, "n_after": 10}
    totals = {kind: sum(len(f[kind]) for f in built.values())
              for kind in ("removed", "added", "mutated")}
    return {"families": built, "totals": totals, "conserved": totals["removed"] == 0}


def _change_of(**kw) -> dict:
    base = {"disabled": [], "enabled": [], "removed_checks": [], "added_checks": [],
            "loosened": [], "tightened": [], "unknown": [], "unchanged": True}
    base.update(kw)
    return base


# ------------------------------------------------------------------------- exercise 1
def test_run_suite_status_per_check():
    got = lesson.run_suite(PACK, SUITE)
    assert set(got) == {"results", "green", "red", "not_run", "verdict"}, (
        f"run_suite returned {sorted(got)}; the five documented keys are required")
    want = _ref_results(PACK, SUITE)
    assert got["results"] == want, (
        "the per-check rows do not match a direct call to lesson.CHECKS — carry "
        "reason_code, affected, population and detail through unchanged")
    assert got["red"] == sorted(want) and got["green"] == [] and got["not_run"] == [], (
        f"buckets came back green={got['green']} red={got['red']} "
        f"not_run={got['not_run']}; all six checks fail on the shipped pack")


def test_run_suite_never_calls_or_passes_a_disabled_check():
    def explode(pack, params):
        raise AssertionError("run_suite called a check the suite had switched off")

    suite = copy.deepcopy(SUITE)
    suite["oversight_plausibility"]["enabled"] = False
    checks = dict(lesson.CHECKS)
    checks["oversight_plausibility"] = explode
    got = lesson.run_suite(PACK, suite, checks)
    row = got["results"]["oversight_plausibility"]
    assert row["status"] == "not_run", (
        f"a disabled check came back {row['status']!r}. It has three statuses for this "
        "reason: nobody ran it, so nobody can say it passed")
    assert row["reason_code"] == "not_verified", (
        f"reason_code came back {row['reason_code']!r}, expected 'not_verified'")
    assert row["affected"] is None and row["population"] is None and row["detail"] == {}, (
        "a check that did not run has no counts: 0 affected reads as 'we looked and found "
        "nothing', which is the opposite of what happened")
    assert got["not_run"] == ["oversight_plausibility"]
    assert "oversight_plausibility" not in got["green"]


def test_run_suite_verdict_and_unknown_checks():
    suite = copy.deepcopy(SUITE)
    suite["a_check_nobody_wrote"] = {"enabled": True, "params": {}}
    got = lesson.run_suite(PACK, suite)
    row = got["results"]["a_check_nobody_wrote"]
    assert row["status"] == "not_run" and row["reason_code"] == "no_such_check", (
        f"a suite entry with no implementation came back {row['status']}/"
        f"{row['reason_code']}; it must be 'not_run'/'no_such_check', not a crash and not "
        "a pass")
    clean, _ = lesson.apply_repairs(PACK, SUITE, [])
    empty = {cid: {**copy.deepcopy(row), "enabled": False} for cid, row in SUITE.items()}
    assert lesson.run_suite(clean, empty)["verdict"] == "fail", (
        "a suite with every check switched off returned 'pass'. The verdict is 'pass' only "
        "when nothing is red AND nothing is unverified")
    one = {"log_integrity": copy.deepcopy(SUITE["log_integrity"])}
    repaired, _ = lesson.apply_repairs(
        PACK, SUITE, [{"op": "restore_log_segment", "segment": "export-2026-05-14"}])
    assert lesson.run_suite(repaired, one)["verdict"] == "pass", (
        "a suite whose only check is green and ran must come back 'pass'")


# ------------------------------------------------------------------------- exercise 2
def test_findings_cover_red_and_not_run():
    got = lesson.inspect_pack(RUN)
    assert [f["finding_id"] for f in got] == [f["finding_id"] for f in REF_FINDINGS], (
        f"findings came back {[f['finding_id'] for f in got]}, expected "
        f"{[f['finding_id'] for f in REF_FINDINGS]}")
    suite = copy.deepcopy(SUITE)
    suite["document_sourcing"]["enabled"] = False
    quiet = lesson.inspect_pack(_ref_run(PACK, suite))
    assert len(quiet) == 6, (
        f"{len(quiet)} findings when one check is switched off, expected 6 — a check "
        "nobody ran is a finding, not the absence of one")
    codes = {f["check"]: f["reason_code"] for f in quiet}
    assert codes["document_sourcing"] == "not_verified"


def test_findings_carry_meta_and_evidence_keys():
    got = {f["check"]: f for f in lesson.inspect_pack(RUN)}
    for check, finding in got.items():
        meta = lesson.CHECK_META[check]
        assert set(finding) == {"finding_id", "check", "evidence_id", "module", "basis",
                                "reason_code", "affected", "population", "share",
                                "evidence_keys", "detail"}, (
            f"{check} finding has keys {sorted(finding)}; return exactly the eleven "
            "documented names")
        assert (finding["evidence_id"], finding["module"], finding["basis"]) == (
            meta["evidence_id"], meta["module"], meta["basis"]), (
            f"{check}: evidence_id, module and basis come straight from CHECK_META")
        assert finding["evidence_keys"] == BY_CHECK[check]["evidence_keys"], (
            f"{check}: evidence_keys must be sorted(detail['keys']) — exercise 6 asks "
            "whether the repair stayed inside them, and an empty list there passes "
            "everything")
    assert len(got["oversight_plausibility"]["evidence_keys"]) == 800, (
        "the oversight finding names 400 verifications AND the 400 decisions they support")


def test_findings_share_is_a_ratio_and_orders_the_list():
    got = lesson.inspect_pack(RUN)
    for finding, want in zip(got, REF_FINDINGS):
        assert abs(finding["share"] - want["share"]) < 1e-12, (
            f"{finding['finding_id']}: share came back {finding['share']}, expected "
            f"{want['share']} = affected / population")
    shares = [f["share"] for f in got]
    assert shares == sorted(shares, reverse=True), (
        f"findings came back in share order {shares}; rank by the share of the evidence "
        "each defect touches, descending — 14 purged entries out of 441 is a smaller hole "
        "than 1 undeclarable metric out of 3, and sorting by the raw count says otherwise")
    assert got[0]["check"] == "declared_accuracy" and got[-1]["check"] == "log_integrity"
    # The sort key and the zero-population guard, isolated from the order the results
    # happen to arrive in: two checks tie on share, and the third counts nothing at all.
    def _row(code, affected, population):
        return {"status": "red", "reason_code": code, "affected": affected,
                "population": population, "detail": {}}

    tied = {"results": {"oversight_plausibility": _row("defect", 1, 4),
                        "declaration_currency": _row("defect", 1, 4),
                        "log_integrity": _row("defect", 0, 0)},
            "green": [], "not_run": [],
            "red": ["declaration_currency", "log_integrity", "oversight_plausibility"],
            "verdict": "fail"}
    order = [f["finding_id"] for f in lesson.inspect_pack(tied)]
    assert order == ["declaration_currency:defect", "oversight_plausibility:defect",
                     "log_integrity:defect"], (
        f"findings came back {order}. Two checks tie on share, and the tie is broken by "
        "finding_id ASCENDING — not by the order the results arrived in, which is whatever "
        "your own run_suite happened to build")
    assert lesson.inspect_pack(tied)[-1]["share"] == 0.0, (
        "a red check that counted nothing has share 0.0, not a ZeroDivisionError: a "
        "population of 0 is the one case where affected / population cannot be taken")


def test_a_check_that_did_not_run_is_the_worst_finding():
    suite = copy.deepcopy(SUITE)
    suite["log_integrity"]["enabled"] = False
    got = lesson.inspect_pack(_ref_run(PACK, suite))
    assert got[0]["check"] == "log_integrity", (
        f"with the log check switched off the worst finding was {got[0]['check']}; an "
        "unverified check leaves ALL of its evidence in doubt and sorts first")
    assert got[0]["share"] == 1.0, (
        f"share came back {got[0]['share']} for a check nobody ran, expected 1.0")
    assert got[0]["affected"] is None and got[0]["population"] is None, (
        "affected and population pass through unchanged; only 'share' is derived")
    assert got[0]["evidence_keys"] == []


# ------------------------------------------------------------------------- exercise 3
def test_delta_partitions_every_key():
    before = {"log": {"a": "1", "b": "2", "c": "3"}, "oversight": {"x": "9"}}
    after = {"log": {"b": "2", "c": "4", "d": "5"}, "oversight": {"x": "9"}}
    got = lesson.evidence_delta(before, after)
    assert set(got) == {"families", "totals", "conserved"}, (
        f"evidence_delta returned {sorted(got)}; three top-level keys")
    log = got["families"]["log"]
    assert log["removed"] == ["a"], f"removed came back {log['removed']}, expected ['a']"
    assert log["added"] == ["d"], f"added came back {log['added']}, expected ['d']"
    assert log["mutated"] == ["c"], (
        f"mutated came back {log['mutated']}, expected ['c'] — a key in both indices whose "
        "digest changed. 'b' is identical and must not appear anywhere")
    assert (log["n_before"], log["n_after"]) == (3, 3), (
        "n_before and n_after are the family's sizes, not the size of the difference")
    assert got["families"]["oversight"] == {"removed": [], "added": [], "mutated": [],
                                            "n_before": 1, "n_after": 1}


def test_delta_handles_a_family_missing_from_either_side():
    before = {"log": {"a": "1"}, "annex_iv": {"s": "2"}}
    after = {"log": {"a": "1"}, "accuracy": {"d": "3"}}
    got = lesson.evidence_delta(before, after)
    assert set(got["families"]) == {"log", "annex_iv", "accuracy"}, (
        f"families came back {sorted(got['families'])}; walk the UNION, so a repair that "
        "drops a whole family is reported rather than raising KeyError")
    assert got["families"]["annex_iv"]["removed"] == ["s"]
    assert got["families"]["annex_iv"]["n_after"] == 0
    assert got["families"]["accuracy"]["added"] == ["d"]
    assert got["conserved"] is False


def test_delta_totals_and_conserved_on_the_real_pack():
    deleted, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "delete_records", "family": "conformity_changes", "keys": ["C-0031"]}])
    got = lesson.evidence_delta(lesson.evidence_index(PACK),
                                lesson.evidence_index(deleted))
    want = _ref_delta(lesson.evidence_index(PACK), lesson.evidence_index(deleted))
    assert got["totals"] == {"removed": 1, "added": 0, "mutated": 0}, (
        f"totals came back {got['totals']}; deleting one change removes one key and "
        "changes nothing else")
    assert got["conserved"] is False
    assert got["families"]["conformity"]["removed"] == want["conformity"]["removed"]
    same = lesson.evidence_delta(lesson.evidence_index(PACK), lesson.evidence_index(PACK))
    assert same["conserved"] is True and same["totals"]["mutated"] == 0, (
        "a pack against itself reported a difference; the index is deterministic")
    rewritten, _ = lesson.apply_repairs(PACK, SUITE, [{"op": "rechain_log"}])
    churned = lesson.evidence_delta(lesson.evidence_index(PACK),
                                    lesson.evidence_index(rewritten))
    assert churned["totals"]["mutated"] == 246 and churned["totals"]["removed"] == 0, (
        f"re-chaining the log came back {churned['totals']}; the index keys log entries on "
        "their EVENT digest, so a renumbering mutates the entries after the hole and "
        "removes nothing")
    assert churned["conserved"] is True, (
        "'conserved' is documented as being about removals only, and this repair removed "
        "nothing while rewriting 246 entries. An implementation that folds mutations into "
        "it calls every honest recomputation an evidence loss")
    restored, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "restore_log_segment", "segment": "export-2026-05-14"}])
    grown = lesson.evidence_delta(lesson.evidence_index(PACK),
                                  lesson.evidence_index(restored))
    assert grown["totals"] == {"removed": 0, "added": 14, "mutated": 0}, (
        f"restoring the cold-store segment came back {grown['totals']}; the fourteen purged "
        "entries return as ADDED keys, under event digests the index has never seen")
    assert grown["conserved"] is True, (
        "a repair that puts evidence BACK came back not conserved. The flag is about "
        "removals and nothing else — folding additions into it reports the honest repair "
        "to this pack, which adds twenty keys and removes none, as an evidence loss")


# ------------------------------------------------------------------------- exercise 4
def test_suite_delta_direction_decides_loosened():
    before = {"c": {"enabled": True, "params": {"min_seconds": 10, "k": 3.0}}}
    after = {"c": {"enabled": True, "params": {"min_seconds": 2, "k": 9.0}}}
    got = lesson.suite_delta(before, after)
    assert [r["param"] for r in got["loosened"]] == ["k", "min_seconds"], (
        f"loosened came back {got['loosened']}; k is weaker HIGHER and min_seconds is "
        "weaker LOWER, and rows are sorted by (check, param)")
    assert got["loosened"][0] == {"check": "c", "param": "k", "before": 3.0, "after": 9.0}
    assert got["tightened"] == [] and got["unchanged"] is False


def test_suite_delta_tightening_is_not_silencing():
    before = {"c": {"enabled": True, "params": {"min_seconds": 10, "k": 3.0}}}
    after = {"c": {"enabled": True, "params": {"min_seconds": 45, "k": 2.0}}}
    got = lesson.suite_delta(before, after)
    assert got["loosened"] == [], (
        f"raising min_seconds and lowering k came back as {got['loosened']}; both make the "
        "check STRICTER, and an audit that pages on a tightening is muted within a month")
    assert [r["param"] for r in got["tightened"]] == ["k", "min_seconds"]
    assert got["unchanged"] is True, (
        "'unchanged' is documented as being about weakening: a suite that was only "
        "tightened comes back unchanged")


def test_suite_delta_disabled_removed_and_dropped_parameters():
    before = {"a": {"enabled": True, "params": {"k": 3.0}},
              "b": {"enabled": True, "params": {}},
              "c": {"enabled": False, "params": {}}}
    after = {"a": {"enabled": False, "params": {}},
             "c": {"enabled": True, "params": {}},
             "d": {"enabled": True, "params": {}}}
    got = lesson.suite_delta(before, after)
    assert got["disabled"] == ["a"], f"disabled came back {got['disabled']}"
    assert got["enabled"] == ["c"], f"enabled came back {got['enabled']}"
    assert got["removed_checks"] == ["b"], (
        f"removed_checks came back {got['removed_checks']}; a check deleted from the suite "
        "is a check switched off with the evidence of switching it off removed")
    assert got["added_checks"] == ["d"]
    assert got["loosened"] == [{"check": "a", "param": "k", "before": 3.0, "after": None}], (
        f"a parameter that vanished came back as {got['loosened']}; a limit you stopped "
        "applying is the widest possible widening, not a no-op")
    assert got["unchanged"] is False


def test_suite_delta_never_guesses_an_unknown_direction():
    before = {"c": {"enabled": True, "params": {"fudge": 1, "k": 3.0}}}
    after = {"c": {"enabled": True, "params": {"fudge": 99, "k": 3.0}}}
    got = lesson.suite_delta(before, after)
    assert got["unknown"] == ["c.fudge"], (
        f"unknown came back {got['unknown']}; a parameter with no entry in PARAM_DIRECTION "
        "is reported, not guessed at")
    assert got["loosened"] == [] and got["tightened"] == []
    assert got["unchanged"] is False, "an unknown change leaves the suite not-unchanged"
    added = {"c": {"enabled": True, "params": {"k": 3.0, "brand_new": 5}}}
    later = lesson.suite_delta(before={"c": {"enabled": True, "params": {"k": 3.0}}},
                               after=added)
    assert later["loosened"] == [] and later["unknown"] == [], (
        "a parameter only the AFTER suite has is a new rule, not a loosening and not an "
        "unknown change")


# ------------------------------------------------------------------------- exercise 5
def test_unsupported_on_the_shipped_pack():
    got = lesson.unsupported_values(PACK)
    claims = {row["claim"]: row["reason"] for row in got}
    assert [row["claim"] for row in got] == sorted(claims), "sort the rows by claim"
    assert set(claims) == {"accuracy.declarations.accuracy", "monitoring.limits"}, (
        f"the shipped pack came back with {sorted(claims)}; the accuracy declaration has "
        "no interval to reproduce and the control limits were refitted over the excursion "
        "they are supposed to catch. The other two declarations DO reproduce and must not "
        "be listed")
    assert claims["accuracy.declarations.accuracy"] == "no_interval"
    assert claims["monitoring.limits"] == "not_from_the_sealed_baseline"
    assert all(set(row) == {"claim", "reason", "expected", "found"} for row in got), (
        f"rows came back with keys {sorted(got[0])}; four names, exactly")


def test_unsupported_catches_an_interval_that_was_typed_in():
    real = PACK["accuracy"]["declarations"]["recall"]
    faked = dict(real)
    faked["lo"] = round(real["lo"] - 0.02, 6)
    pack, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "set_field", "path": "accuracy.declarations.recall", "value": faked}])
    claims = {row["claim"]: row["reason"] for row in lesson.unsupported_values(pack)}
    assert claims.get("accuracy.declarations.recall") == "does_not_reproduce", (
        "a declared interval widened by hand passed. Recompute it with bootstrap_ci over "
        "pack['accuracy']['evaluation'], using the declaration's own metric, n_boot, level "
        "and seed, and compare point, lo AND hi")
    no_spec = dict(real)
    no_spec.pop("seed")
    pack, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "set_field", "path": "accuracy.declarations.recall", "value": no_spec}])
    claims = {row["claim"]: row["reason"] for row in lesson.unsupported_values(pack)}
    assert claims.get("accuracy.declarations.recall") == "no_recompute_spec", (
        "a declaration with no seed cannot be recomputed by anybody, which is a different "
        "failure from one that recomputes to a different number")


def test_unsupported_checks_the_document_against_its_source():
    hand, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "set_field", "path": "sources.metrics.payload.primary_metric_value",
         "value": 0.9}])
    claims = {row["claim"]: row["reason"] for row in lesson.unsupported_values(hand)}
    assert claims.get("annex_iv.sections.4") == "does_not_match_its_source", (
        "the artefact now says 0.9 and the emitted section still shows the old value; a "
        "section that disagrees with the source it names is not sourced")
    assert claims.get("sources.metrics.primary_metric_value") == \
        "not_traceable_to_a_measurement", (
        "the number in the artefact carries no digest linking it to the declaration of the "
        "metric the artefact says it is about, so it came from somebody's head. Check "
        "f'{field}_source' against digest_of(declarations[payload['primary_metric']])")
    wrong_metric, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "recompute_declaration", "metric": "accuracy"},
        {"op": "resource_section", "section": "4", "artefact": "metrics",
         "field": "primary_metric_value", "from_declaration": "false_positive_rate"}])
    claims = {row["claim"]: row["reason"] for row in lesson.unsupported_values(wrong_metric)}
    assert claims.get("sources.metrics.primary_metric_value") == \
        "not_traceable_to_a_measurement", (
        "the metrics artefact says its primary metric is 'accuracy' and the value in it is "
        "stamped with the false-positive-rate declaration. That is a real number in the "
        "wrong row, and it passes every test that only asks whether the stamp matches SOME "
        "declaration")
    stale, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "set_field", "path": "annex_iv.sections.3.source_digest", "value": "0" * 64}])
    claims = {row["claim"]: row["reason"] for row in lesson.unsupported_values(stale)}
    assert "annex_iv.sections.3" not in claims, (
        "section 3 carries no 'value', so there is no claim in it to reproduce; only "
        "sections that state a number are checked")


def test_unsupported_checks_the_limits_and_the_ledger():
    wide, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "set_field", "path": "monitoring.limits.ucl", "value": 0.99}])
    claims = {row["claim"]: row["reason"] for row in lesson.unsupported_values(wide)}
    assert claims.get("monitoring.limits") == "not_from_the_sealed_baseline"
    edited, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "edit_ledger_entry", "seq": 1, "field": "errors", "value": 132}])
    claims = {row["claim"]: row["reason"] for row in lesson.unsupported_values(edited)}
    assert claims.get("ledger") == "ledger_chain_broken", (
        "a sealed ledger payload was edited and nothing noticed. Recompute each entry's "
        "digest over seq, kind, payload and prev, the way ledger_append computed it")
    honest, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "refit_limits_to_sealed_baseline"}])
    claims = {row["claim"] for row in lesson.unsupported_values(honest)}
    assert "monitoring.limits" not in claims, (
        "limits refitted from the sealed baseline entry, at the k they record, must come "
        "back supported")
    typed_centre, _ = lesson.apply_repairs(PACK, SUITE, [
        {"op": "refit_limits_to_sealed_baseline"},
        {"op": "set_field", "path": "monitoring.limits.centre", "value": 0.5}])
    claims = {row["claim"]: row["reason"] for row in
              lesson.unsupported_values(typed_centre)}
    assert claims.get("monitoring.limits") == "not_from_the_sealed_baseline", (
        "the limits were refitted and then the centre was typed over by hand, which leaves "
        "ucl exactly where the sealed baseline puts it. Compare all four of centre, sigma, "
        "ucl and lcl — a comparison on ucl alone waves this through")


# ------------------------------------------------------------------------- exercise 6
def test_audit_a_check_that_did_not_run_is_silenced():
    findings = [_finding("a"), _finding("b")]
    before = _run_of(a=("red", "defect"), b=("red", "defect"))
    after = _run_of(a="not_run", b=("green", "ok"))
    got = lesson.silencing_audit(findings, before, after, _delta_of(), _change_of(),
                                 [])
    assert got["verdicts"]["a:defect"]["verdict"] == "silenced", (
        "a finding whose check did not run in the after-run came back "
        f"{got['verdicts']['a:defect']['verdict']!r}; nobody looked, so nobody repaired it")
    assert got["verdicts"]["a:defect"]["reasons"] == ["check_not_run"]
    gone = _run_of(b=("green", "ok"))
    removed = lesson.silencing_audit(findings, before, gone, _delta_of(),
                                     _change_of(removed_checks=["a"], unchanged=False), [])
    assert removed["verdicts"]["a:defect"]["reasons"] == ["check_not_run",
                                                          "check_removed_from_suite"], (
        f"a check deleted from the suite came back {removed['verdicts']['a:defect']}; it "
        "is missing from the results AND named in removed_checks")


def test_audit_a_red_check_is_unfixed_not_silenced():
    findings = [_finding("a", "purged_window")]
    before = _run_of(a=("red", "purged_window"))
    after = _run_of(a=("red", "chain_broken"))
    got = lesson.silencing_audit(findings, before, after,
                                 _delta_of(log={"removed": ["x", "y"]}),
                                 _change_of(disabled=["a"], unchanged=False), [])
    row = got["verdicts"]["a:purged_window"]
    assert row["verdict"] == "unfixed", (
        f"a still-red check came back {row['verdict']!r}; the silencing signals only "
        "matter for a check that is GREEN, and a red check is honestly unfixed whatever "
        "else the repair did")
    assert row["reasons"] == ["still_red:chain_broken"], (
        f"reasons came back {row['reasons']}; report the reason code the check gives NOW, "
        "not the one in the finding")
    assert got["counts"] == {"repaired": 0, "silenced": 0, "unfixed": 1}


def test_audit_removed_evidence_silences_a_green_check():
    findings = [_finding("oversight_plausibility", "implausible_review",
                         keys=["verification:V-1"])]
    before = _run_of(oversight_plausibility=("red", "implausible_review"))
    after = _run_of(oversight_plausibility=("green", "plausible"))
    delta = _delta_of(oversight={"removed": ["verification:V-1", "decision:D-1"]},
                      log={"removed": ["event:zzz"]})
    got = lesson.silencing_audit(findings, before, after, delta, _change_of(), [])
    row = got["verdicts"]["oversight_plausibility:implausible_review"]
    assert row["verdict"] == "silenced" and row["reasons"] == [
        "evidence_removed:oversight:2"], (
        f"reasons came back {row['reasons']}; count the removals in the families the CHECK "
        "reads (CHECK_FAMILIES), name the family and the count, and do not let another "
        "family's removals leak in")
    assert got["honest"] is False


def test_audit_a_loosened_parameter_silences_only_its_own_check():
    findings = [_finding("oversight_plausibility", "implausible_review"),
                _finding("monitoring_limits", "limits_not_from_sealed_baseline")]
    before = _run_of(oversight_plausibility=("red", "implausible_review"),
                     monitoring_limits=("red", "limits_not_from_sealed_baseline"))
    after = _run_of(oversight_plausibility=("green", "plausible"),
                    monitoring_limits=("green", "in_control"))
    change = _change_of(unchanged=False, loosened=[
        {"check": "oversight_plausibility", "param": "max_reviews_per_hour",
         "before": 60, "after": 1000},
        {"check": "oversight_plausibility", "param": "min_seconds", "before": 10,
         "after": 1}])
    got = lesson.silencing_audit(findings, before, after, _delta_of(), change, [])
    assert got["verdicts"]["oversight_plausibility:implausible_review"]["reasons"] == [
        "param_loosened:max_reviews_per_hour", "param_loosened:min_seconds"], (
        "each loosened parameter of THIS check contributes one reason, in the order the "
        "rows appear")
    assert got["verdicts"]["monitoring_limits:limits_not_from_sealed_baseline"][
        "verdict"] == "repaired", (
        "widening a parameter of one check must not silence a different check's finding")
    assert got["counts"] == {"repaired": 1, "silenced": 1, "unfixed": 0}


def test_audit_ignores_mutations_inside_the_finding():
    inside = _finding("declared_accuracy", "no_interval",
                      keys=["declaration:accuracy"])
    before = _run_of(declared_accuracy=("red", "no_interval"))
    after = _run_of(declared_accuracy=("green", "declared"))
    delta = _delta_of(accuracy={"mutated": ["declaration:accuracy"],
                                "added": ["declaration:brand_new"]})
    got = lesson.silencing_audit([inside], before, after, delta, _change_of(), [])
    row = got["verdicts"]["declared_accuracy:no_interval"]
    assert row["verdict"] == "repaired", (
        f"a repair that changed exactly the evidence the finding named came back "
        f"{row['verdict']!r} with reasons {row['reasons']}. Recomputing the declaration "
        "the finding is about is the repair; flagging it fails every honest fix")
    outside = _finding("declared_accuracy", "no_interval", keys=[])
    row = lesson.silencing_audit([outside], before, after, delta, _change_of(),
                                 [])["verdicts"]["declared_accuracy:no_interval"]
    assert row["reasons"] == ["evidence_changed_outside_the_finding:accuracy:1"], (
        f"reasons came back {row['reasons']}; a mutated key the finding never named is "
        "one signal, and an ADDED key is never one — added evidence is what a repair does")


def test_audit_maps_unsupported_claims_counts_and_collateral():
    findings = [_finding("declared_accuracy", "no_interval",
                         keys=["declaration:accuracy"]),
                _finding("monitoring_limits", "limits_not_from_sealed_baseline",
                         keys=["limits"])]
    before = _run_of(declared_accuracy=("red", "no_interval"),
                     monitoring_limits=("red", "limits_not_from_sealed_baseline"),
                     log_integrity=("green", "intact"))
    after = _run_of(declared_accuracy=("green", "declared"),
                    monitoring_limits=("green", "in_control"),
                    log_integrity=("red", "chain_broken"))
    unsupported = [{"claim": "accuracy.declarations.accuracy",
                    "reason": "does_not_reproduce", "expected": None, "found": None},
                   {"claim": "ledger", "reason": "ledger_chain_broken", "expected": None,
                    "found": 2}]
    got = lesson.silencing_audit(findings, before, after, _delta_of(), _change_of(),
                                 unsupported)
    assert got["verdicts"]["declared_accuracy:no_interval"]["reasons"] == [
        "unsupported:accuracy.declarations.accuracy"], (
        "a claim that does not reproduce silences the check it belongs to — use "
        "checks_for_claim(), which is given to you")
    assert got["verdicts"]["monitoring_limits:limits_not_from_sealed_baseline"][
        "reasons"] == ["unsupported:ledger"], (
        "the ledger claim belongs to two checks, and monitoring_limits is one of them")
    assert got["counts"] == {"repaired": 0, "silenced": 2, "unfixed": 0}
    assert got["collateral"] == ["log_integrity"], (
        f"collateral came back {got['collateral']}; log_integrity was green before the "
        "repair and is not green after it, and nobody had complained about it")
    assert got["honest"] is False
    # All four kinds of reason at once, on one check and one family: the order is the
    # docstring's four numbered steps, not whatever order they were convenient to collect.
    ladder = lesson.silencing_audit(
        [_finding("declared_accuracy", "no_interval", keys=["declaration:accuracy"])],
        _run_of(declared_accuracy=("red", "no_interval")),
        _run_of(declared_accuracy=("green", "declared")),
        _delta_of(accuracy={"removed": ["declaration:recall"],
                            "mutated": ["declaration:accuracy", "evaluation"]}),
        _change_of(unchanged=False, loosened=[
            {"check": "declared_accuracy", "param": "min_interval_level",
             "before": 0.95, "after": 0.5}]),
        [{"claim": "accuracy.declarations.accuracy", "reason": "does_not_reproduce",
          "expected": None, "found": None}])
    assert ladder["verdicts"]["declared_accuracy:no_interval"]["reasons"] == [
        "param_loosened:min_interval_level", "evidence_removed:accuracy:1",
        "evidence_changed_outside_the_finding:accuracy:1",
        "unsupported:accuracy.declarations.accuracy"], (
        "reasons came back "
        f"{ladder['verdicts']['declared_accuracy:no_interval']['reasons']}; collect them in "
        "the four documented steps, in order: loosened parameters, then removed evidence, "
        "then evidence changed outside the finding, then claims that do not reproduce")
    # A check that was green and has simply stopped running is collateral too, and a
    # repair that leaves one behind is not honest however well its own findings scored.
    quiet = lesson.silencing_audit(
        [_finding("a")], _run_of(a=("red", "defect"), log_integrity=("green", "intact")),
        _run_of(a=("green", "ok"), log_integrity="not_run"), _delta_of(), _change_of(), [])
    assert quiet["counts"] == {"repaired": 1, "silenced": 0, "unfixed": 0}
    assert quiet["collateral"] == ["log_integrity"], (
        f"collateral came back {quiet['collateral']}; a check that was green and is now "
        "'not_run' is not green any more, and reading collateral as 'went red' misses the "
        "one way of losing a check that leaves no red row to notice")
    assert quiet["honest"] is False, (
        "'honest' came back True for a repair that fixed its own finding and stopped a "
        "check that was passing. It is True only when every verdict is 'repaired' AND "
        "collateral is empty")


# ------------------------------------------------------------------------- exercise 7
def test_plan_uses_only_honest_operations_and_names_its_findings():
    plan = lesson.plan_repairs(PACK, REF_FINDINGS)
    assert isinstance(plan, list) and plan, "plan_repairs returned nothing"
    stray = sorted({op["op"] for op in plan} - set(lesson.HONEST_OPS))
    assert not stray, (
        f"the plan uses {stray}. Those change the check rather than the world: deleting "
        "the records, widening a threshold, switching a check off, re-chaining a log, "
        "republishing an anchor, editing a sealed entry or writing a value in by hand")
    ids = {f["finding_id"] for f in REF_FINDINGS}
    assert all(op.get("finding") in ids for op in plan), (
        "every operation carries the finding_id it belongs to; the report joins on it")
    assert {op["finding"] for op in plan} == ids, (
        f"the plan covers {len({op['finding'] for op in plan})} of the {len(ids)} "
        "findings; every one of the six is repairable from what the pack already holds")


def test_plan_restores_the_segment_that_holds_the_missing_entries():
    plan = lesson.plan_repairs(PACK, REF_FINDINGS)
    restores = [op for op in plan if op["op"] == "restore_log_segment"]
    assert len(restores) == 1, f"{len(restores)} restore operations, expected 1"
    assert restores[0]["segment"] == "export-2026-05-14", (
        f"restored {restores[0]['segment']!r}; match each cold-store segment's 'covers' "
        "range against detail['missing_seq'] — the other export holds entries the log "
        "already has, and restoring it succeeds and changes nothing")
    after, _ = lesson.apply_repairs(PACK, SUITE, restores)
    result = lesson.CHECKS["log_integrity"](after, {})
    assert result["green"], (
        f"after the restore the log check still says {result['reason_code']}; the entries "
        "go back at their own sequence numbers, which is what reconnects the chain to the "
        "published anchor")


def test_plan_recomputes_the_interval_before_the_document_copies_it():
    plan = lesson.plan_repairs(PACK, REF_FINDINGS)
    order = [op["op"] for op in plan]
    assert "recompute_declaration" in order and "resource_section" in order, (
        f"the plan is {order}; the undeclarable metric is recomputed from the retained "
        "evaluation split and the document section is re-sourced from the result")
    assert order.index("recompute_declaration") < order.index("resource_section"), (
        "the document repair copies the number the accuracy repair produces, so it has to "
        "come second; the findings are already in that order")
    sourcing = [op for op in plan if op["op"] == "resource_section"][0]
    assert (sourcing["artefact"], sourcing["field"]) == ("metrics",
                                                         "primary_metric_value"), (
        f"resource_section names artefact {sourcing.get('artefact')!r} field "
        f"{sourcing.get('field')!r}; split them out of the empty path the finding reports")
    after, _ = lesson.apply_repairs(PACK, SUITE, plan)
    for check in ("declared_accuracy", "document_sourcing"):
        result = lesson.CHECKS[check](after, SUITE[check]["params"])
        assert result["green"], f"{check} still says {result['reason_code']}"
    claims = {row["claim"] for row in lesson.unsupported_values(after)}
    assert not claims, f"after the repair these numbers still do not reproduce: {claims}"


def test_plan_voids_the_reviews_and_withholds_the_decisions():
    plan = lesson.plan_repairs(PACK, REF_FINDINGS)
    voids = [op for op in plan if op["op"] == "void_verifications"]
    withholds = [op for op in plan if op["op"] == "withhold_decisions"]
    assert voids and withholds, (
        "voiding the implausible reviews leaves the decisions they supported standing on "
        "nothing, which is a different red; withhold them too")
    assert sorted(voids[0]["ids"]) == sorted(
        k.split(":", 1)[1] for k in BY_CHECK["oversight_plausibility"]["evidence_keys"]
        if k.startswith("verification:")), (
        "the ids to void are the verification: keys of the finding, without the prefix")
    assert voids[0].get("authority", "").strip() and voids[0].get("reason", "").strip(), (
        "void_verifications refuses to run without an authority and a reason: a review "
        "voided by nobody, for nothing, is a deletion with extra steps")
    after, _ = lesson.apply_repairs(PACK, SUITE, plan)
    result = lesson.CHECKS["oversight_plausibility"](
        after, SUITE["oversight_plausibility"]["params"])
    assert result["green"], (
        f"the oversight check still says {result['reason_code']} after the repair")
    index = _ref_delta(lesson.evidence_index(PACK), lesson.evidence_index(after))
    assert index["oversight"]["removed"] == [], (
        "the repair removed oversight records. Voiding keeps the record and marks it; "
        "deleting it is the first entry in TEMPTING_PLANS")


def test_plan_withdraws_reassesses_and_reissues_in_that_order():
    plan = lesson.plan_repairs(PACK, REF_FINDINGS)
    steps = [op["op"] for op in plan
             if op["op"] in ("withdraw_declaration", "reassess_conformity",
                             "reissue_declaration")]
    assert steps == ["withdraw_declaration", "reassess_conformity",
                     "reissue_declaration"], (
        f"the declaration steps came back {steps}; reissue_declaration refuses to run "
        "while the old declaration is still in force, and an assessment dated before the "
        "modification it covers is refused too")
    reassess = [op for op in plan if op["op"] == "reassess_conformity"][0]
    latest = BY_CHECK["declaration_currency"]["detail"]["latest_substantial"]
    assert str(reassess["on"]) >= str(latest), (
        f"the reassessment is dated {reassess['on']}, before the substantial modification "
        f"on {latest} that made it necessary")
    after, _ = lesson.apply_repairs(PACK, SUITE, plan)
    result = lesson.CHECKS["declaration_currency"](after, {})
    assert result["green"], (
        f"the declaration check still says {result['reason_code']}")
    index = _ref_delta(lesson.evidence_index(PACK), lesson.evidence_index(after))
    assert index["conformity"]["removed"] == [], (
        "the repair removed conformity evidence; the change log is what made the "
        "reassessment necessary, and deleting it is the point of SILENT_PLAN")
    assert lesson.verify_ledger(after["ledger"])["intact"], (
        "the ledger chain is broken after the repair: append new entries, never edit the "
        "sealed ones")


def test_plan_refits_the_limits_and_files_the_incident():
    plan = lesson.plan_repairs(PACK, REF_FINDINGS)
    refits = [op for op in plan if op["op"] == "refit_limits_to_sealed_baseline"]
    incidents = [op for op in plan if op["op"] == "file_incident"]
    assert refits and incidents, (
        "refitting the limits to the sealed baseline un-hides an excursion nobody has "
        "filed, and a monitor that has found one and filed nothing is still red")
    assert plan.index(refits[0]) < plan.index(incidents[0])
    hidden = BY_CHECK["monitoring_limits"]["detail"]["windows_hidden"]
    assert sorted(int(w) for w in incidents[0]["windows"]) == sorted(hidden), (
        f"the incident covers windows {incidents[0]['windows']}, and the finding says the "
        f"refitted limits were hiding {hidden}")
    assert str(incidents[0].get("aware_on", "")).strip(), (
        "an incident needs an awareness date: the Article 73 clock starts when the "
        "provider becomes aware, which is the day of this inspection")
    after, suite_after = lesson.apply_repairs(PACK, SUITE, plan)
    result = lesson.CHECKS["monitoring_limits"](after, suite_after["monitoring_limits"]
                                                ["params"])
    assert result["green"], (
        f"the monitoring check still says {result['reason_code']}")
    assert suite_after == SUITE, "the repair edited the suite; k is not a repair"


def test_plan_repairs_on_a_variant_pack():
    variant = _variant_pack()
    findings = _ref_findings(_ref_run(variant, SUITE))
    plan = lesson.plan_repairs(variant, findings)
    after, suite_after = lesson.apply_repairs(variant, SUITE, plan)
    results = _ref_results(after, suite_after)
    still_red = sorted(c for c, r in results.items() if r["status"] != "green")
    assert not still_red, (
        f"on a pack with the same six defects in different places, these are still not "
        f"green: {[(c, results[c]['reason_code']) for c in still_red]}. The variant purges "
        "a different window into a different export, makes thirty more reviews "
        "implausible, adds a second substantial modification and points the document "
        "section at a different metric — every one of those is in the findings you were "
        "handed")
    index = _ref_delta(lesson.evidence_index(variant), lesson.evidence_index(after))
    removed = {f: v["removed"] for f, v in index.items() if v["removed"]}
    assert not removed, f"the repair removed evidence: {removed}"
    assert suite_after == SUITE, "the repair edited the suite"
    assert not lesson.unsupported_values(after), (
        f"numbers that do not reproduce after the variant repair: "
        f"{lesson.unsupported_values(after)}")


# ------------------------------------------------------------------------- exercise 8
def _fake_result(verdicts: dict, before: dict, after: dict, **kw) -> dict:
    findings = [_finding(check, code) for check, code in
                [(f.split(":")[0], f.split(":")[1]) for f in verdicts]]
    counts = {kind: sum(1 for v in verdicts.values() if v == kind)
              for kind in ("repaired", "silenced", "unfixed")}
    audit = {"verdicts": {fid: {"verdict": v, "reasons": [f"because:{v}"]}
                          for fid, v in verdicts.items()},
             "counts": counts, "collateral": kw.get("collateral", []),
             "honest": counts["repaired"] == len(verdicts) and not kw.get("collateral")}
    return {"findings": findings, "before_run": before, "after_run": after,
            "after_pack": {"system_id": "loan-copilot"},
            "delta": _delta_of(log={"added": ["a"], "mutated": ["b", "c"]}),
            "suite_change": _change_of(**kw.get("change", {})),
            "unsupported": kw.get("unsupported", []), "audit": audit}


def test_report_shape_and_rows():
    result = _fake_result({"a:one": "repaired", "b:two": "silenced"},
                          _run_of(a=("red", "one"), b=("red", "two"), c=("green", "ok")),
                          _run_of(a=("green", "ok"), b=("green", "ok"), c=("green", "ok")))
    got = lesson.inspection_report(result)
    assert set(got) == {"system_id", "as_of", "checks", "green_before", "green_after",
                        "findings", "counts", "collateral", "evidence", "suite_unchanged",
                        "unsupported", "verdict"}, (
        f"the report has keys {sorted(got)}; return exactly the twelve documented names")
    assert got["system_id"] == "loan-copilot" and got["as_of"] == lesson.AS_OF.isoformat()
    assert (got["checks"], got["green_before"], got["green_after"]) == (3, 1, 3), (
        f"checks/green_before/green_after came back "
        f"{(got['checks'], got['green_before'], got['green_after'])}, expected (3, 1, 3)")
    assert [r["finding_id"] for r in got["findings"]] == ["a:one", "b:two"], (
        "rows stay in the findings' own order, worst first")
    assert set(got["findings"][0]) == {"finding_id", "evidence_id", "module", "share",
                                       "verdict", "reasons"}, (
        f"a row has keys {sorted(got['findings'][0])}; six names, exactly")
    assert got["findings"][1]["verdict"] == "silenced"
    assert got["findings"][1]["reasons"] == ["because:silenced"]


def test_report_verdict_precedence():
    both = _fake_result({"a:one": "silenced", "b:two": "unfixed"},
                        _run_of(a=("red", "one"), b=("red", "two")),
                        _run_of(a=("green", "ok"), b=("red", "two")))
    assert lesson.inspection_report(both)["verdict"] == "rejected: evidence silenced", (
        "with one silenced finding and one unfixed one the report must lead with the "
        "silenced one: a pack that is honestly red is in better shape than one that has "
        "been made green")
    unfixed = _fake_result({"a:one": "repaired", "b:two": "unfixed"},
                           _run_of(a=("red", "one"), b=("red", "two")),
                           _run_of(a=("green", "ok"), b=("red", "two")))
    assert lesson.inspection_report(unfixed)["verdict"] == "rejected: findings unresolved"
    clean = _fake_result({"a:one": "repaired"}, _run_of(a=("red", "one")),
                         _run_of(a=("green", "ok")))
    assert lesson.inspection_report(clean)["verdict"] == "accepted"
    collateral = _fake_result({"a:one": "repaired"}, _run_of(a=("red", "one"),
                                                             c=("green", "ok")),
                              _run_of(a=("green", "ok"), c=("red", "broke")),
                              collateral=["c"])
    assert lesson.inspection_report(collateral)["verdict"] == \
        "rejected: findings unresolved", (
        "a repair that broke a check nobody had complained about is a rejection too; "
        "collateral is in the audit for this reason")


def test_report_carries_the_measurements_through():
    result = _fake_result({"a:one": "repaired"}, _run_of(a=("red", "one")),
                          _run_of(a=("green", "ok")),
                          change={"unchanged": False, "disabled": ["z"]},
                          unsupported=[{"claim": "ledger", "reason": "ledger_chain_broken",
                                        "expected": None, "found": 1},
                                       {"claim": "accuracy.declarations.accuracy",
                                        "reason": "does_not_reproduce", "expected": None,
                                        "found": None}])
    got = lesson.inspection_report(result)
    assert got["evidence"] == {"removed": 0, "added": 1, "mutated": 2}, (
        f"evidence came back {got['evidence']}; pass delta['totals'] through unchanged")
    assert got["suite_unchanged"] is False
    assert got["unsupported"] == ["accuracy.declarations.accuracy", "ledger"], (
        f"unsupported came back {got['unsupported']}; the sorted claim names, not the rows")
    assert got["counts"] == {"repaired": 1, "silenced": 0, "unfixed": 0}


RUBRIC = [
    (test_run_suite_status_per_check, 4,
     "call each enabled check and carry reason_code, affected, population and detail "
     "through unchanged"),
    (test_run_suite_never_calls_or_passes_a_disabled_check, 4,
     "a disabled check is 'not_run' with empty counts — do not call it, and never call it "
     "green"),
    (test_run_suite_verdict_and_unknown_checks, 3,
     "'no_such_check' for an id nobody implemented; 'pass' needs red AND not_run empty"),
    (test_findings_cover_red_and_not_run, 4,
     "one finding per check that is not green, the not_run ones included"),
    (test_findings_carry_meta_and_evidence_keys, 3,
     "copy evidence_id, module and basis from CHECK_META and sort detail['keys']"),
    (test_findings_share_is_a_ratio_and_orders_the_list, 4,
     "share is affected/population, and the list is sorted by share descending then id"),
    (test_a_check_that_did_not_run_is_the_worst_finding, 3,
     "a not_run finding has share 1.0, reason_code not_verified and no counts"),
    (test_delta_partitions_every_key, 4,
     "removed = before minus after, added = after minus before, mutated = both with a "
     "different digest"),
    (test_delta_handles_a_family_missing_from_either_side, 4,
     "walk the union of the families, not either index's keys"),
    (test_delta_totals_and_conserved_on_the_real_pack, 3,
     "totals sum over families; conserved is about removals only"),
    (test_suite_delta_direction_decides_loosened, 4,
     "PARAM_DIRECTION says which way is weaker: 'lower' means after < before"),
    (test_suite_delta_tightening_is_not_silencing, 4,
     "a tightened threshold goes in 'tightened' and leaves 'unchanged' True"),
    (test_suite_delta_disabled_removed_and_dropped_parameters, 4,
     "a parameter that disappeared is loosened with after None; a check that disappeared "
     "is removed_checks"),
    (test_suite_delta_never_guesses_an_unknown_direction, 2,
     "a parameter with no direction goes in 'unknown'; one only 'after' has is ignored"),
    (test_unsupported_on_the_shipped_pack, 4,
     "two claims on the shipped pack: the bare point estimate and the refitted limits"),
    (test_unsupported_catches_an_interval_that_was_typed_in, 4,
     "recompute with the declaration's own metric, n_boot, level and seed; no seed is a "
     "different failure"),
    (test_unsupported_checks_the_document_against_its_source, 4,
     "the section must equal the field it cites, and the field must be digest-linked to a "
     "declaration"),
    (test_unsupported_checks_the_limits_and_the_ledger, 4,
     "recompute control_limits from the sealed entry at the limits' own k; verify the "
     "ledger chain"),
    (test_audit_a_check_that_did_not_run_is_silenced, 4,
     "missing or not_run is silenced before anything else is considered"),
    (test_audit_a_red_check_is_unfixed_not_silenced, 4,
     "a red check is unfixed whatever the repair did elsewhere; report its CURRENT reason "
     "code"),
    (test_audit_removed_evidence_silences_a_green_check, 4,
     "count removals only in CHECK_FAMILIES[check], and name the family and the count"),
    (test_audit_a_loosened_parameter_silences_only_its_own_check, 4,
     "match the loosened rows on their 'check' field"),
    (test_audit_ignores_mutations_inside_the_finding, 4,
     "a mutated key listed in the finding's evidence_keys is the repair; an added key is "
     "never a signal"),
    (test_audit_maps_unsupported_claims_counts_and_collateral, 4,
     "use checks_for_claim(); collateral is green-before and not-green-after"),
    (test_plan_uses_only_honest_operations_and_names_its_findings, 4,
     "only the ten HONEST_OPS, and every operation carries its finding_id"),
    (test_plan_restores_the_segment_that_holds_the_missing_entries, 5,
     "match each segment's 'covers' range against detail['missing_seq']"),
    (test_plan_recomputes_the_interval_before_the_document_copies_it, 5,
     "recompute first, then re-source the section from the declaration it produced"),
    (test_plan_voids_the_reviews_and_withholds_the_decisions, 6,
     "void with an authority and a reason, then withhold the decisions those "
     "verifications supported"),
    (test_plan_withdraws_reassesses_and_reissues_in_that_order, 6,
     "withdraw, then reassess no earlier than the modification, then reissue"),
    (test_plan_refits_the_limits_and_files_the_incident, 6,
     "refit to the sealed baseline, then file an incident covering detail"
     "['windows_hidden']"),
    (test_plan_repairs_on_a_variant_pack, 6,
     "read the segment, the ids, the dates and the metric out of the findings rather than "
     "out of this notebook"),
    (test_report_shape_and_rows, 4,
     "twelve keys, six per row, rows in the findings' order"),
    (test_report_verdict_precedence, 4,
     "silenced outranks unfixed; collateral is a rejection too"),
    (test_report_carries_the_measurements_through, 2,
     "pass delta['totals'] and suite_change['unchanged'] through; sort the claim names"),
]
