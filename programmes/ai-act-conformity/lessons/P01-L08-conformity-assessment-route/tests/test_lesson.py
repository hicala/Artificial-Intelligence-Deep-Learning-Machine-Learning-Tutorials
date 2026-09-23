"""Autograder rubric for P01-L08. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Every expected value is RE-DERIVED here, in plain Python, from the lesson's fixtures and
  the articles the lesson cites. Nothing is a literal copied out of the notebook's output,
  and nothing calls the student's function to decide what the student's function should have
  returned. `_expected_route`, `_expected_modification`, `_expected_buckets` and
  `_expected_point` are written in a different shape from the lesson's own implementations.
* Nothing in `lesson` is mutated. Any test that needs a different pack, record, context or
  plan builds one, deep-copying through `json` where a nested dict is touched. A mutated
  fixture would leak into the tests that run after it and make a failure depend on rubric
  order.
* The downstream exercises are graded on route records, readiness reports and declaration
  results BUILT HERE rather than produced by the student's earlier functions, so a student
  who has not written `assessment_route` yet can still score full marks on `readiness`. The
  one exception is the end-to-end test, which is an integration test and says so.
* Several tests run on HAND-BUILT inputs rather than the lesson's nine systems, because the
  fixture has one instance of each condition and an implementation can pass it by
  hard-coding a system id. The points 2-to-8 sweep, the bucket-precedence pack, the verdict
  ladder, the CE marking truth table and the "clean twin" of cv-ranker are all of that kind.
* `test_route_points_2_to_8_ignore_the_standards_gap` is the load-bearing one. Article 43(2)
  has no standards test in it, and the single most plausible wrong implementation is to run
  43(1)'s test on every Annex III system. It sweeps all seven points against all sixteen
  standards contexts, so nothing passes it by accident.
* `test_end_to_end_cv_ranker_is_refused_and_a_clean_twin_is_not` is the trap MODULES.md
  asks for, and it is deliberately two-sided: cv-ranker's pack is complete and its
  declaration must still be refused, AND a twin whose only difference is that the
  purpose-changing modification never happened must get one. An implementation that refuses
  everything scores nothing here, and neither does one that refuses nothing.
"""
import importlib.util, json, os, re
from datetime import date
from pathlib import Path

# CI grades the reference implementation; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

AS_OF = lesson.AS_OF
SYSTEM_IDS = tuple(lesson.RECORDS)


def _copy(obj):
    """A deep copy that shares nothing with the lesson's fixtures."""
    return json.loads(json.dumps(obj, default=list))


# --- independent re-derivations, in plain Python ----------------------------------------

def _expected_point(record: dict):
    """Find the Annex III point without the lesson's regex or its traversal."""
    for step in record["question_trail"]:
        if step["question"] != "q3_annex_iii_area":
            continue
        answer = step["answer"]
        if not answer:
            return None
        text = str(answer).lower()
        marker = "annex iii point "
        at = text.find(marker)
        if at < 0:
            return None
        digits = ""
        for ch in text[at + len(marker):]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            return None
        value = int(digits)
        return value if 1 <= value <= 8 else None
    return None


def _expected_gaps(ctx: dict) -> list:
    """The four cases of Article 43(1), second subparagraph."""
    std = bool(ctx.get("harmonised_standard_exists"))
    spec = bool(ctx.get("common_specification_exists"))
    out = []
    if not (std or spec):
        out.append(lesson.GAP_NO_STANDARD)
    if std and ctx.get("harmonised_standard_applied") != "full":
        out.append(lesson.GAP_NOT_FULLY_APPLIED)
    if spec and not ctx.get("common_specification_applied"):
        out.append(lesson.GAP_COMMON_SPEC_NOT_APPLIED)
    if std and ctx.get("harmonised_standard_restricted"):
        out.append(lesson.GAP_RESTRICTED)
    return sorted(out)


def _expected_route(record: dict, ctx: dict) -> dict:
    """Article 43, re-derived from the articles rather than from the lesson's branch order."""
    tier = record["risk_tier"]
    point, gaps = None, []
    if tier == "prohibited":
        route, reason = "none", lesson.REASON_PROHIBITED
    elif tier == "high_risk_annex_i":
        route, reason = "annex_i_product_procedure", lesson.REASON_ANNEX_I
    elif tier != "high_risk_annex_iii":
        route, reason = "not_high_risk", lesson.REASON_NOT_HIGH_RISK
    else:
        point = _expected_point(record)
        if point is None:
            route, reason = "undetermined", lesson.REASON_POINT_UNKNOWN
        elif point > 1:
            route, reason = "annex_vi_internal_control", lesson.REASON_POINTS_2_TO_8
        else:
            gaps = _expected_gaps(ctx)
            if gaps:
                route, reason = "annex_vii_notified_body", lesson.REASON_POINT_1_STANDARDS_GAP
            else:
                chosen = ctx.get("chosen_route")
                route = chosen if chosen in ("annex_vi_internal_control",
                                             "annex_vii_notified_body") \
                    else "annex_vi_internal_control"
                reason = lesson.REASON_POINT_1_CHOICE
    if route == "annex_vii_notified_body":
        needed = True
    elif route == "annex_i_product_procedure":
        needed = bool(ctx.get("annex_i_notified_body_involved"))
    else:
        needed = False
    role = None
    if needed:
        role = ("market_surveillance_authority"
                if ctx.get("deployer_authority") in lesson.AUTHORITY_DEPLOYERS
                else "notified_body")
    # The article each outcome rests on. A route record that cites the wrong article is a
    # route record an inspector cannot check, so the basis is graded, not merely non-empty.
    basis = {lesson.REASON_PROHIBITED: "Article 5",
             lesson.REASON_ANNEX_I: "Article 43(3)",
             lesson.REASON_POINT_UNKNOWN: "Article 43(1)-(2)",
             lesson.REASON_POINT_1_CHOICE: "Article 43(1)",
             lesson.REASON_POINT_1_STANDARDS_GAP: "Article 43(1) second subparagraph",
             lesson.REASON_POINTS_2_TO_8: "Article 43(2)",
             lesson.REASON_NOT_HIGH_RISK: "Article 43 (applies to high-risk AI systems)"}[reason]
    return {"annex_iii_point": point, "route": route, "gaps": gaps,
            "notified_body_required": needed, "notified_body_role": role, "reason": reason,
            "basis": basis}


def _expected_modification(change: dict, plan: dict) -> tuple:
    """(substantial, reason) from Article 3(23) and Article 43(4)."""
    placed, on = plan.get("placed_on_market"), change.get("on")
    if placed and on and date.fromisoformat(on) < date.fromisoformat(placed):
        return False, lesson.REASON_BEFORE_PLACING
    if change.get("modifies_intended_purpose"):
        return True, lesson.REASON_INTENDED_PURPOSE
    if change.get("change_id") in tuple(plan.get("pre_determined") or ()) \
            and plan.get("documented_in_annex_iv_2f"):
        return False, lesson.REASON_PRE_DETERMINED
    if change.get("affects_chapter_iii_compliance"):
        return True, lesson.REASON_COMPLIANCE_AFFECTED
    return False, lesson.REASON_NO_EFFECT


def _expected_buckets(route: str, pack: dict, as_of) -> dict:
    """present / missing / out_of_date / untraceable, without the lesson's loop shape."""
    required = list(lesson.ROUTE_EVIDENCE[route])
    evidence, artefacts = pack["evidence"], pack["artefacts"]
    present, missing = [], []
    for item in required:
        record = evidence.get(item)
        doc = record.get("document") if record and record.get("status") == "present" else None
        if isinstance(doc, str) and doc.strip():
            present.append(item)
        else:
            missing.append(item)
    stale, untraceable = [], []
    for item in present:
        record = evidence[item]
        issued = record.get("date")
        review = record.get("review_due")
        bad_date = (not issued) or date.fromisoformat(issued) > as_of
        bad_review = bool(review) and date.fromisoformat(review) < as_of
        if bad_date or bad_review:
            stale.append(item)
        elif record.get("source") not in artefacts:
            untraceable.append(item)
    return {"required": sorted(required), "present": sorted(present),
            "missing": sorted(missing), "out_of_date": sorted(stale),
            "untraceable": sorted(untraceable),
            "coverage": len(present) / len(required) if required else 1.0}


def _route_record(system_id="x", route="annex_vi_internal_control", tier="high_risk_annex_iii",
                  point=5, nb=False, role=None):
    """A route record built HERE, so downstream exercises grade independently of exercise 3."""
    return {"system_id": system_id, "risk_tier": tier, "annex_iii_point": point,
            "route": route, "gaps": [], "notified_body_required": nb,
            "notified_body_role": role, "reason": "fixed-by-the-rubric", "basis": "Article 43"}


def _readiness_report(verdict="ready", **extra):
    base = {"system_id": "x", "route": "annex_vi_internal_control", "required": [],
            "present": [], "missing": [], "out_of_date": [], "untraceable": [],
            "coverage": 1.0, "blocking": [], "route_reset": False, "verdict": verdict,
            "ready": verdict == "ready"}
    base.update(extra)
    return base


def _clean_pack(**over):
    """A pack with nothing wrong with it, on the internal-control route."""
    pack = _copy(lesson.PACKS["face-match"])
    pack.update(over)
    return pack


def _change(cid, on, compliance=False, purpose=False):
    return {"change_id": cid, "on": on, "description": "t",
            "affects_chapter_iii_compliance": compliance,
            "modifies_intended_purpose": purpose}


def _plan(placed="2026-01-01", assessed="2026-01-01", pre=(), documented=False):
    return {"placed_on_market": placed, "assessed_on": assessed,
            "pre_determined": tuple(pre), "documented_in_annex_iv_2f": documented}


# --- exercise 1: annex_iii_point ---------------------------------------------------------

def test_annex_iii_point_matches_an_independent_derivation():
    got = {sid: lesson.annex_iii_point(lesson.RECORDS[sid]) for sid in SYSTEM_IDS}
    want = {sid: _expected_point(lesson.RECORDS[sid]) for sid in SYSTEM_IDS}
    assert got == want, f"got {got}, expected {want}"
    assert all(v is None or isinstance(v, int) for v in got.values()), (
        f"the point is an int or None, never a string: {got}")
    assert sum(1 for v in got.values() if v is not None) == 6, (
        f"{sum(1 for v in got.values() if v is not None)} of the nine records determine a "
        "point; six of them do")


def test_annex_iii_point_rejects_a_point_outside_annex_iii():
    for text, want in (("(Annex III point 1)", 1), ("(Annex III point 8)", 8),
                       ("(Annex III point 0)", None), ("(Annex III point 9)", None),
                       ("(Annex III point 12)", None), ("(Annex III point 80)", None)):
        rec = lesson._record("t", "high_risk_annex_iii",
                             lesson._annex_iii_trail(f"an area {text}", "not_claimed"))
        got = lesson.annex_iii_point(rec)
        assert got == want, (
            f"{text!r} came back {got!r}, expected {want!r} — read the WHOLE number and "
            "reject anything outside ANNEX_III_POINTS; 'point 12' is not point 1")


def test_annex_iii_point_reads_the_q3_step_and_nothing_else():
    no_q3 = lesson._record("t", "high_risk_annex_iii",
                           [("q1_prohibited", False, False),
                            ("q2_annex_i_safety_component", True, True)])
    assert lesson.annex_iii_point(no_q3) is None, "no q3 step at all means None"
    blank = lesson._record("t", "high_risk_annex_iii",
                           lesson._annex_iii_trail("", "not_claimed"))
    assert lesson.annex_iii_point(blank) is None, "an empty q3 answer means None"
    elsewhere = lesson._record("t", "high_risk_annex_iii",
                               [("q1_prohibited", False, False),
                                ("q3_annex_iii_area", "an unlabelled area", False),
                                ("q4_article_6_3", "see Annex III point 7", True)])
    assert lesson.annex_iii_point(elsewhere) is None, (
        "the point is read off the q3 step alone; scanning the whole trail lets an Article "
        "6(3) note decide the conformity assessment route")
    twice = lesson._record("t", "high_risk_annex_iii",
                           [("q3_annex_iii_area", "employment (Annex III point 4(a))", False),
                            ("q3_annex_iii_area", "biometrics (Annex III point 1)", False)])
    assert lesson.annex_iii_point(twice) == 4, (
        "take the FIRST q3 step; a re-asked question is a correction to record, not a new "
        "answer to prefer silently")


# --- exercise 2: standards_gap -----------------------------------------------------------

def _all_contexts():
    for std in (False, True):
        for applied in ("full", "partial", "none"):
            for restricted in (False, True):
                for spec in (False, True):
                    for spec_applied in (False, True):
                        yield {"harmonised_standard_exists": std,
                               "harmonised_standard_applied": applied,
                               "harmonised_standard_restricted": restricted,
                               "common_specification_exists": spec,
                               "common_specification_applied": spec_applied}


def test_standards_gap_matches_an_independent_derivation():
    wrong = []
    for ctx in _all_contexts():
        got = lesson.standards_gap(ctx)
        want = _expected_gaps(ctx)
        if got != want:
            wrong.append((ctx, got, want))
    assert not wrong, (
        f"{len(wrong)} of 48 contexts disagree; first: {wrong[0][0]} gave {wrong[0][1]}, "
        f"expected {wrong[0][2]}")


def test_standards_gap_collects_every_gap_sorted():
    ctx = {"harmonised_standard_exists": True, "harmonised_standard_applied": "none",
           "harmonised_standard_restricted": True, "common_specification_exists": True,
           "common_specification_applied": False}
    got = lesson.standards_gap(ctx)
    want = sorted([lesson.GAP_NOT_FULLY_APPLIED, lesson.GAP_COMMON_SPEC_NOT_APPLIED,
                   lesson.GAP_RESTRICTED])
    assert got == want, (
        f"got {got}, expected {want} — three gaps fire together here. Returning the first "
        "one you find throws away the other two reasons a reviewer needs")
    assert got == sorted(got), "the list is sorted"
    assert all(g in lesson.STANDARDS_GAPS for g in got), (
        f"every id must come from STANDARDS_GAPS; got {got}")


def test_standards_gap_treats_a_missing_key_as_falsy():
    assert lesson.standards_gap({}) == [lesson.GAP_NO_STANDARD], (
        "an empty context has neither a standard nor a common specification, which is "
        "gap (a) alone — (b), (c) and (d) each need something to exist first")
    assert lesson.standards_gap({"common_specification_exists": True,
                                 "common_specification_applied": True}) == [], (
        "a common specification that exists and was applied closes gap (a) and does not "
        "open gap (c)")
    assert lesson.standards_gap({"harmonised_standard_exists": True}) == [
        lesson.GAP_NOT_FULLY_APPLIED], (
        "a standard with no applied field recorded is not 'full', so gap (b) fires; read "
        "every field with .get() rather than indexing")


# --- exercise 3: assessment_route --------------------------------------------------------

def test_route_matches_an_independent_derivation_on_every_record():
    for sid in SYSTEM_IDS:
        got = lesson.assessment_route(lesson.RECORDS[sid], lesson.CONTEXTS[sid])
        want = _expected_route(lesson.RECORDS[sid], lesson.CONTEXTS[sid])
        assert set(got) == {"system_id", "risk_tier", "annex_iii_point", "route", "gaps",
                            "notified_body_required", "notified_body_role", "reason",
                            "basis"}, f"{sid}: keys were {sorted(got)}"
        for key, expected in want.items():
            assert got[key] == expected, (
                f"{sid}: {key} came back {got[key]!r}, expected {expected!r}")
        assert got["system_id"] == sid and got["risk_tier"] == lesson.RECORDS[sid]["risk_tier"]
        assert got["route"] in lesson.ROUTES, f"{sid}: {got['route']!r} is not in ROUTES"
        assert isinstance(got["basis"], str) and got["basis"].strip(), (
            f"{sid}: every outcome cites the article it rests on")
    bases = {lesson.assessment_route(lesson.RECORDS[sid], lesson.CONTEXTS[sid])["basis"]
             for sid in SYSTEM_IDS}
    assert len(bases) == 7, (
        f"the nine records reach seven different outcomes and cite {len(bases)} different "
        f"article(s): {sorted(bases)}. A basis that is the same string whatever the rule "
        "tells the next reader nothing about which sentence of Article 43 decided this")


def test_route_points_2_to_8_ignore_the_standards_gap():
    contexts = list(_all_contexts())[:16]
    for point in range(2, 9):
        rec = lesson._record("t", "high_risk_annex_iii", lesson._annex_iii_trail(
            f"an area (Annex III point {point})", "not_claimed"))
        for ctx in contexts:
            full = dict(ctx, chosen_route="annex_vii_notified_body", deployer_authority=None)
            got = lesson.assessment_route(rec, full)
            assert got["route"] == "annex_vi_internal_control", (
                f"point {point} with gaps {_expected_gaps(ctx)} came back {got['route']!r}. "
                "Article 43(2) is unconditional for points 2 to 8: internal control, no "
                "notified body, no standards test and no provider choice")
            assert got["gaps"] == [], (
                f"point {point} reported gaps {got['gaps']} — the gap did not bear on the "
                "decision, and reporting it tells the next reader that it did")
            assert got["notified_body_required"] is False and got["notified_body_role"] is None
            assert got["reason"] == lesson.REASON_POINTS_2_TO_8


def test_route_point_1_choice_is_only_a_choice_without_a_gap():
    clean = {"harmonised_standard_exists": True, "harmonised_standard_applied": "full",
             "harmonised_standard_restricted": False, "common_specification_exists": False,
             "common_specification_applied": False, "deployer_authority": None}
    rec = lesson._record("t", "high_risk_annex_iii",
                         lesson._annex_iii_trail("biometrics (Annex III point 1)",
                                                 "not_claimed"))
    for chosen, want in (("annex_vi_internal_control", "annex_vi_internal_control"),
                         ("annex_vii_notified_body", "annex_vii_notified_body"),
                         (None, "annex_vi_internal_control"),
                         ("annex_i_product_procedure", "annex_vi_internal_control"),
                         ("whatever", "annex_vi_internal_control")):
        got = lesson.assessment_route(rec, dict(clean, chosen_route=chosen))
        assert got["route"] == want, (
            f"chosen_route={chosen!r} gave {got['route']!r}, expected {want!r} — an absent "
            "or nonsensical preference falls back to internal control, and a preference for "
            "a route Article 43(1) does not offer is not honoured")
        assert got["reason"] == lesson.REASON_POINT_1_CHOICE
    for broken in ({"harmonised_standard_applied": "partial"},
                   {"harmonised_standard_restricted": True},
                   {"harmonised_standard_exists": False, "harmonised_standard_applied": "none"}):
        ctx = dict(clean, chosen_route="annex_vi_internal_control", **broken)
        got = lesson.assessment_route(rec, ctx)
        assert got["route"] == "annex_vii_notified_body", (
            f"{broken} left the route at {got['route']!r} — with any gap open the provider's "
            "stated preference is not consulted at all")
        assert got["gaps"] == _expected_gaps(ctx) and got["gaps"], (
            f"a point 1 decision reports the gaps that drove it; got {got['gaps']}")
        assert got["reason"] == lesson.REASON_POINT_1_STANDARDS_GAP


def test_route_undetermined_never_falls_back_to_a_procedure():
    rec = lesson._record("t", "high_risk_annex_iii",
                         lesson._annex_iii_trail("an area nobody labelled", "not_claimed"))
    for ctx in ({}, {"chosen_route": "annex_vi_internal_control"},
                {"harmonised_standard_exists": True, "harmonised_standard_applied": "full"}):
        got = lesson.assessment_route(rec, ctx)
        assert got["route"] == "undetermined", (
            f"an unreadable point came back {got['route']!r} — defaulting it to a procedure "
            "chooses a route on the strength of a record nobody can read")
        assert got["annex_iii_point"] is None and got["gaps"] == []
        assert got["notified_body_required"] is False
        assert got["reason"] == lesson.REASON_POINT_UNKNOWN


def test_route_notified_body_role_and_the_annex_i_case():
    rec = lesson._record("t", "high_risk_annex_iii",
                         lesson._annex_iii_trail("biometrics (Annex III point 1)",
                                                 "not_claimed"))
    forced = {"harmonised_standard_exists": True, "harmonised_standard_applied": "partial",
              "harmonised_standard_restricted": False, "common_specification_exists": False,
              "common_specification_applied": False, "chosen_route": None}
    for deployer in lesson.AUTHORITY_DEPLOYERS:
        got = lesson.assessment_route(rec, dict(forced, deployer_authority=deployer))
        assert got["notified_body_role"] == "market_surveillance_authority", (
            f"deployer {deployer!r} gave role {got['notified_body_role']!r} — Article 43(1)'s "
            "third subparagraph puts the market surveillance authority in the chair for all "
            f"{len(lesson.AUTHORITY_DEPLOYERS)} of them")
    got = lesson.assessment_route(rec, dict(forced, deployer_authority="a private bank"))
    assert got["notified_body_role"] == "notified_body", (
        f"an ordinary deployer gives role 'notified_body', got {got['notified_body_role']!r}")

    annex_i = lesson.RECORDS["lift-door-sensor"]
    quiet = lesson.assessment_route(annex_i, {"deployer_authority": "law_enforcement"})
    assert quiet["notified_body_required"] is False and quiet["notified_body_role"] is None, (
        "an Annex I system whose context does not say a body was involved requires none, and "
        "a role with no body required is None")
    loud = lesson.assessment_route(annex_i, {"annex_i_notified_body_involved": True,
                                             "deployer_authority": "law_enforcement"})
    assert loud["notified_body_required"] is True, (
        "whether the product legislation already involves a notified body is a fact about "
        "that legislation, and it must be reported")
    assert loud["notified_body_role"] == "market_surveillance_authority"
    vi = lesson.assessment_route(lesson.RECORDS["credit-scorer"],
                                 dict(lesson.CONTEXTS["credit-scorer"],
                                      deployer_authority="law_enforcement"))
    assert vi["notified_body_role"] is None, (
        "no body is required on internal control, so there is no role to name — an authority "
        "deployer does not conjure one")


# --- exercise 4: is_substantial_modification ---------------------------------------------

def test_substantial_matches_an_independent_derivation():
    plan = _plan(placed="2026-06-01", assessed="2026-07-30", pre=("C-PRE",), documented=True)
    cases = []
    for cid in ("C-PRE", "C-OTHER"):
        for on in ("2026-05-31", "2026-06-01", "2026-09-01"):
            for compliance in (False, True):
                for purpose in (False, True):
                    cases.append(_change(cid, on, compliance, purpose))
    wrong = []
    for change in cases:
        got = lesson.is_substantial_modification(change, plan)
        want_sub, want_reason = _expected_modification(change, plan)
        if (got["substantial"], got["reason"]) != (want_sub, want_reason):
            wrong.append((change, got, (want_sub, want_reason)))
    assert not wrong, (
        f"{len(wrong)} of {len(cases)} cases disagree; first: {wrong[0][0]} gave "
        f"({wrong[0][1]['substantial']}, {wrong[0][1]['reason']!r}), expected {wrong[0][2]}")
    got = lesson.is_substantial_modification(cases[0], plan)
    assert set(got) == {"change_id", "on", "substantial", "reason", "basis"}, (
        f"keys were {sorted(got)}")
    assert got["change_id"] == cases[0]["change_id"] and got["on"] == cases[0]["on"]


def test_substantial_pre_determined_needs_the_annex_iv_description():
    change = _change("C-PRE", "2026-09-01", compliance=True)
    documented = _plan(placed="2026-06-01", pre=("C-PRE",), documented=True)
    undocumented = _plan(placed="2026-06-01", pre=("C-PRE",), documented=False)
    got = lesson.is_substantial_modification(change, documented)
    assert got["substantial"] is False and got["reason"] == lesson.REASON_PRE_DETERMINED, (
        f"pre-determined AND documented is Article 43(4)'s carve-out; got {got['reason']!r}")
    assert got["basis"] == "Article 43(4)", f"basis came back {got['basis']!r}"
    got = lesson.is_substantial_modification(change, undocumented)
    assert got["substantial"] is True and got["reason"] == lesson.REASON_COMPLIANCE_AFFECTED, (
        f"got {got['reason']!r} — without the Annex IV point 2(f) description the carve-out "
        "does not apply, and the change falls through to the compliance test")
    stranger = lesson.is_substantial_modification(_change("C-NEW", "2026-09-01",
                                                          compliance=True), documented)
    assert stranger["substantial"] is True, (
        "the carve-out covers the changes named in the plan, not every change made by a "
        "provider who happens to have a plan")


def test_substantial_intended_purpose_runs_before_the_carve_out():
    plan = _plan(placed="2026-06-01", pre=("C-PRE",), documented=True)
    got = lesson.is_substantial_modification(
        _change("C-PRE", "2026-09-01", compliance=False, purpose=True), plan)
    assert got["substantial"] is True and got["reason"] == lesson.REASON_INTENDED_PURPOSE, (
        f"got ({got['substantial']}, {got['reason']!r}) — Article 43(4)'s carve-out is "
        "written about the system and its performance, so this lesson tests the intended "
        "purpose FIRST: you cannot pre-declare your way into a new purpose")
    assert got["basis"] == "Article 3(23)", f"basis came back {got['basis']!r}"
    both = lesson.is_substantial_modification(
        _change("C-NEW", "2026-09-01", compliance=True, purpose=True), plan)
    assert both["reason"] == lesson.REASON_INTENDED_PURPOSE, (
        f"got {both['reason']!r} — when both limbs of Article 3(23) are met, the purpose "
        "limb is the one named, because it is the one tested first")


def test_substantial_before_placing_is_strict_and_optional():
    plan = _plan(placed="2026-06-01", pre=(), documented=False)
    early = lesson.is_substantial_modification(
        _change("C1", "2026-05-31", compliance=True, purpose=True), plan)
    assert early["substantial"] is False and early["reason"] == lesson.REASON_BEFORE_PLACING, (
        f"got ({early['substantial']}, {early['reason']!r}) — a change before placing is part "
        "of what the initial assessment assessed, whatever limb it touches")
    assert early["basis"] == "Article 43(4)"
    same_day = lesson.is_substantial_modification(_change("C2", "2026-06-01", compliance=True),
                                                  plan)
    assert same_day["substantial"] is True, (
        "rule 1 is STRICTLY earlier: a change dated the day of placing is not before it")
    nodate = lesson.is_substantial_modification(_change("C3", "2026-01-01", compliance=True),
                                                {"pre_determined": (),
                                                 "documented_in_annex_iv_2f": False})
    assert nodate["substantial"] is True and nodate["reason"] == \
        lesson.REASON_COMPLIANCE_AFFECTED, (
        "a plan with no placing date on file cannot fire rule 1; it must not raise and it "
        f"must not excuse the change. Got {nodate['reason']!r}")
    quiet = lesson.is_substantial_modification(_change("C4", "2026-09-01"), plan)
    assert quiet["substantial"] is False and quiet["reason"] == lesson.REASON_NO_EFFECT, (
        "a change touching neither limb is a change, not a substantial modification")


# --- exercise 5: modification_history ----------------------------------------------------

def test_history_matches_an_independent_derivation():
    for sid in SYSTEM_IDS:
        log, plan = lesson.CHANGE_LOGS[sid], lesson.PLANS[sid]
        got = lesson.modification_history(_copy(log), _copy(plan))
        assert set(got) == {"assessed", "substantial_ids", "last_substantial_on",
                            "since_assessment", "route_reset"}, (
            f"{sid}: keys were {sorted(got)}")
        want_sub = sorted(c["change_id"] for c in log
                          if _expected_modification(c, plan)[0])
        assert got["substantial_ids"] == want_sub, (
            f"{sid}: substantial_ids came back {got['substantial_ids']}, expected {want_sub}")
        dates = [c["on"] for c in log if _expected_modification(c, plan)[0]]
        assert got["last_substantial_on"] == (max(dates) if dates else None), (
            f"{sid}: last_substantial_on is the LATEST substantial date, got "
            f"{got['last_substantial_on']!r}")
        assert len(got["assessed"]) == len(log), (
            f"{sid}: one verdict per change, got {len(got['assessed'])} for {len(log)}")


def test_history_reset_counts_only_uncovered_changes():
    plan = _plan(placed="2026-01-01", assessed="2026-06-01")
    log = [_change("A", "2026-03-01", compliance=True),
           _change("B", "2026-09-01", compliance=True)]
    got = lesson.modification_history(log, plan)
    assert got["substantial_ids"] == ["A", "B"], f"got {got['substantial_ids']}"
    assert got["since_assessment"] == ["B"] and got["route_reset"] is True, (
        f"since_assessment came back {got['since_assessment']} — A predates the assessment "
        "on file and is covered by it. A history that resets on ANY substantial change "
        "cannot tell a provider who re-assessed from one who did not")
    covered = lesson.modification_history([log[0]], plan)
    assert covered["substantial_ids"] == ["A"] and covered["route_reset"] is False, (
        f"a substantial change the assessment already covers must not reset the route; got "
        f"route_reset={covered['route_reset']}")
    boundary = lesson.modification_history([_change("C", "2026-06-01", compliance=True)], plan)
    assert boundary["route_reset"] is True, (
        "a substantial change dated exactly on assessed_on counts as uncovered — a day "
        "cannot order two events inside it, and the lesson takes the reading that leaves "
        "the provider re-assessing")
    day_before = lesson.modification_history([_change("D", "2026-05-31", compliance=True)],
                                             plan)
    assert day_before["route_reset"] is False, (
        "and the day before it is covered; the comparison is >=, not >")
    never = lesson.modification_history([_change("E", "2026-09-01", compliance=True)],
                                        _plan(placed="2026-01-01", assessed=None))
    assert never["route_reset"] is True, (
        "a plan with no assessed_on has assessed nothing, so every substantial change is "
        "uncovered — not covered by a comparison that silently succeeds")


def test_history_orders_the_log_without_mutating_it():
    plan = _plan(placed="2026-01-01", assessed="2027-01-01")
    # The ids run the OTHER WAY from the dates on purpose: an implementation that sorts by
    # change_id and forgets the date cannot pass this, and one that sorts by date and forgets
    # the tie-break cannot separate Y from Z.
    log = [_change("A", "2026-09-01"), _change("Z", "2026-03-01"),
           _change("Y", "2026-03-01"), _change("M", "2026-05-01")]
    before = json.dumps(log, sort_keys=True)
    got = lesson.modification_history(log, plan)
    assert json.dumps(log, sort_keys=True) == before, (
        "the caller's list came back reordered — sort a copy")
    assert [v["change_id"] for v in got["assessed"]] == ["Y", "Z", "M", "A"], (
        f"order came back {[v['change_id'] for v in got['assessed']]} — sort by (date, "
        "change_id), not by change_id: the log reads in TIME order however it arrived, and "
        "the id only breaks a tie between two changes on the same day")


def test_history_on_an_empty_and_an_all_quiet_log():
    plan = _plan(placed="2026-01-01", assessed="2026-06-01")
    empty = lesson.modification_history([], plan)
    assert empty["assessed"] == [] and empty["substantial_ids"] == [], f"got {empty}"
    assert empty["last_substantial_on"] is None and empty["route_reset"] is False, (
        "an empty log has no latest substantial date and resets nothing")
    quiet = lesson.modification_history([_change("Q", "2026-09-01")], plan)
    assert quiet["substantial_ids"] == [] and quiet["route_reset"] is False, (
        "a log of changes that touch nothing resets nothing")
    assert len(quiet["assessed"]) == 1, "every change still gets a verdict"


# --- exercise 6: readiness ---------------------------------------------------------------

def test_readiness_buckets_match_an_independent_derivation():
    for sid in SYSTEM_IDS:
        route = _expected_route(lesson.RECORDS[sid], lesson.CONTEXTS[sid])["route"]
        rr = _route_record(sid, route=route)
        history = {"route_reset": False}
        got = lesson.readiness(rr, _copy(lesson.PACKS[sid]), history, AS_OF)
        assert set(got) == {"system_id", "route", "required", "present", "missing",
                            "out_of_date", "untraceable", "coverage", "blocking",
                            "route_reset", "verdict", "ready"}, (
            f"{sid}: keys were {sorted(got)}")
        want = _expected_buckets(route, lesson.PACKS[sid], AS_OF)
        for key in ("required", "present", "missing", "out_of_date", "untraceable",
                    "coverage"):
            assert got[key] == want[key], (
                f"{sid}: {key} came back {got[key]!r}, expected {want[key]!r}")
        union = sorted(set(want["missing"]) | set(want["out_of_date"])
                       | set(want["untraceable"]))
        assert got["blocking"] == union, (
            f"{sid}: blocking is the sorted union of the three buckets; got "
            f"{got['blocking']}, expected {union}")


def test_readiness_uses_the_completeness_checker_it_was_given():
    for sid in SYSTEM_IDS:
        route = _expected_route(lesson.RECORDS[sid], lesson.CONTEXTS[sid])["route"]
        pack = _copy(lesson.PACKS[sid])
        reference = lesson.completeness_report(lesson.pack_as_trace(pack),
                                               tuple(lesson.ROUTE_EVIDENCE[route]))
        got = lesson.readiness(_route_record(sid, route=route), pack,
                               {"route_reset": False}, AS_OF)
        for key in ("present", "missing", "coverage"):
            assert got[key] == reference[key], (
                f"{sid}: {key} came back {got[key]!r} but P01-L01's completeness_report says "
                f"{reference[key]!r}. Two definitions of 'present' in one programme is a "
                "pack that passes one gate and fails the other with no way to say which "
                "was right")
    empty = lesson.readiness(_route_record("x", route="not_high_risk"),
                             _copy(lesson.PACKS["mood-meter"]), {"route_reset": False}, AS_OF)
    assert empty["coverage"] == 1.0 and empty["required"] == [], (
        f"a route with no required items is vacuously covered at 1.0, got "
        f"{empty['coverage']!r} — that is what completeness_report already returns, and a "
        "hand-rolled version divides by zero here")


def test_readiness_bucket_precedence_and_date_boundaries():
    art = "ART-ok"
    def pack(**record):
        base = {"status": "present", "document": "D-1", "date": "2026-01-01",
                "review_due": None, "source": art}
        base.update(record)
        return {"system_id": "t", "evidence": {"eu_database_registration": base,
                                               **{i: {"status": "present", "document": "D",
                                                      "date": "2026-01-01",
                                                      "review_due": None, "source": art}
                                                  for i in lesson.CORE_EVIDENCE}},
                "artefacts": {art: {"holds": "everything"}}}

    rr = _route_record("t", route="annex_vi_internal_control")
    both = lesson.readiness(rr, pack(review_due="2026-01-02", source="ART-gone"),
                            {"route_reset": False}, AS_OF)
    assert both["out_of_date"] == ["eu_database_registration"], (
        f"out_of_date came back {both['out_of_date']}")
    assert both["untraceable"] == [], (
        f"untraceable came back {both['untraceable']} — an item that is already out of date "
        "is reported once, in the first bucket it falls into; nobody needs to chase a "
        "pointer on a document they must reissue anyway")
    on_the_day = lesson.readiness(rr, pack(review_due=AS_OF.isoformat()),
                                  {"route_reset": False}, AS_OF)
    assert on_the_day["out_of_date"] == [], (
        "a review due exactly today has not lapsed; the comparison is strictly before as_of")
    issued_today = lesson.readiness(rr, pack(date=AS_OF.isoformat()),
                                    {"route_reset": False}, AS_OF)
    assert issued_today["out_of_date"] == [], "a document issued today is in date"
    future = lesson.readiness(rr, pack(date="2027-01-01"), {"route_reset": False}, AS_OF)
    assert future["out_of_date"] == ["eu_database_registration"], (
        f"got {future['out_of_date']} — a document dated in the future has not been issued")
    undated = lesson.readiness(rr, pack(date=None), {"route_reset": False}, AS_OF)
    assert undated["out_of_date"] == ["eu_database_registration"], (
        "an undated document cannot be shown to be in date")
    blank_source = lesson.readiness(rr, pack(source=""), {"route_reset": False}, AS_OF)
    assert blank_source["untraceable"] == ["eu_database_registration"], (
        "a blank source is not a key of the artefact index, so it is untraceable")
    clean = lesson.readiness(rr, pack(), {"route_reset": False}, AS_OF)
    assert clean["blocking"] == [] and clean["verdict"] == "ready", (
        f"a pack with nothing wrong with it must come back ready; got {clean['verdict']!r} "
        f"blocking {clean['blocking']}. A check that can only fail is not a check")


def test_readiness_verdict_ladder_is_in_the_documented_order():
    dirty = _copy(lesson.PACKS["credit-scorer"])
    clean = _copy(lesson.PACKS["face-match"])
    reset, held = {"route_reset": True}, {"route_reset": False}
    vi = _route_record("t", route="annex_vi_internal_control")

    assert lesson.readiness(vi, clean, held, AS_OF)["verdict"] == "ready"
    assert lesson.readiness(vi, dirty, held, AS_OF)["verdict"] == "blocked"
    got = lesson.readiness(vi, dirty, reset, AS_OF)
    assert got["verdict"] == "reassessment_required", (
        f"a pack that is BOTH incomplete and superseded came back {got['verdict']!r}. "
        "reassessment_required runs first: the missing items are missing from a pack built "
        "for an assessment that no longer covers the system")
    assert got["blocking"], (
        "the blocking list is still reported even when the verdict is reassessment_required "
        "— the verdict decides what to do first, not what to hide")
    for route, want in (("not_high_risk", "not_applicable"), ("none", "not_applicable"),
                        ("undetermined", "route_undetermined")):
        rr = _route_record("t", route=route)
        v = lesson.readiness(rr, dirty, reset, AS_OF)["verdict"]
        assert v == want, (
            f"route {route!r} with a reset and a dirty pack came back {v!r}, expected "
            f"{want!r} — a system with no Article 43 route has nothing to be reassessed for")
    for rec in (lesson.readiness(vi, clean, held, AS_OF),
                lesson.readiness(vi, dirty, reset, AS_OF)):
        assert rec["ready"] is (rec["verdict"] == "ready"), (
            "ready is true for exactly one verdict")
    assert lesson.readiness(vi, clean, held, AS_OF)["route_reset"] is False, (
        "route_reset must report the history it was handed, not a constant")
    assert lesson.readiness(vi, clean, reset, AS_OF)["route_reset"] is True, (
        "route_reset came back False for a history that resets — the readiness report is "
        "what the declaration and the next reader see, so it carries the reset rather than "
        "swallowing it into the verdict")
    truthy = lesson.readiness(vi, clean, {"route_reset": 1}, AS_OF)
    assert truthy["route_reset"] is True and truthy["verdict"] == "reassessment_required", (
        f"a truthy non-bool history gave route_reset={truthy['route_reset']!r} — it is "
        "bool(history['route_reset']), so the key is always a real bool")


def test_readiness_does_not_mutate_the_pack():
    pack = _copy(lesson.PACKS["credit-scorer"])
    before = json.dumps(pack, sort_keys=True)
    lesson.readiness(_route_record("t"), pack, {"route_reset": False}, AS_OF)
    assert json.dumps(pack, sort_keys=True) == before, (
        "readiness edited the pack it was handed — a report that repairs its own input "
        "cannot be re-run")


# --- exercise 7: declaration_of_conformity -----------------------------------------------

def test_declaration_refuses_and_names_the_verdict_that_stopped_it():
    for verdict in ("blocked", "reassessment_required", "not_applicable",
                    "route_undetermined"):
        got = lesson.declaration_of_conformity(
            _route_record("t"), _readiness_report(verdict), _copy(lesson.PACKS["face-match"]),
            AS_OF)
        assert set(got) == {"issued", "refusal_reason", "declaration"}, (
            f"keys were {sorted(got)}")
        assert got["issued"] is False and got["declaration"] is None, (
            f"verdict {verdict!r} produced a declaration. Article 47(4) makes drawing one up "
            "the act of assuming responsibility; there is no draft state")
        assert got["refusal_reason"] == verdict, (
            f"refused with {got['refusal_reason']!r}, expected {verdict!r} — the verdict is "
            "the reason, so a reader learns which of the four problems they have")
    ok = lesson.declaration_of_conformity(_route_record("t"), _readiness_report("ready"),
                                          _copy(lesson.PACKS["face-match"]), AS_OF)
    assert ok["issued"] is True and ok["refusal_reason"] is None and ok["declaration"], (
        "a ready pack earns a declaration; a function that only ever refuses is not a gate")


def test_declaration_annex_v_fields_and_the_ten_year_clock():
    pack = _copy(lesson.PACKS["face-match"])
    got = lesson.declaration_of_conformity(_route_record("face-match"),
                                           _readiness_report("ready"), pack, AS_OF)
    decl = got["declaration"]
    want_keys = {"annex_v_1_system", "annex_v_2_provider", "annex_v_3_sole_responsibility",
                 "annex_v_4_conformity", "annex_v_5_data_protection", "annex_v_6_standards",
                 "annex_v_7_notified_body", "annex_v_8_signature", "keep_until",
                 "record_hash"}
    assert set(decl) == want_keys, f"declaration keys were {sorted(decl)}"
    assert decl["annex_v_3_sole_responsibility"] is True
    assert decl["annex_v_2_provider"] == pack["provider"] and \
        decl["annex_v_2_provider"] is not pack["provider"], (
        "Annex V point 2 carries a COPY of the provider block, not a reference into the pack")
    assert decl["annex_v_6_standards"] == sorted(pack["standards_applied"])
    assert decl["annex_v_5_data_protection"] == pack["data_protection"]
    vi_label = lesson.ROUTE_LABEL["annex_vi_internal_control"]
    for part in ("Regulation (EU) 2024/1689", vi_label):
        assert part in decl["annex_v_4_conformity"], (
            f"Annex V point 4 came back {decl['annex_v_4_conformity']!r} — it states "
            f"conformity with the Regulation AND names the route it was reached by; {part!r} "
            "is missing")
    other = lesson.declaration_of_conformity(
        _route_record("t", route="annex_vii_notified_body", nb=False),
        _readiness_report("ready"), _copy(lesson.PACKS["face-match"]), AS_OF)["declaration"]
    assert lesson.ROUTE_LABEL["annex_vii_notified_body"] in other["annex_v_4_conformity"] \
        and other["annex_v_4_conformity"] != decl["annex_v_4_conformity"], (
        "two systems on two different routes got the same Annex V point 4 statement; the "
        "route the declaration was reached by is part of what it states")
    sig = decl["annex_v_8_signature"]
    assert set(sig) == {"place", "date", "name", "function"}, f"signature keys {sorted(sig)}"
    assert sig["date"] == AS_OF.isoformat() and sig["name"] == pack["signatory"]["name"]
    for part in (pack["system_name"], pack["system_type"], "face-match"):
        assert part in decl["annex_v_1_system"], (
            f"Annex V point 1 wants name, type AND an unambiguous reference; {part!r} is "
            f"missing from {decl['annex_v_1_system']!r}")
    placed = date.fromisoformat(pack["placed_on_market"])
    assert decl["keep_until"] == placed.replace(year=placed.year + 10).isoformat(), (
        f"keep_until came back {decl['keep_until']!r} — Article 47(1)'s ten years run from "
        "the placing on the market, not from as_of and not from the signature date")
    leap = _copy(lesson.PACKS["face-match"])
    leap["placed_on_market"] = "2028-02-29"
    leaped = lesson.declaration_of_conformity(_route_record("t"), _readiness_report("ready"),
                                              leap, AS_OF)["declaration"]
    assert leaped["keep_until"] == "2038-02-28", (
        f"29 February plus ten years came back {leaped['keep_until']!r} — plus_years() is "
        "given to you and clamps it rather than raising")
    assert lesson.seal(decl)["record_hash"] == decl["record_hash"], (
        "the declaration must reseal to the same digest: seal over the record WITHOUT "
        "record_hash in it, exactly as seal() does")


def test_declaration_notified_body_entry_follows_the_route():
    pack = _copy(lesson.PACKS["border-biometrics"])
    quiet = lesson.declaration_of_conformity(
        _route_record("border-biometrics", route="annex_vii_notified_body", nb=False),
        _readiness_report("ready"), pack, AS_OF)["declaration"]
    assert quiet["annex_v_7_notified_body"] is None, (
        "no notified body required means Annex V point 7 is empty, even when the pack names "
        "a body — read notified_body_required, not pack['notified_body']")
    loud = lesson.declaration_of_conformity(
        _route_record("border-biometrics", route="annex_vii_notified_body", nb=True,
                      role="market_surveillance_authority"),
        _readiness_report("ready"), pack, AS_OF)["declaration"]
    entry = loud["annex_v_7_notified_body"]
    assert set(entry) == {"name", "identification_number", "certificate"}, (
        f"annex_v_7 keys were {sorted(entry)}")
    assert entry["name"] == pack["notified_body"]["name"]
    assert entry["identification_number"] == pack["notified_body"]["identification_number"]
    cert_item = lesson.CERTIFICATE_ITEM["annex_vii_notified_body"]
    assert entry["certificate"] == pack["evidence"][cert_item]["document"], (
        f"certificate came back {entry['certificate']!r} — Annex VII's route cites the "
        f"{cert_item} in the pack, not the QMS approval and not the technical documentation")
    annex_i = _copy(lesson.PACKS["lift-door-sensor"])
    embedded = lesson.declaration_of_conformity(
        _route_record("lift-door-sensor", route="annex_i_product_procedure", nb=True,
                      role="notified_body"),
        _readiness_report("ready"), annex_i, AS_OF)["declaration"]
    other_item = lesson.CERTIFICATE_ITEM["annex_i_product_procedure"]
    assert embedded["annex_v_7_notified_body"]["certificate"] == \
        annex_i["evidence"][other_item]["document"], (
        "the Annex I route cites its own integrated product assessment — the certificate "
        "item is looked up per route, not hard-wired to the Annex VII one")


# --- exercise 8: ce_marking_record --------------------------------------------------------

def _declaration_result(issued=True, nb_number=None):
    if not issued:
        return {"issued": False, "refusal_reason": "blocked", "declaration": None}
    body = None if nb_number is None else {"name": "Body", "identification_number": nb_number,
                                           "certificate": "CERT-1"}
    decl = lesson.seal({"annex_v_1_system": "s", "annex_v_2_provider": {},
                        "annex_v_3_sole_responsibility": True, "annex_v_4_conformity": "c",
                        "annex_v_5_data_protection": None, "annex_v_6_standards": [],
                        "annex_v_7_notified_body": body,
                        "annex_v_8_signature": {"place": "p", "date": "2026-09-22",
                                                "name": "n", "function": "f"},
                        "keep_until": "2036-01-01"})
    return {"issued": True, "refusal_reason": None, "declaration": decl}


def test_ce_marking_refuses_without_a_declaration():
    for system in ({"provided_digitally": False, "surface_permits_marking": True},
                   {"provided_digitally": True, "machine_readable_access": True},
                   {"provided_digitally": False, "surface_permits_marking": False},
                   # The case that pins the ORDER of the two refusals: this system would also
                   # fail Article 48(2), and the declaration is still the reason it is told.
                   {"provided_digitally": True, "machine_readable_access": False}):
        got = lesson.ce_marking_record(_route_record("t"), _declaration_result(issued=False),
                                       system)
        assert set(got) == {"affixed", "refusal_reason", "marking"}, f"keys {sorted(got)}"
        assert got["affixed"] is False and got["marking"] is None, (
            "the marking attests to a declaration; without one there is nothing to attest to")
        assert got["refusal_reason"] == "no_declaration_of_conformity", (
            f"refused with {got['refusal_reason']!r} for {system} — the missing declaration is "
            "the reason, and it is checked BEFORE anything about the system's surface: a "
            "provider with no declaration does not need to hear about its marking options")


def test_ce_marking_form_and_basis_over_the_whole_truth_table():
    wrong = []
    for digital in (False, True):
        for machine in (False, True):
            for surface in (False, True):
                system = {"provided_digitally": digital, "machine_readable_access": machine,
                          "surface_permits_marking": surface}
                got = lesson.ce_marking_record(_route_record("t"), _declaration_result(),
                                               system)
                if digital and not machine:
                    want = (False, "digital_marking_not_accessible", None, None)
                elif digital:
                    want = (True, None, "digital", "Article 48(2)")
                elif surface:
                    want = (True, None, "physical", "Article 48(3)")
                else:
                    want = (True, None, "accompanying_documentation", "Article 48(3)")
                form = got["marking"]["form"] if got["marking"] else None
                basis = got["marking"]["basis"] if got["marking"] else None
                if (got["affixed"], got["refusal_reason"], form, basis) != want:
                    wrong.append((system, (got["affixed"], got["refusal_reason"], form,
                                           basis), want))
    assert not wrong, (
        f"{len(wrong)} of 8 combinations disagree; first: {wrong[0][0]} gave {wrong[0][1]}, "
        f"expected {wrong[0][2]}. Article 48(2) permits the digital form ONLY if it can "
        "easily be accessed, and 48(3) sends a marking that will not fit onto the "
        "accompanying documentation")
    only_surface = lesson.ce_marking_record(
        _route_record("t"), _declaration_result(),
        {"provided_digitally": True, "machine_readable_access": True,
         "surface_permits_marking": False})
    assert only_surface["marking"]["form"] == "digital", (
        "a digitally provided system with an accessible marking is marked digitally; the "
        "surface question does not arise for it")


def test_ce_marking_number_is_read_off_the_declaration():
    system = {"provided_digitally": False, "surface_permits_marking": True}
    none = lesson.ce_marking_record(_route_record("t", nb=False), _declaration_result(),
                                    system)
    assert none["marking"]["notified_body_number"] is None, (
        f"got {none['marking']['notified_body_number']!r} — with no body in Annex V point 7, "
        "Article 48(4)'s 'where applicable' adds nothing")
    numbered = _declaration_result(nb_number="0621")
    got = lesson.ce_marking_record(_route_record("t", nb=True, role="notified_body"),
                                   numbered, system)
    assert got["marking"]["notified_body_number"] == "0621", (
        f"got {got['marking']['notified_body_number']!r} — Article 48(4) names the body "
        "responsible for the Article 43 procedure, which is the body the declaration "
        "already cites in Annex V point 7")
    assert got["marking"]["declaration_hash"] == numbered["declaration"]["record_hash"], (
        "the marking cites the declaration it attests to, by its seal, so the pair cannot be "
        "separated and recombined")
    assert set(got["marking"]) == {"form", "notified_body_number", "declaration_hash",
                                   "basis"}, f"marking keys were {sorted(got['marking'])}"


# --- the trap, end to end -----------------------------------------------------------------

def test_end_to_end_cv_ranker_is_refused_and_a_clean_twin_is_not():
    """MODULES.md's trap: a substantial modification resets the route.

    Two-sided on purpose. cv-ranker's pack is complete, in date and traceable, and its
    declaration must still be refused. A twin whose ONLY difference is that the
    purpose-changing modification never happened must get one — so refusing everything
    scores nothing here either.
    """
    record, ctx = lesson.RECORDS["cv-ranker"], lesson.CONTEXTS["cv-ranker"]
    pack, plan = lesson.PACKS["cv-ranker"], lesson.PLANS["cv-ranker"]

    def chain(log):
        route = lesson.assessment_route(record, ctx)
        history = lesson.modification_history(_copy(log), _copy(plan))
        ready = lesson.readiness(route, _copy(pack), history, AS_OF)
        decl = lesson.declaration_of_conformity(route, ready, _copy(pack), AS_OF)
        marking = lesson.ce_marking_record(route, decl, lesson.SYSTEMS["cv-ranker"])
        return route, history, ready, decl, marking

    route, history, ready, decl, marking = chain(lesson.CHANGE_LOGS["cv-ranker"])
    assert ready["blocking"] == [] and ready["coverage"] == 1.0, (
        f"cv-ranker's pack has nothing wrong with it; got blocking {ready['blocking']} at "
        f"coverage {ready['coverage']}")
    assert history["since_assessment"] == ["CHG-63"], (
        f"got {history['since_assessment']} — CHG-63 changed the intended purpose after the "
        f"{plan['assessed_on']} assessment, and CHG-58 predates it")
    assert ready["verdict"] == "reassessment_required" and decl["issued"] is False, (
        f"a complete pack and a superseded assessment produced issued={decl['issued']} at "
        f"verdict {ready['verdict']!r}. A substantial modification sends the system back "
        "through assessment; the declaration may not be drawn up in the meantime")
    assert decl["refusal_reason"] == "reassessment_required"
    assert marking["affixed"] is False and \
        marking["refusal_reason"] == "no_declaration_of_conformity"

    twin = [c for c in lesson.CHANGE_LOGS["cv-ranker"] if c["change_id"] != "CHG-63"]
    _, hist2, ready2, decl2, marking2 = chain(twin)
    assert hist2["route_reset"] is False and ready2["verdict"] == "ready", (
        f"with the purpose change removed the route holds; got {ready2['verdict']!r}. The "
        "refusal has to come from the modification, not from the system's name")
    assert decl2["issued"] is True and marking2["affixed"] is True, (
        "the twin earns its declaration and its marking — a chain that refuses everything "
        "is as useless as one that refuses nothing")
    assert route["route"] == "annex_vi_internal_control"


RUBRIC = [
    (test_annex_iii_point_matches_an_independent_derivation, 4,
     "walk question_trail for the first q3_annex_iii_area step and read its answer"),
    (test_annex_iii_point_rejects_a_point_outside_annex_iii, 3,
     "the matched number must be in ANNEX_III_POINTS; 'point 12' is not point 1"),
    (test_annex_iii_point_reads_the_q3_step_and_nothing_else, 3,
     "only the q3 step, only the first one, and a falsy answer is None"),
    (test_standards_gap_matches_an_independent_derivation, 4,
     "(a) needs BOTH absent; (b) and (d) need the standard to exist; (c) needs the spec to"),
    (test_standards_gap_collects_every_gap_sorted, 3,
     "collect every gap that fires, then sort; do not return the first one"),
    (test_standards_gap_treats_a_missing_key_as_falsy, 3,
     "read every field with .get(), and only 'full' closes gap (b)"),
    (test_route_matches_an_independent_derivation_on_every_record, 5,
     "tier first, then the Annex III point, then the gap — and cite the article"),
    (test_route_points_2_to_8_ignore_the_standards_gap, 6,
     "Article 43(2) has no standards test: internal control, no body, no choice, no gaps"),
    (test_route_point_1_choice_is_only_a_choice_without_a_gap, 4,
     "consult chosen_route only when standards_gap is empty; fall back to Annex VI"),
    (test_route_undetermined_never_falls_back_to_a_procedure, 4,
     "an unreadable point is 'undetermined', not a default route"),
    (test_route_notified_body_role_and_the_annex_i_case, 4,
     "role is None unless a body is required; AUTHORITY_DEPLOYERS names the MSA"),
    (test_substantial_matches_an_independent_derivation, 5,
     "before-placing, then intended purpose, then the carve-out, then compliance"),
    (test_substantial_pre_determined_needs_the_annex_iv_description, 4,
     "Article 43(4) needs pre-determined AND documented in Annex IV point 2(f)"),
    (test_substantial_intended_purpose_runs_before_the_carve_out, 4,
     "test modifies_intended_purpose before consulting the pre-determined list"),
    (test_substantial_before_placing_is_strict_and_optional, 4,
     "strictly earlier than placed_on_market, and a missing placing date cannot fire it"),
    (test_history_matches_an_independent_derivation, 4,
     "one verdict per change; last_substantial_on is the latest, not the first"),
    (test_history_reset_counts_only_uncovered_changes, 5,
     "route_reset is a substantial change dated on or after assessed_on, not any of them"),
    (test_history_orders_the_log_without_mutating_it, 3,
     "sort a COPY by (on, change_id)"),
    (test_history_on_an_empty_and_an_all_quiet_log, 3,
     "an empty log has no last_substantial_on and does not reset"),
    (test_readiness_buckets_match_an_independent_derivation, 5,
     "missing from the checker; then out_of_date; then untraceable; blocking is the union"),
    (test_readiness_uses_the_completeness_checker_it_was_given, 4,
     "take present, missing and coverage from completeness_report, not a second checker"),
    (test_readiness_bucket_precedence_and_date_boundaries, 5,
     "each item lands in ONE bucket; review_due == as_of has not lapsed"),
    (test_readiness_verdict_ladder_is_in_the_documented_order, 5,
     "not_applicable, route_undetermined, reassessment_required, blocked, ready"),
    (test_readiness_does_not_mutate_the_pack, 2,
     "read the pack; never write to it"),
    (test_declaration_refuses_and_names_the_verdict_that_stopped_it, 5,
     "refuse unless the verdict is 'ready', and report the verdict as the reason"),
    (test_declaration_annex_v_fields_and_the_ten_year_clock, 5,
     "Annex V's eight points plus keep_until, measured from placed_on_market"),
    (test_declaration_notified_body_entry_follows_the_route, 4,
     "point 7 tracks notified_body_required, and the certificate comes from CERTIFICATE_ITEM"),
    (test_ce_marking_refuses_without_a_declaration, 3,
     "check declaration_result['issued'] before anything about the system"),
    (test_ce_marking_form_and_basis_over_the_whole_truth_table, 5,
     "digital only if accessible; otherwise physical, otherwise the documentation"),
    (test_ce_marking_number_is_read_off_the_declaration, 4,
     "the number lives in the declaration's Annex V point 7, not in the pack"),
    (test_end_to_end_cv_ranker_is_refused_and_a_clean_twin_is_not, 6,
     "a complete pack plus an uncovered substantial modification means no declaration"),
]
