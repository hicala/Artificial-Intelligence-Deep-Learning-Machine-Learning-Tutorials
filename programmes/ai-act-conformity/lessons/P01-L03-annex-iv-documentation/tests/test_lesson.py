"""Autograder rubric for P01-L03. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Every expectation is re-derived HERE, by `_ref_value`, `_ref_sourced` and `_ref_status`,
  which are independent re-implementations of the three things the student writes. Nothing in
  this file asks the student's own code what the right answer is. If the re-derivation and
  the lesson ever disagree, one of them is wrong and the rubric says so.
* The staleness tests never call the student's `render_document`. They build a document in
  its documented shape from `_document_of`, so exercise 6 can score full marks while
  exercise 5 is still a stub. Partial credit that depends on a previous exercise is not
  partial credit.
* `test_is_sourced_keeps_measurements` is what stops a truthiness test from scoring. 0,
  0.0 and False are results a pipeline measured; a checker that calls them blank reports a
  fully sourced section as empty and sends an engineer to look for a bug that is not there.
* `test_is_sourced_recurses_into_containers` is the other half: a non-empty list of blank
  template rows is truthy and sources nothing. Between them these two cases reject both of
  the obvious wrong rules.
* `test_check_section_any_of_paths_are_not_fields` exists because the tempting simplification
  is to fold `any_of` paths into `requires` and report the unsatisfied one as missing. Annex
  IV point 7 is genuinely an either/or; a report that nags about the branch nobody took is a
  report people learn to skip.
* `test_staleness_catches_the_hand_edit` and `test_staleness_ignores_a_rerun` are a matched
  pair. Either one alone can be passed by a wrong rule — the first by flagging everything,
  the second by flagging nothing — so both are graded, and a report keyed on the modification
  time fails exactly one of them in each direction.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

SOURCES = lesson.SOURCES
AS_FILED = lesson.SOURCES_AS_FILED
ANNEX_IV = lesson.ANNEX_IV
HEXDIGITS = set("0123456789abcdef")


class _Absent:
    """The rubric's own absence marker, deliberately not the lesson's."""

    def __repr__(self) -> str:
        return "<absent:rubric>"


_ABSENT = _Absent()


# --- independent re-derivations --------------------------------------------------------

def _ref_value(sources: dict, path: str):
    """What `field_value` should return, derived here rather than asked of the lesson."""
    parts = path.split(".")
    if len(parts) < 2 or parts[0] not in sources:
        return _ABSENT
    node = sources[parts[0]].payload
    for key in parts[1:]:
        if not isinstance(node, dict) or key not in node:
            return _ABSENT
        node = node[key]
    return node


def _ref_sourced(value) -> bool:
    """What `is_sourced` should return. Emptiness, not falsiness, and it recurses."""
    if value is _ABSENT or value is lesson.MISSING or value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, dict):
        return any(_ref_sourced(inner) for inner in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_ref_sourced(inner) for inner in value)
    return True


def _ref_status(spec, sources: dict) -> dict:
    """What `check_section` should report, as a plain dict."""
    missing, empty, unmet, artefacts = [], [], [], set()
    for path in spec.requires:
        value = _ref_value(sources, path)
        if value is _ABSENT:
            missing.append(path)
        elif not _ref_sourced(value):
            empty.append(path)
        else:
            artefacts.add(path.split(".")[0])
    for group in spec.any_of:
        hit = [p for p in group if _ref_sourced(_ref_value(sources, p))]
        if hit:
            artefacts.update(p.split(".")[0] for p in hit)
        else:
            unmet.append(tuple(group))
    return {"complete": not (missing or empty or unmet),
            "artefacts": tuple(sorted(artefacts)), "missing": tuple(missing),
            "empty": tuple(empty), "unmet_choices": tuple(unmet)}


def _ref_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ref_stamp(artefact) -> dict:
    return {"artefact": artefact.name, "run_id": artefact.run_id,
            "generated_at": artefact.generated_at,
            "content_sha256": _ref_digest(artefact.payload)}


def _document_of(rows) -> dict:
    """A document in render_document's documented shape, built without the student's code.

    `rows` is [(section_number, [artefact, ...]), ...]; each artefact is stamped as it stands.
    """
    return {"sections": [{"number": number, "title": "", "basis": "", "fields": {},
                          "provenance": [_ref_stamp(a) for a in artefacts]}
                         for number, artefacts in rows],
            "omitted": [], "complete": True}


def _spec(number):
    return next(s for s in ANNEX_IV if s.number == number)


# --- exercise 1: field_value -----------------------------------------------------------

def test_field_value_walks_the_path():
    for path in ("metrics.primary_metric", "model_registry.system.intended_purpose",
                 "model_registry.provider.name", "dataset_manifest.datasets",
                 "event_log_design.retention_minimum_days", "standards.other_solutions",
                 "risk_register.residual_risks", "pmm_plan.review_cadence_days"):
        want = _ref_value(SOURCES, path)
        got = lesson.field_value(SOURCES, path)
        assert got == want, (
            f"field_value(SOURCES, {path!r}) returned {got!r}, expected {want!r} — the first "
            "segment names the artefact, the rest walks its payload"
        )


def test_field_value_reports_absence_without_raising():
    for path in ("declaration.signed_copy", "declaration.signed_on",
                 "control_design.human_oversight_measures", "metrics.nope",
                 "model_registry.system.nope", "metrics.primary_metric.length",
                 "pmm_plan.review_cadence_days.days", "metrics", "", "metrics.",
                 "nosuchartefact.anything"):
        got = lesson.field_value(SOURCES, path)
        assert got is lesson.MISSING, (
            f"field_value(SOURCES, {path!r}) returned {got!r} — every one of these paths is "
            "absent, dead halfway, or not a field at all, and each must come back MISSING "
            "rather than raising or returning None"
        )


def test_field_value_does_not_confuse_falsy_with_absent():
    for path, want in (("metrics.open_severe_findings", 0),
                       ("dataset_manifest.labelling_procedure", ""),
                       ("standards.harmonised_applied", [])):
        got = lesson.field_value(SOURCES, path)
        assert got is not lesson.MISSING, (
            f"field_value(SOURCES, {path!r}) returned MISSING, but that field is PRESENT and "
            f"carries {want!r}. Folding a blank into MISSING destroys the difference between "
            "'nobody produced this' and 'the pipeline produced nothing', which is the "
            "distinction the whole document generator is built on"
        )
        assert got == want, f"field_value(SOURCES, {path!r}) returned {got!r}, expected {want!r}"


# --- exercise 2: is_sourced ------------------------------------------------------------

def test_is_sourced_rejects_absence_and_blankness():
    for value in (lesson.MISSING, None, "", " ", "\t\n ", [], {}, (), set()):
        got = lesson.is_sourced(value)
        assert got is False, (
            f"is_sourced({value!r}) returned {got!r} — an absent value, a blank string and an "
            "empty container all source nothing"
        )


def test_is_sourced_keeps_measurements():
    for value in (0, 0.0, False, "0", "false", 0.871, [0], (False,), {"a": 0},
                  ["x"], {"k": "v"}, 183):
        got = lesson.is_sourced(value)
        assert got is True, (
            f"is_sourced({value!r}) returned {got!r} — 0, 0.0 and False are things a pipeline "
            "MEASURED. A truthiness test throws them away and reports a fully sourced section "
            "as blank"
        )


def test_is_sourced_recurses_into_containers():
    blank_row = [{"risk": "", "accepted_by": "", "review_due": ""}]
    assert lesson.is_sourced(blank_row) is False, (
        "a non-empty list of entirely blank rows is truthy and sources nothing — this is the "
        "pasted template row that turns a pack green. Recurse instead of testing len()"
    )
    assert lesson.is_sourced([{"risk": "", "accepted_by": "u:ehsan"}]) is True, (
        "one real value anywhere inside makes the container sourced: the rule is ANY, not ALL"
    )
    assert lesson.is_sourced({"a": {"b": [None, "", {}]}}) is False, (
        "recursion goes all the way down, through dicts and lists alike"
    )
    assert lesson.is_sourced({"a": {"b": [None, "", {"c": 0}]}}) is True, (
        "a measured 0 three levels down still sources the container"
    )
    assert lesson.is_sourced(["", "   ", []]) is False, (
        "a list of blanks is a blank"
    )


# --- exercise 3: check_section ---------------------------------------------------------

def test_check_section_shape_and_a_complete_section():
    status = lesson.check_section(_spec("1"), SOURCES)
    assert isinstance(status, lesson.SectionStatus), (
        f"return a SectionStatus, got {type(status).__name__}"
    )
    want = _ref_status(_spec("1"), SOURCES)
    assert status.number == "1" and status.title == _spec("1").title, (
        "number and title come straight off the spec"
    )
    assert status.complete is True, f"section 1 is fully sourced; you reported {status!r}"
    assert status.artefacts == want["artefacts"] == ("model_registry",), (
        f"artefacts is the sorted DISTINCT set of artefact names behind the SOURCED paths; "
        f"you reported {status.artefacts!r}"
    )
    for field in ("artefacts", "missing", "empty", "unmet_choices"):
        assert isinstance(getattr(status, field), tuple), (
            f"{field} must be a tuple, got {type(getattr(status, field)).__name__}"
        )
    # Every artefact in SOURCES, required in a deliberately scrambled order. This is the only
    # place the ORDER of `artefacts` is pinned across more than two names, and it has to be:
    # on this schema the one multi-source section reads two artefacts that happen to be listed
    # alphabetically already, so an insertion-ordered answer agrees with a sorted one there,
    # and a bare `tuple(set)` answer agrees about half the time because CPython randomises
    # string hashes at every interpreter start. Ten names out of order kill both.
    wide = lesson.SectionSpec(
        "W", "every artefact at once",
        requires=("standards.other_solutions", "model_registry.provider.name",
                  "change_log.assessment_of_substantial_modification",
                  "metrics.primary_metric", "pmm_plan.plan_reference",
                  "dataset_manifest.provenance", "event_log_design.chain_algorithm",
                  "validation_report.signed_by", "risk_register.method",
                  "control_design.input_specifications"))
    spread = lesson.check_section(wide, SOURCES)
    assert spread.complete is True, (
        f"all ten of those paths are sourced, so the section is complete; you reported "
        f"{spread!r}"
    )
    assert (spread.artefacts == _ref_status(wide, SOURCES)["artefacts"]
            == tuple(sorted(SOURCES))), (
        f"artefacts must come back SORTED — not in the order the paths happen to be listed, "
        f"and not in a set's own iteration order. You reported {spread.artefacts!r}. An "
        "unsorted answer re-orders the provenance table of every multi-source section between "
        "one interpreter run and the next, so two builds of the same document differ for no "
        "reason a reader can see"
    )


def test_check_section_separates_missing_from_empty():
    for number in ("2", "3", "5"):
        spec = _spec(number)
        want = _ref_status(spec, SOURCES)
        status = lesson.check_section(spec, SOURCES)
        assert status.missing == want["missing"], (
            f"section {number}: missing={status.missing!r}, expected {want['missing']!r} — a "
            "path is MISSING only when it is absent from the artefact"
        )
        assert status.empty == want["empty"], (
            f"section {number}: empty={status.empty!r}, expected {want['empty']!r} — a path "
            "that is present and carries nothing is EMPTY, not missing. 'add the field' and "
            "'the pipeline produced a blank' go to two different teams"
        )
        assert status.complete is False, f"section {number} cannot be complete"
    assert lesson.check_section(_spec("4"), SOURCES).complete is True, (
        "section 4 is complete: metrics.open_severe_findings is 0, a measurement, and "
        "must not be reported as blank"
    )


def test_check_section_handles_an_absent_artefact():
    spec = _spec("8")
    want = _ref_status(spec, SOURCES)
    status = lesson.check_section(spec, SOURCES)
    assert status.missing == want["missing"] == ("declaration.signed_copy",
                                                 "declaration.signed_on"), (
        f"with no `declaration` artefact in SOURCES, EVERY path into it is missing, in spec "
        f"order; you reported {status.missing!r}"
    )
    assert status.empty == () and status.unmet_choices == (), (
        f"nothing about section 8 is blank or an unmet choice; you reported empty="
        f"{status.empty!r} unmet_choices={status.unmet_choices!r}"
    )
    assert status.artefacts == (), (
        f"no path was sourced, so no artefact contributed; you reported {status.artefacts!r}"
    )
    stripped = {k: v for k, v in SOURCES.items() if k != "metrics"}
    lost = lesson.check_section(_spec("4"), stripped)
    assert lost.missing == _spec("4").requires and lost.artefacts == (), (
        "remove the metrics artefact and all seven of section 4's paths go missing; you "
        f"reported {lost.missing!r}"
    )


def test_check_section_any_of_paths_are_not_fields():
    spec = _spec("7")
    want = _ref_status(spec, SOURCES)
    status = lesson.check_section(spec, SOURCES)
    assert status.complete is True and status.unmet_choices == (), (
        "section 7's either/or is satisfied by standards.other_solutions even though "
        f"standards.harmonised_applied is an empty list; you reported {status!r}"
    )
    assert status.missing == () and status.empty == (), (
        f"an any_of path that is empty is NOT reported as a missing or empty field — only "
        f"the group verdict matters. You reported missing={status.missing!r} "
        f"empty={status.empty!r}, which nags a provider who did nothing wrong"
    )
    assert status.artefacts == want["artefacts"] == ("standards",), (
        "the artefact behind the sourced branch still counts as a source"
    )
    # One branch ABSENT from the artefact, the other present and blank. The group is unmet
    # either way, and neither kind of failure may be reported as a missing or an empty FIELD —
    # a checker that reports the absent branch is nagging a provider who applied no harmonised
    # standards and said so, which is exactly what point 7 permits.
    dark = copy.deepcopy(dict(SOURCES))
    dark["standards"] = SOURCES["standards"]._replace(payload={"other_solutions": "   "})
    unmet = lesson.check_section(spec, dark)
    assert unmet.complete is False and unmet.unmet_choices == (
        ("standards.harmonised_applied", "standards.other_solutions"),), (
        f"with NEITHER branch sourced the whole group is unmet, reported as a tuple of its "
        f"paths; you reported {unmet.unmet_choices!r}"
    )
    assert unmet.missing == () and unmet.empty == (), (
        f"even when unmet, an any_of path never lands in missing or empty — including the "
        f"branch that is ABSENT from the artefact altogether. You reported "
        f"missing={unmet.missing!r} empty={unmet.empty!r}"
    )
    assert unmet.artefacts == (), (
        f"nothing in the group was sourced, so standards contributed no source; you reported "
        f"{unmet.artefacts!r}"
    )
    # The same group, but with the first branch PRESENT and empty rather than absent. An
    # `all()` written where `any()` belongs calls the empty list sourced — vacuously true over
    # no elements — and declares point 7 satisfied by a field that says nothing.
    blank = copy.deepcopy(dict(SOURCES))
    blank["standards"] = SOURCES["standards"]._replace(
        payload={"harmonised_applied": [], "other_solutions": "   "})
    still = lesson.check_section(spec, blank)
    assert still.complete is False and still.unmet_choices == (
        ("standards.harmonised_applied", "standards.other_solutions"),), (
        f"an empty list of harmonised standards sources nothing, and neither does a "
        f"whitespace-only description, so the group is still unmet; you reported {still!r}"
    )
    assert still.missing == () and still.empty == () and still.artefacts == (), (
        f"still no field-level report and still no contributing artefact; you reported "
        f"missing={still.missing!r} empty={still.empty!r} artefacts={still.artefacts!r}"
    )


# --- exercise 4: provenance_stamp ------------------------------------------------------

def test_provenance_stamp_shape_and_values():
    for name in ("metrics", "model_registry", "standards"):
        stamp = lesson.provenance_stamp(SOURCES[name])
        want = _ref_stamp(SOURCES[name])
        assert isinstance(stamp, dict) and set(stamp) == set(want), (
            f"a stamp carries exactly {sorted(want)}; yours has {sorted(stamp)}"
        )
        assert stamp["artefact"] == want["artefact"] and stamp["run_id"] == want["run_id"], (
            f"{name}: name and run come straight off the artefact; you produced {stamp!r}"
        )
        assert stamp["generated_at"] == want["generated_at"], (
            f"{name}: generated_at is the artefact's own timestamp, not today's"
        )
        digest = stamp["content_sha256"]
        assert isinstance(digest, str) and len(digest) == 64 and set(digest) <= HEXDIGITS, (
            f"{name}: content_sha256 is a 64-character lowercase hex digest, got {digest!r}"
        )
        assert digest == want["content_sha256"], (
            f"{name}: the digest must be content_hash() of the artefact's PAYLOAD; yours "
            "does not match a digest taken over the payload alone"
        )


def test_provenance_stamp_digest_covers_the_payload_only():
    base = SOURCES["metrics"]
    digest = lesson.provenance_stamp(base)["content_sha256"]
    rerun = base._replace(run_id="met-2026-09-13", generated_at="2026-09-13T06:00:00Z")
    assert lesson.provenance_stamp(rerun)["content_sha256"] == digest, (
        "a re-run that produced byte-identical output must carry the same content digest. "
        "Yours moved, so the run id or the timestamp is inside the hash — and a staleness "
        "report built on it flags all nine sections every morning until somebody turns it off"
    )
    renamed = base._replace(name="metrics_2025q4")
    assert lesson.provenance_stamp(renamed)["content_sha256"] == digest, (
        "renaming the artefact does not change its content, so the digest must not move"
    )
    edited = base._replace(payload={**base.payload, "value": 0.842})
    assert lesson.provenance_stamp(edited)["content_sha256"] != digest, (
        "a changed payload must change the digest, or no drift is ever detectable"
    )
    reordered = base._replace(payload=dict(reversed(list(base.payload.items()))))
    assert lesson.provenance_stamp(reordered)["content_sha256"] == digest, (
        "the same payload built in a different key order is the same payload — that is what "
        "the canonical serialisation you were given is for"
    )


# --- exercise 5: render_document -------------------------------------------------------

def test_render_emits_only_the_sourced_sections():
    doc = lesson.render_document(ANNEX_IV, SOURCES)
    assert isinstance(doc, dict) and set(doc) == {"sections", "omitted", "complete"}, (
        f"the document has exactly those three keys; yours has {sorted(doc)}"
    )
    want_emitted = [s.number for s in ANNEX_IV if _ref_status(s, SOURCES)["complete"]]
    want_omitted = [s.number for s in ANNEX_IV if not _ref_status(s, SOURCES)["complete"]]
    assert [s["number"] for s in doc["sections"]] == want_emitted == ["1", "4", "6", "7", "9"], (
        f"emit exactly the sourced sections, in schema order; you emitted "
        f"{[s['number'] for s in doc['sections']]}"
    )
    assert [s["number"] for s in doc["omitted"]] == want_omitted == ["2", "3", "5", "8"], (
        f"refuse the rest, in schema order; you omitted "
        f"{[s['number'] for s in doc['omitted']]}"
    )
    assert doc["complete"] is False, (
        "four sections were refused, so the document is not complete. A generator that "
        "reports complete anyway produces the green pack this module exists to prevent"
    )
    for section in doc["sections"]:
        assert set(section) == {"number", "title", "basis", "fields", "provenance"}, (
            f"section {section.get('number')!r} has keys {sorted(section)}"
        )
    whole = lesson.render_document(tuple(s for s in ANNEX_IV if s.number in {"1", "4"}),
                                   SOURCES)
    assert whole["complete"] is True and whole["omitted"] == [], (
        "a spec list whose every section is sourced produces a complete document"
    )


def test_render_never_emits_an_unsourced_value():
    doc = lesson.render_document(ANNEX_IV, SOURCES)
    for section in doc["sections"]:
        spec = _spec(section["number"])
        want_paths = [p for p in spec.requires]
        want_paths += [p for g in spec.any_of for p in g
                       if _ref_sourced(_ref_value(SOURCES, p))]
        assert list(section["fields"]) == want_paths, (
            f"section {section['number']}: fields should be {want_paths}, got "
            f"{list(section['fields'])} — every required path plus the sourced side of an "
            "any_of group, in that order"
        )
        for path, value in section["fields"].items():
            assert _ref_sourced(value), (
                f"section {section['number']} emitted {path} as {value!r}. An unsourced value "
                "blocks the section; it is never rendered as a placeholder"
            )
            assert value == _ref_value(SOURCES, path), (
                f"section {section['number']}: {path} was emitted as {value!r} rather than "
                "the artefact's own value"
            )
    four = next(s for s in doc["sections"] if s["number"] == "4")
    assert four["fields"]["metrics.open_severe_findings"] == 0, (
        "the measured zero belongs in the document; dropping it for being falsy loses the "
        "one number a reviewer most wants to see"
    )
    seven = next(s for s in doc["sections"] if s["number"] == "7")
    assert list(seven["fields"]) == ["standards.other_solutions"], (
        f"section 7 emits only the sourced side of its either/or; you emitted "
        f"{list(seven['fields'])}"
    )


def test_render_explains_every_refusal():
    doc = lesson.render_document(ANNEX_IV, SOURCES)
    for section in doc["omitted"]:
        assert set(section) == {"number", "title", "basis", "reason", "missing", "empty",
                                "unmet_choices"}, (
            f"omitted section {section.get('number')!r} has keys {sorted(section)}"
        )
        want = _ref_status(_spec(section["number"]), SOURCES)
        assert tuple(section["missing"]) == want["missing"], (
            f"section {section['number']}: missing={section['missing']!r}, expected "
            f"{want['missing']!r}"
        )
        assert tuple(section["empty"]) == want["empty"], (
            f"section {section['number']}: empty={section['empty']!r}, expected "
            f"{want['empty']!r}"
        )
        assert tuple(tuple(g) for g in section["unmet_choices"]) == want["unmet_choices"], (
            f"section {section['number']}: unmet_choices={section['unmet_choices']!r}"
        )
        reason = section["reason"]
        assert isinstance(reason, str) and reason.strip(), (
            f"section {section['number']} was refused with no reason — a refusal nobody can "
            "act on is just a hole in the document"
        )
        for path in list(want["missing"]) + list(want["empty"]):
            assert path in reason, (
                f"section {section['number']}'s reason does not name {path!r}. The reason "
                "string alone has to tell an engineer what to go and fix; yours reads "
                f"{reason!r}"
            )
    dark = dict(SOURCES)
    dark["standards"] = SOURCES["standards"]._replace(
        payload={"harmonised_applied": [], "other_solutions": ""})
    refused = next(s for s in lesson.render_document(ANNEX_IV, dark)["omitted"]
                   if s["number"] == "7")
    for path in ("standards.harmonised_applied", "standards.other_solutions"):
        assert path in refused["reason"], (
            f"an unmet either/or must name BOTH branches, so the reader can satisfy either; "
            f"yours reads {refused['reason']!r}"
        )


def test_render_stamps_every_emitted_section():
    doc = lesson.render_document(ANNEX_IV, SOURCES)
    for section in doc["sections"]:
        want = _ref_status(_spec(section["number"]), SOURCES)
        stamps = section["provenance"]
        assert [s["artefact"] for s in stamps] == list(want["artefacts"]), (
            f"section {section['number']}: stamped {[s['artefact'] for s in stamps]}, "
            f"expected {list(want['artefacts'])} — one stamp per distinct artefact the "
            "section actually read, in the status's sorted order"
        )
        for stamp in stamps:
            assert stamp == _ref_stamp(SOURCES[stamp["artefact"]]), (
                f"section {section['number']}: the stamp for {stamp['artefact']} is "
                f"{stamp!r}, which is not that artefact's own name, run, timestamp and "
                "payload digest"
            )
    six = next(s for s in doc["sections"] if s["number"] == "6")
    assert [s["artefact"] for s in six["provenance"]] == ["change_log", "model_registry"], (
        "section 6 draws on two artefacts and must stamp both — a section stamped with only "
        "the first source it read cannot be re-derived, and its drift is invisible"
    )


# --- exercise 6: staleness_report ------------------------------------------------------

def test_staleness_shape_and_one_row_per_stamp():
    doc = _document_of([("1", [SOURCES["model_registry"]]),
                        ("6", [SOURCES["change_log"], SOURCES["model_registry"]]),
                        ("9", [SOURCES["pmm_plan"]])])
    report = lesson.staleness_report(doc, SOURCES)
    assert isinstance(report, list) and all(isinstance(d, lesson.Drift) for d in report), (
        f"return a list of Drift, got {type(report).__name__} of "
        f"{ {type(d).__name__ for d in report} }"
    )
    assert [(d.number, d.artefact) for d in report] == [
        ("1", "model_registry"), ("6", "change_log"), ("6", "model_registry"),
        ("9", "pmm_plan")], (
        f"one row per STAMP, in document order then stamp order; you produced "
        f"{[(d.number, d.artefact) for d in report]} — a section with two sources needs two "
        "rows, or half its drift is unreportable"
    )
    assert {d.verdict for d in report} == {"current"}, (
        "a document stamped from today's sources and checked against today's sources is "
        f"current in every row; you reported {[d.verdict for d in report]}"
    )
    assert not any(d.hash_changed or d.timestamp_changed for d in report), (
        "nothing moved, so neither flag may be set"
    )


def test_staleness_catches_the_hand_edit():
    doc = _document_of([("1", [AS_FILED["model_registry"]])])
    row = lesson.staleness_report(doc, SOURCES)[0]
    assert row.verdict == "stale", (
        f"the registry's payload changed while its generated_at did not — somebody edited "
        f"the source in place. You reported {row!r}. A report keyed on the modification time "
        "misses this case for as long as it runs, which is every hand-edit ever made"
    )
    assert row.hash_changed is True and row.timestamp_changed is False, (
        f"report BOTH signals: the digest moved and the clock did not. You reported {row!r}"
    )
    both = AS_FILED["model_registry"]._replace(generated_at="2026-01-01T06:00:00Z")
    moved = lesson.staleness_report(_document_of([("1", [both])]), SOURCES)[0]
    assert moved.verdict == "stale" and moved.hash_changed and moved.timestamp_changed, (
        f"a source that was regenerated with different content is stale too, with both flags "
        f"set; you reported {moved!r}"
    )


def test_staleness_ignores_a_rerun():
    doc = _document_of([("9", [AS_FILED["pmm_plan"]])])
    row = lesson.staleness_report(doc, SOURCES)[0]
    assert row.verdict == "current", (
        f"the monitoring plan re-ran under a new run id and produced byte-identical output. "
        f"That is not drift, and flagging it is how a staleness report gets switched off. "
        f"You reported {row!r}"
    )
    assert row.hash_changed is False and row.timestamp_changed is True, (
        f"the clock moved and the content did not: report both, and decide on the content. "
        f"You reported {row!r}"
    )
    renamed = SOURCES["metrics"]._replace(run_id="met-9999-99-99")
    same = lesson.staleness_report(_document_of([("4", [renamed])]), SOURCES)[0]
    assert same.verdict == "current" and not same.timestamp_changed, (
        f"a different run id with the same timestamp and the same bytes is current, and the "
        f"timestamp flag is about generated_at, not about the run id; you reported {same!r}"
    )


def test_staleness_says_so_when_the_source_is_gone():
    doc = _document_of([("4", [SOURCES["metrics"]]), ("9", [SOURCES["pmm_plan"]])])
    without = {k: v for k, v in SOURCES.items() if k != "metrics"}
    report = lesson.staleness_report(doc, without)
    gone = next(d for d in report if d.artefact == "metrics")
    assert gone.verdict == "source_missing", (
        f"the artefact the section was generated from is no longer in the sources — that is "
        f"neither current nor stale, and calling it either is a guess. You reported {gone!r}"
    )
    assert gone.hash_changed is False and gone.timestamp_changed is False, (
        f"with the artefact gone there is nothing to compare, so neither flag may claim a "
        f"change was observed; you reported {gone!r}"
    )
    survivor = next(d for d in report if d.artefact == "pmm_plan")
    assert survivor.verdict == "current", (
        "one missing artefact must not derail the rest of the report"
    )
    empty = lesson.staleness_report({"sections": [], "omitted": [], "complete": True}, SOURCES)
    assert empty == [], f"a document with no emitted sections has no drift, got {empty!r}"


RUBRIC = [
    (test_field_value_walks_the_path, 3,
     "split on '.': the first segment is the artefact, the rest walks artefact.payload"),
    (test_field_value_reports_absence_without_raising, 3,
     "return MISSING for an absent artefact, an absent key, a walk into a non-dict, and a "
     "path with no dot — and never raise"),
    (test_field_value_does_not_confuse_falsy_with_absent, 2,
     "0, '' and [] are PRESENT values; hand them back unchanged and let is_sourced judge them"),
    (test_is_sourced_rejects_absence_and_blankness, 3,
     "MISSING, None, a whitespace-only string and an empty container are all unsourced"),
    (test_is_sourced_keeps_measurements, 3,
     "do not write `if not value`: 0, 0.0 and False are measurements, not absences"),
    (test_is_sourced_recurses_into_containers, 2,
     "a container is sourced when ANY element (for a dict, any value) is itself sourced"),
    (test_check_section_shape_and_a_complete_section, 3,
     "SectionStatus(number, title, complete, artefacts, missing, empty, unmet_choices), all "
     "four collections as tuples, artefacts sorted and distinct"),
    (test_check_section_separates_missing_from_empty, 4,
     "MISSING -> missing; present but not is_sourced -> empty; keep the two lists apart"),
    (test_check_section_handles_an_absent_artefact, 3,
     "an artefact that is not in sources makes every path into it missing, and contributes "
     "no artefact name"),
    (test_check_section_any_of_paths_are_not_fields, 3,
     "an any_of group is satisfied by one sourced path; its paths never appear in missing or "
     "empty, and an unsatisfied group goes into unmet_choices as a tuple"),
    (test_provenance_stamp_shape_and_values, 3,
     "{'artefact', 'run_id', 'generated_at', 'content_sha256'} — the first three off the "
     "artefact, the last from content_hash(artefact.payload)"),
    (test_provenance_stamp_digest_covers_the_payload_only, 3,
     "hash the payload alone: fold in the run id or the timestamp and every re-run reads as "
     "drift"),
    (test_render_emits_only_the_sourced_sections, 4,
     "check_section per spec; complete -> sections, otherwise -> omitted; complete is True "
     "only when nothing was omitted"),
    (test_render_never_emits_an_unsourced_value, 4,
     "fields is {path: value} for every required path plus the sourced any_of paths, in that "
     "order, and never a placeholder"),
    (test_render_explains_every_refusal, 3,
     "carry the status's three lists through, and build a reason string that names every "
     "blocking path, including both branches of an unmet either/or"),
    (test_render_stamps_every_emitted_section, 3,
     "one provenance_stamp per name in status.artefacts, in that order"),
    (test_staleness_shape_and_one_row_per_stamp, 3,
     "iterate document['sections'], then each section's 'provenance' stamps; one Drift each"),
    (test_staleness_catches_the_hand_edit, 3,
     "the verdict is decided by the content digest, never by generated_at"),
    (test_staleness_ignores_a_rerun, 3,
     "identical bytes under a new run id and a new timestamp are 'current' with "
     "timestamp_changed True"),
    (test_staleness_says_so_when_the_source_is_gone, 3,
     "artefact not in sources -> 'source_missing' with both flags False, and keep going"),
]
