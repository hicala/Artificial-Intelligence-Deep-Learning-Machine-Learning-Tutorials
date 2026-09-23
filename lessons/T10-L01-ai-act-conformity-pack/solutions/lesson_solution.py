# %% [markdown]
# # T10-L01 · The conformity evidence pack — reference solution
#
# The same notebook as `lesson.py`, with the three stubs implemented and the self-check
# answers at the foot of the file. Runs top to bottom in a fresh interpreter.
#
# > **This is not legal advice.** It is an engineering exercise about the *structure* of a
# > regulation. The dates are sourced in `claims.yaml`; the evidence mapping is this lesson's
# > modelling choice.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import contextlib
import hashlib
import io
import json
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Callable

import yaml

print("python", __import__("sys").version.split()[0], "· pyyaml", yaml.__version__)

AS_OF = date(2026, 9, 16)
print("as of", AS_OF.isoformat())

_FAILED_CHECKS: list[str] = []
_STATUS: dict[str, str] = {}  # label -> "passed" | "failed" | "not started"

# The exercises, in the order you meet them. The progress board at the foot of the
# notebook is built from this, and a demo that is waiting on one names it from here.
_EXERCISES = {
    "exercise 1": "classify()",
    "exercise 2": "missing_evidence()",
    "exercise 3": "conformity_report() and render_report()",
}


def _try(label: str, check: Callable[[], None], needs: tuple[str, ...] = ()) -> None:
    """Run a check, or a demo that depends on your code, without derailing the notebook.

    A stub you have not filled in yet simply says so. A wrong answer prints the check's own
    message — which names the likely mistake — and the notebook carries on to the next cell,
    so one broken exercise never hides the feedback on the other two.

    Nothing is swallowed: every failure is recorded, and the final cell of this file exits
    non-zero if the list is not empty. That is what stops tools/execute.py from ever
    reporting PASS on a reference implementation that is quietly wrong.

    A demo names the exercises it `needs`. Until each of them has passed its check, the demo
    says which one it is waiting for and skips, rather than failing half-way through its
    output. Every outcome is recorded in `_STATUS`, which the progress board at the foot reads.
    """
    waiting = [name for name in needs if _STATUS.get(name) != "passed"]
    if waiting:
        _STATUS[label] = "not started"
        todo = ", ".join(f"{name} ({_EXERCISES[name]})" for name in waiting)
        print(f"{label}: skipped — this needs {todo} to pass first.")
        return
    try:
        check()
    except NotImplementedError as exc:
        _STATUS[label] = "not started"
        stub = traceback.extract_tb(exc.__traceback__)[-1].name
        print(f"{label}: not implemented yet — fill in {stub}() above, then re-run this cell.")
    except AssertionError as exc:
        _STATUS[label] = "failed"
        _FAILED_CHECKS.append(label)
        print(f"{label}: FAILED — {exc}")
    except Exception as exc:  # a half-finished implementation raising something else
        _STATUS[label] = "failed"
        _FAILED_CHECKS.append(label)
        print(f"{label}: raised {type(exc).__name__}: {exc}")
    else:
        _STATUS[label] = "passed"


_MARKS = {"passed": "✅", "failed": "❌", "not started": "⏳"}


def _progress_board() -> None:
    """One line per exercise, marked passed, failed or not started, then the tally."""
    width = max(len(what) for what in _EXERCISES.values())
    print("progress board")
    for label, what in _EXERCISES.items():
        state = _STATUS.get(label, "not started")
        print(f"  {_MARKS[state]} {label:<11} {what:<{width}}  {state}")
    done = sum(_STATUS.get(label) == "passed" for label in _EXERCISES)
    print(f"\n{done} of {len(_EXERCISES)} exercises complete")
    failing = [label for label, state in _STATUS.items() if state == "failed"]
    if failing:
        print("failing right now: " + ", ".join(failing) + ". Each one printed what went wrong "
              "in its own cell above, and every exercise heading has hints you can open.")
    elif done < len(_EXERCISES):
        print("work top to bottom: every exercise heading has hints you can open.")

# %% [markdown]
# ## 1. The date ladder

# %%
MILESTONES = {
    "2025-02-02": "Article 5 prohibited practices, and Article 4 AI literacy",
    "2025-08-02": "General-purpose AI model obligations, and the governance rules",
    "2026-08-02": "General application of the AI Act, including Article 50 transparency",
    "2026-12-02": "Article 50(2) marking, for systems on the market before 2026-08-02",
    "2027-12-02": "Chapter III duties for Annex III stand-alone high-risk systems",
    "2028-08-02": "Chapter III duties for Annex I product-embedded high-risk systems",
}

print(f"{'date':12s} {'':4s} what starts to bind")
for _iso, _what in sorted(MILESTONES.items()):
    _state = "PAST" if date.fromisoformat(_iso) <= AS_OF else "->"
    print(f"{_iso:12s} {_state:4s} {_what}")
print(f"\n{sum(date.fromisoformat(d) <= AS_OF for d in MILESTONES)} of "
      f"{len(MILESTONES)} rungs already bind as of {AS_OF.isoformat()}")

# %% [markdown]
# ## 2. The registry

# %%
def registry_path() -> Path:
    """Locate the shipped registry, searching this file's directory and its parent."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:  # a notebook has no __file__
        here = Path.cwd()
    for candidate in (here / "assets" / "systems.yaml",
                      here.parent / "assets" / "systems.yaml",
                      Path.cwd() / "assets" / "systems.yaml",
                      Path.cwd().parent / "assets" / "systems.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("assets/systems.yaml not found next to this lesson")


def load_registry(path: Path | None = None) -> list:
    """Read the registry and hand back a list of system dictionaries."""
    return yaml.safe_load((path or registry_path()).read_text())["systems"]


def parse_date(value) -> date:
    """Accept an ISO string or a date and return a date. Raise on anything else."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"not a date: {value!r}")


REGISTRY = load_registry()
print(f"{len(REGISTRY)} systems, "
      f"{sum(len(s['evidence']) for s in REGISTRY)} evidence rows between them\n")
for _s in REGISTRY:
    print(f"  {_s['id']:16s} {_s['role']:8s} {_s['name'][:58]}")

# %%
print(json.dumps(REGISTRY[1], indent=2)[:900])

# %% [markdown]
# ## 3. The rulebook

# %%
OBLIGATIONS = {
    "art4_ai_literacy": {
        "label": "AI literacy for the people who build and operate the system",
        "article": "Article 4", "applies_from": "2025-02-02"},
    "art5_prohibited_practice": {
        "label": "Prohibited practice: it must not be placed on the market or used",
        "article": "Article 5", "applies_from": "2025-02-02"},
    "art50_inform_interaction": {
        "label": "Tell people when they are interacting with an AI system",
        "article": "Article 50(1)", "applies_from": "2026-08-02"},
    "art50_mark_synthetic": {
        "label": "Mark synthetic content in a machine-readable format",
        "article": "Article 50(2)", "applies_from": "2026-08-02"},
    "art53_gpai_model": {
        "label": "GPAI model documentation, copyright policy, training-data summary",
        "article": "Article 53", "applies_from": "2025-08-02"},
    "ch3_high_risk_annex_iii": {
        "label": "Chapter III duties, stand-alone high-risk (Annex III)",
        "article": "Article 6(2)", "applies_from": "2027-12-02"},
    "ch3_high_risk_annex_i": {
        "label": "Chapter III duties, product-embedded high-risk (Annex I)",
        "article": "Article 6(1)", "applies_from": "2028-08-02"},
}

GENERAL_APPLICATION = "2026-08-02"
ART50_MARK_STANDARD = "2026-08-02"
ART50_MARK_LEGACY = "2026-12-02"

HIGH_RISK_CORE_EVIDENCE = (
    "risk_management_system",                    # Article 9
    "data_governance_record",                    # Article 10
    "technical_documentation",                   # Article 11
    "automatic_logging_design",                  # Article 12
    "instructions_for_use",                      # Article 13
    "human_oversight_plan",                      # Article 14
    "accuracy_robustness_cybersecurity_report",  # Article 15
    "quality_management_system",                 # Article 17
    "conformity_assessment_record",              # Article 43
    "eu_declaration_of_conformity",              # Article 47
    "ce_marking_record",                         # Article 48
    "post_market_monitoring_plan",               # Article 72
    "serious_incident_procedure",                # Article 73
)

EVIDENCE_REQUIRED = {
    "art4_ai_literacy": ("ai_literacy_training_record",),
    "art5_prohibited_practice": ("withdrawal_decision_record",),
    "art50_inform_interaction": ("user_facing_ai_disclosure",),
    "art50_mark_synthetic": ("machine_readable_marking_spec", "content_provenance_test_report"),
    "art53_gpai_model": ("model_technical_documentation", "copyright_policy",
                         "training_data_summary"),
    "ch3_high_risk_annex_iii": HIGH_RISK_CORE_EVIDENCE + ("eu_database_registration",),
    "ch3_high_risk_annex_i": HIGH_RISK_CORE_EVIDENCE + ("integrated_product_conformity_assessment",),
}

print(f"{len(OBLIGATIONS)} obligations, "
      f"{len(set().union(*EVIDENCE_REQUIRED.values()))} distinct evidence items")
print(f"the heaviest obligation asks for "
      f"{max(len(v) for v in EVIDENCE_REQUIRED.values())} of them")

# %% [markdown]
# ## 4. Exercise 1 — `classify(system)`

# %%
def classify(system: dict) -> dict:
    """Map a described system to its risk tier, obligation set and dates.

    Precedence: prohibited > Annex I embedding > Annex III area > GPAI model > transparency
    > minimal. Obligations cumulate; the tier names the heaviest route, it does not replace
    the set. The Article 50(2) date is per system.
    """
    placed = parse_date(system["placed_on_market"]) if system.get("placed_on_market") else None
    obligations = {"art4_ai_literacy"}

    if system.get("prohibited_practice"):
        tier = "prohibited"
        obligations.add("art5_prohibited_practice")
    else:
        embedded = bool(system.get("embedded_in_regulated_product"))
        annex_iii = bool(system.get("annex_iii_area"))
        gpai = bool(system.get("is_gpai_model"))
        interacts = bool(system.get("interacts_with_humans"))
        synthetic = bool(system.get("generates_synthetic_content"))

        # The embedding wins: a safety component of a product already covered by EU
        # product-safety law travels the Annex I route, and that is the later date.
        if embedded:
            tier = "high_risk_annex_i"
            obligations.add("ch3_high_risk_annex_i")
        elif annex_iii:
            tier = "high_risk_annex_iii"
            obligations.add("ch3_high_risk_annex_iii")
        elif gpai:
            tier = "gpai_model"
        elif interacts or synthetic:
            tier = "transparency_only"
        else:
            tier = "minimal"

        # Cumulation: each of these is decided on its own, whatever the tier came out as.
        if gpai:
            obligations.add("art53_gpai_model")
        if interacts:
            obligations.add("art50_inform_interaction")
        if synthetic:
            obligations.add("art50_mark_synthetic")

    deadlines = {}
    for obligation in obligations:
        if obligation == "art50_mark_synthetic":
            legacy = placed is not None and placed < parse_date(GENERAL_APPLICATION)
            deadlines[obligation] = ART50_MARK_LEGACY if legacy else ART50_MARK_STANDARD
        else:
            deadlines[obligation] = OBLIGATIONS[obligation]["applies_from"]

    return {
        "risk_tier": tier,
        "obligations": sorted(obligations),
        "deadlines": deadlines,
        "applies_from": min(deadlines.values()),  # ISO dates sort chronologically
    }


def _check_classify() -> None:
    plain = {"id": "p", "placed_on_market": "2026-01-01"}
    got = classify(plain)
    assert set(got) == {"risk_tier", "obligations", "deadlines", "applies_from"}, (
        f"keys were {sorted(got)} — return exactly the four documented names.")
    assert got["risk_tier"] == "minimal", (
        f"a system with no flags is 'minimal', you returned {got['risk_tier']!r}.")
    assert got["obligations"] == ["art4_ai_literacy"], (
        f"minimal risk is one duty, not none: expected ['art4_ai_literacy'], got "
        f"{got['obligations']} — Article 4 applies to every system here.")

    chatty = {"id": "c", "annex_iii_area": "credit_scoring", "interacts_with_humans": True,
              "generates_synthetic_content": True, "placed_on_market": "2026-08-30"}
    got = classify(chatty)
    assert got["risk_tier"] == "high_risk_annex_iii", (
        f"got {got['risk_tier']!r} — a named Annex III area makes it high risk even though "
        "it is also a chatbot.")
    assert set(got["obligations"]) == {"art4_ai_literacy", "ch3_high_risk_annex_iii",
                                       "art50_inform_interaction", "art50_mark_synthetic"}, (
        f"got {got['obligations']} — obligations cumulate; being high risk does not replace "
        "the Article 50 duties, it adds to them.")
    assert got["obligations"] == sorted(got["obligations"]), "return the ids sorted."
    assert got["deadlines"]["art50_mark_synthetic"] == ART50_MARK_STANDARD, (
        f"placed on 2026-08-30, after general application, so marking is due "
        f"{ART50_MARK_STANDARD}, not {got['deadlines']['art50_mark_synthetic']}.")
    assert got["applies_from"] == "2025-02-02", (
        f"applies_from is the EARLIEST date in deadlines, {got['applies_from']!r} is not it "
        "— Article 4 has been binding since 2025-02-02.")

    legacy = dict(chatty, id="l", annex_iii_area=None, placed_on_market="2025-05-20")
    assert classify(legacy)["deadlines"]["art50_mark_synthetic"] == ART50_MARK_LEGACY, (
        "a system placed on the market before general application gets the transitional "
        f"marking date {ART50_MARK_LEGACY}.")

    both = {"id": "b", "annex_iii_area": "critical_infrastructure",
            "embedded_in_regulated_product": True, "placed_on_market": "2026-05-05"}
    assert classify(both)["risk_tier"] == "high_risk_annex_i", (
        "when a system is both an Annex III use case and embedded in a regulated product, "
        "the embedding decides the route — and it is the 2028 date, not the 2027 one.")

    banned = dict(both, id="z", prohibited_practice="social_scoring")
    got = classify(banned)
    assert got["risk_tier"] == "prohibited", "a prohibited practice outranks every other flag."
    assert got["obligations"] == ["art4_ai_literacy", "art5_prohibited_practice"], (
        f"got {got['obligations']} — once prohibited, stop: do not also add the Chapter III "
        "duties of a system that must not exist.")
    print("exercise 1 looks right")


# %%
if __name__ == "__main__":
    _try("exercise 1", _check_classify)


# %% [markdown]
# ## 5. Exercise 2 — `missing_evidence(system, as_of)`

# %%
def missing_evidence(system: dict, as_of: date) -> list:
    """Return the sorted ids of required evidence items that do not actually count.

    Four tests, all of which must hold: status is "present", the document reference is a
    non-empty string, the issue date exists and is not in the future, and any review_due has
    not already passed.
    """
    required = set()
    for obligation in classify(system)["obligations"]:
        required.update(EVIDENCE_REQUIRED.get(obligation, ()))

    evidence = system.get("evidence") or {}
    missing = []
    for item in required:
        record = evidence.get(item)
        if not isinstance(record, dict):
            missing.append(item)                       # absent entirely
            continue
        if record.get("status") != "present":
            missing.append(item)                       # "planned" is a promise
            continue
        if not str(record.get("document") or "").strip():
            missing.append(item)                       # a row with no document behind it
            continue
        issued = record.get("date")
        if issued is None or parse_date(issued) > as_of:
            missing.append(item)                       # undated, or not issued yet
            continue
        review = record.get("review_due")
        if review is not None and parse_date(review) < as_of:
            missing.append(item)                       # evidence has a shelf life
            continue
    return sorted(missing)


def _check_missing_evidence() -> None:
    base = {"id": "e", "is_gpai_model": True, "placed_on_market": "2025-09-01", "evidence": {}}
    got = missing_evidence(base, AS_OF)
    assert got == sorted(["ai_literacy_training_record", "model_technical_documentation",
                          "copyright_policy", "training_data_summary"]), (
        f"an empty evidence map means everything required is missing; got {got}.")

    ok = {"status": "present", "document": "DOC-1", "date": "2025-01-01"}
    cases = {
        "planned is not evidence": {"status": "planned", "document": "DOC-1",
                                    "date": "2025-01-01"},
        "empty document reference": {"status": "present", "document": "   ",
                                     "date": "2025-01-01"},
        "no date at all": {"status": "present", "document": "DOC-1", "date": None},
        "dated in the future": {"status": "present", "document": "DOC-1",
                                "date": "2027-01-01"},
        "review already lapsed": {"status": "present", "document": "DOC-1",
                                  "date": "2025-01-01", "review_due": "2026-01-01"},
    }
    for why, record in cases.items():
        s = {"id": "e", "placed_on_market": "2026-01-01",
             "evidence": {"ai_literacy_training_record": record}}
        assert missing_evidence(s, AS_OF) == ["ai_literacy_training_record"], (
            f"{why}: this record must not count, but your function accepted it.")
    s = {"id": "e", "placed_on_market": "2026-01-01",
         "evidence": {"ai_literacy_training_record": ok}}
    assert missing_evidence(s, AS_OF) == [], (
        "a present, referenced, in-date record with no review_due must count — check you are "
        "not rejecting records that simply have no review_due key.")

    future_review = dict(ok, review_due="2027-01-01")
    s = {"id": "e", "placed_on_market": "2026-01-01",
         "evidence": {"ai_literacy_training_record": future_review}}
    assert missing_evidence(s, AS_OF) == [], (
        "a review due in the future has not lapsed; only a review_due BEFORE as_of kills it.")
    print("exercise 2 looks right")


# %%
if __name__ == "__main__":
    _try("exercise 2", _check_missing_evidence, needs=("exercise 1",))


# %% [markdown]
# ## 6. Exercise 3 — `conformity_report(systems, as_of)`

# %%
def conformity_report(systems: list, as_of: date) -> dict:
    """Build the evidence table and its summary.

    `overdue` is the field that earns the report its credibility: a gap counts as overdue
    only when the obligation it belongs to already binds.
    """
    rows = []
    for system in systems:
        verdict = classify(system)
        missing = missing_evidence(system, as_of)
        missing_set = set(missing)
        overdue = any(
            parse_date(verdict["deadlines"][obligation]) <= as_of
            and missing_set & set(EVIDENCE_REQUIRED.get(obligation, ()))
            for obligation in verdict["obligations"]
        )
        rows.append({
            "id": system["id"],
            "name": system.get("name", ""),
            "risk_tier": verdict["risk_tier"],
            "deadlines": verdict["deadlines"],
            "applies_from": verdict["applies_from"],
            "days_to_deadline": (parse_date(verdict["applies_from"]) - as_of).days,
            "missing": missing,
            "complete": not missing,
            "overdue": overdue,
        })

    by_tier = {}
    for row in rows:
        by_tier[row["risk_tier"]] = by_tier.get(row["risk_tier"], 0) + 1

    upcoming = [iso for row in rows if not row["complete"]
                for iso in row["deadlines"].values() if parse_date(iso) > as_of]

    summary = {
        "total": len(rows),
        "complete": sum(1 for r in rows if r["complete"]),
        "incomplete": sum(1 for r in rows if not r["complete"]),
        "overdue": sum(1 for r in rows if r["overdue"]),
        "by_tier": by_tier,
        "next_deadline": min(upcoming) if upcoming else None,
    }
    blocking = sorted(r["id"] for r in rows
                      if r["risk_tier"] == "prohibited" or r["overdue"])
    return {"as_of": as_of.isoformat(), "rows": rows, "summary": summary, "blocking": blocking}


def render_report(report: dict) -> str:
    """Render the report as a plain-text table, and return it."""
    rows, summary = report["rows"], report["summary"]
    as_of = parse_date(report["as_of"])
    width = max((len(r["id"]) for r in rows), default=6) + 1
    lines = [
        f"EU AI Act conformity evidence pack — assembled {report['as_of']}",
        "=" * 82,
        f"  {'system':<{width}} {'route':<22} {'binds':<12} {'next due':<12} {'gaps':>5}  state",
    ]
    for row in rows:
        if row["risk_tier"] == "prohibited" or row["overdue"]:
            state = "BLOCKING"
        elif row["complete"]:
            state = "complete"
        else:
            state = "in progress"
        # "binds" is the same date for every row — Article 4 catches everything, which is the
        # point. "next due" is the one a reviewer can act on this quarter.
        upcoming = sorted(iso for iso in row["deadlines"].values() if parse_date(iso) > as_of)
        lines.append(
            f"  {row['id']:<{width}} {row['risk_tier']:<22} {row['applies_from']:<12} "
            f"{(upcoming[0] if upcoming else 'nothing'):<12} {len(row['missing']):>5}  {state}")

    lines += ["", "  blocking, and why:"]
    if report["blocking"]:
        for row in rows:
            if row["id"] not in report["blocking"]:
                continue
            if row["risk_tier"] == "prohibited":
                lines.append(f"    {row['id']}: prohibited practice — withdraw it, then "
                             "evidence the withdrawal")
            for obligation, iso in sorted(row["deadlines"].items()):
                gaps = sorted(set(row["missing"]) & set(EVIDENCE_REQUIRED.get(obligation, ())))
                if gaps and parse_date(iso) <= as_of:
                    lines.append(f"    {row['id']}: {obligation} bound on {iso}, still "
                                 f"missing {', '.join(gaps)}")
    else:
        lines.append("    nothing")

    lines += [
        "",
        f"  {summary['total']} systems · {summary['complete']} complete · "
        f"{summary['incomplete']} in progress · {summary['overdue']} overdue",
        "  by route: " + ", ".join(f"{tier} {count}"
                                   for tier, count in sorted(summary["by_tier"].items())),
        f"  next deadline: {summary['next_deadline'] or 'none outstanding'}",
        "",
        "  Generated from a rulebook, not from judgement. This is not legal advice.",
    ]
    return "\n".join(lines)


def _check_conformity_report() -> None:
    report = conformity_report(REGISTRY, AS_OF)
    assert set(report) == {"as_of", "rows", "summary", "blocking"}, (
        f"keys were {sorted(report)} — return exactly those four.")
    assert [r["id"] for r in report["rows"]] == [s["id"] for s in REGISTRY], (
        "rows must stay in the input order; a report that reorders itself is not diffable.")
    summary = report["summary"]
    assert summary["total"] == len(REGISTRY), (
        f"summary['total'] was {summary['total']}, expected {len(REGISTRY)} — total counts "
        "systems in the registry, not rows you decided were interesting.")
    assert summary["complete"] + summary["incomplete"] == summary["total"], (
        f"{summary['complete']} + {summary['incomplete']} != {summary['total']}.")
    assert sum(summary["by_tier"].values()) == summary["total"], (
        f"by_tier sums to {sum(summary['by_tier'].values())}, not {summary['total']}.")
    weld = next(r for r in report["rows"] if r["id"] == "weld-inspector")
    assert weld["complete"] and not weld["overdue"], (
        "weld-inspector owes one training record and has it; it should be complete.")
    xray = next(r for r in report["rows"] if r["id"] == "xray-triage")
    assert xray["missing"] and not xray["overdue"], (
        "xray-triage has a gap, but only on a 2028 obligation — gaps are not yet breaches, "
        "so overdue must be False here.")
    text = render_report(report)
    assert isinstance(text, str) and "not legal advice" in text.lower(), (
        "render_report returns the table as a string, ending with its caveat.")
    print("exercise 3 looks right")


# %%
if __name__ == "__main__":
    _try("exercise 3", _check_conformity_report, needs=("exercise 1", "exercise 2"))


# %% [markdown]
# ## 7. The payoff

# %%
def summarise(as_of: date = AS_OF) -> dict:
    """Print the evidence table and return the report."""
    report = conformity_report(REGISTRY, as_of)
    print(render_report(report))
    return report


# %% [markdown]
# ## 8. Common mistakes
#
# As in `lesson.py`. The two that cost the most: treating the tier as the obligation set, and
# counting a "planned" row as evidence.

# %%
# Run the pack itself over the whole registry. The three public checks ran after each exercise.
if __name__ == "__main__":
    _try("the evidence pack", summarise, needs=("exercise 1", "exercise 2", "exercise 3"))

# %% [markdown]
# ## 9. Self-check — answers
#
# **1. (c)** Both, on different dates, plus Article 4. Obligations cumulate: the tier names
# the heaviest route, it does not absorb the lighter ones. The Chapter III duties bind on
# 2027-12-02 and the Article 50(1) duty on 2026-08-02, while Article 4 has bound since
# 2025-02-02 — which is why `classify()` reports `applies_from` as 2025-02-02 and why the
# report's `overdue` field looks at the date of *each* obligation rather than the row's.
#
# **2. (c)** The Digital Omnibus deferred the Chapter III high-risk timetable to 2027-12-02
# (Annex III) and 2028-08-02 (Annex I). It moved neither Article 5, which has bound since
# 2025-02-02, nor Article 4, nor Article 50, which has applied since 2026-08-02. "High-risk
# was delayed" is not "the AI Act was delayed" — and in this registry the systems that end up
# blocking are precisely the ones whose duties were never deferred.
#
# **3. (c)** No. The document is real and the issue date is in the past, but the review fell
# due on 2026-08-10 and the pack is assembled on 2026-09-16, so it lapsed five weeks before
# anyone looked. Evidence has a shelf life; `missing_evidence` treats a lapsed `review_due`
# exactly as it treats an absent document, which is why `cv-screener` shows a gap on its risk
# management system even though the row says "present".
#
# **4. (b)** 2028-08-02. The embedding decides the route: an AI system that is a safety
# component of a product already covered by EU product-safety law is classified under
# Article 6(1) and Annex I, and that route carries the later date. Answering (a) would have a
# team working to a deadline two years earlier than the one that binds — which is why
# `classify()` tests the embedding before it tests the Annex III area.
#
# The marker below is the same one the student notebook ships, with the answers filled in, so
# running this file prints a real 4/4 rather than a claim of one.

# %%
_SELF_CHECK_KEY = {
    1: "089c9bf3af253313",
    2: "4ce9b56e7eefe4fa",
    3: "5dfce51b8232f041",
    4: "4e4144ddc6c88fcc",
}

_SELF_CHECK_HINT = {
    1: "re-read section 4 on cumulation — does being high risk switch the Article 50 duties "
       "off, or add to them?",
    2: "look at which rungs of the ladder printed PAST in section 1, and count them.",
    3: "apply the fourth test in section 5: compare review_due against AS_OF.",
    4: "look at the order classify() is told to test the flags in, and at which of the two "
       "Chapter III dates that route carries.",
}

SELF_CHECK_ANSWERS = {1: "c", 2: "c", 3: "c", 4: "b"}


def check_self_check(answers: dict) -> None:
    """Mark self-check answers against the hashed key."""
    right = 0
    for question in sorted(_SELF_CHECK_KEY):
        given = str(answers.get(question, "")).strip().lower()
        digest = hashlib.sha256(f"T10-L01:q{question}:{given}".encode()).hexdigest()[:16]
        if digest == _SELF_CHECK_KEY[question]:
            right += 1
            print(f"  q{question}  correct")
        elif not given:
            print(f"  q{question}  no answer given")
        else:
            print(f"  q{question}  not {given!r} — {_SELF_CHECK_HINT[question]}")
    print(f"\n{len(_SELF_CHECK_KEY)} questions, {right} right")


# %% [markdown]
# ## What you built, and where it goes next
#
# **This is not legal advice.**

# %%
if __name__ == "__main__":
    print("\nself-check:")
    check_self_check(SELF_CHECK_ANSWERS)

# %%
# Your progress board. Every check is re-run here, quietly, against your code as it stands
# now — each one already printed its feedback in its own cell above — so the board is
# current even if you edited an exercise and did not re-run its check.
if __name__ == "__main__":
    _ALL_CHECKS = (  # (exercise, its check, the exercises that check relies on)
        ("exercise 1", _check_classify, ()),
        ("exercise 2", _check_missing_evidence, ("exercise 1",)),
        ("exercise 3", _check_conformity_report, ("exercise 1", "exercise 2")),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        for _name, _check, _needs in _ALL_CHECKS:
            _try(_name, _check, needs=_needs)
    _progress_board()
    # A stub you have not reached yet is not a failure. A check that ran and came back wrong
    # is: in a script or under CI it ends the run non-zero, rather than letting a green exit
    # code paper over it. Inside a notebook kernel the board above has already said so, in a
    # line rather than a traceback at the foot of the page.
    if _FAILED_CHECKS and "ipykernel" not in sys.modules:
        raise SystemExit("checks failed: " + ", ".join(dict.fromkeys(_FAILED_CHECKS)))
