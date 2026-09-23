#!/usr/bin/env python3
"""Author-side generator for P03-L07. NOT part of the student bundle.

    .venv/bin/python programmes/predictive-maintenance/lessons/P03-L07-ot-network-deployment/authoring/generate.py

Four shipped files come out of two masters, so the student file and the reference file
cannot drift apart:

    authoring/lesson.master.c   ->  solutions/lesson.c            (bodies kept)
                                ->  lesson.c                      (bodies replaced by stubs)
    authoring/lesson.master.py  ->  solutions/lesson_solution.py  (bodies kept)
                                ->  lesson.py                     (bodies replaced by stubs)

Two mechanisms, both line-based:

  * `COMMONS_ONLY_STUDENT` / `COMMONS_ONLY_SOLUTION` / `COMMONS_ONLY_END` blocks keep one branch
    and drop the other, plus the three marker lines. This is how the title, the build paths
    and the self-check answers differ between the two files.
  * a `COMMONS_STUB_END` marker inside a function body means: in the STUDENT file, replace
    everything from that marker to the end of the function with the stub in stubs.py. The
    marker line itself never ships.

The self-check digests are filled in here too, from SELF_CHECK_ANSWERS, so the four letters
appear in exactly one place in the repository: this file's companion, stubs.py.
"""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LESSON = HERE.parent
sys.path.insert(0, str(HERE))
from stubs import C_STUBS, PY_STUBS, SELF_CHECK_ANSWERS  # noqa: E402


def pick_branch(lines, want):
    """Keep the `want` branch of every COMMONS_ONLY block and drop the markers."""
    out, mode = [], "both"
    for line in lines:
        tag = line.strip().lstrip("/#").strip()
        if tag == "COMMONS_ONLY_STUDENT":
            mode = "student"
            continue
        if tag == "COMMONS_ONLY_SOLUTION":
            mode = "solution"
            continue
        if tag == "COMMONS_ONLY_END":
            mode = "both"
            continue
        if mode in ("both", want):
            out.append(line)
    return out


def c_student(lines):
    """Replace each marked C body with its stub, from the marker to the closing brace."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "// COMMONS_STUB_END":
            name = None
            for back in range(len(out) - 1, -1, -1):
                m = re.match(r"^static\s+[\w\s*]+?\b(\w+)\s*\(", out[back])
                if m:
                    name = m.group(1)
                    break
            if name not in C_STUBS:
                raise SystemExit(f"no stub registered for C function {name!r}")
            out.extend(C_STUBS[name].rstrip("\n").split("\n"))
            while i < len(lines) and lines[i] != "}":
                i += 1
            out.append("}")
            i += 1
            continue
        out.append(line)
        i += 1
    return out


def py_student(lines):
    """Replace each marked Python body with its stub, from the marker to the dedent."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "# COMMONS_STUB_END":
            name = None
            for back in range(len(out) - 1, -1, -1):
                m = re.match(r"^def\s+(\w+)\s*\(", out[back])
                if m:
                    name = m.group(1)
                    break
            if name not in PY_STUBS:
                raise SystemExit(f"no stub registered for Python function {name!r}")
            out.extend(PY_STUBS[name].rstrip("\n").split("\n"))
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in " \t"):
                i += 1
            # the two blank lines PEP 8 wants before whatever comes next, which the skip
            # above has just eaten along with the solution body
            if i < len(lines):
                out.extend(["", ""])
            continue
        out.append(line)
        i += 1
    return out


def drop_markers(lines, marker):
    return [l for l in lines if l.strip() != marker]


def fill_digests(text):
    for q, letter in SELF_CHECK_ANSWERS.items():
        digest = hashlib.sha256(f"P03-L07:{q}:{letter}".encode()).hexdigest()
        text = text.replace(f"__DIGEST{q}__", digest)
    return text


def write(path: Path, lines):
    body = fill_digests("\n".join(lines).rstrip("\n") + "\n")
    if "__DIGEST" in body:
        raise SystemExit(f"{path.name}: a self-check digest was not filled in")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    print(f"  wrote {path.relative_to(LESSON)}  ({len(lines)} lines)")


def main() -> int:
    c = (HERE / "lesson.master.c").read_text().split("\n")
    py = (HERE / "lesson.master.py").read_text().split("\n")

    write(LESSON / "solutions" / "lesson.c",
          drop_markers(pick_branch(c, "solution"), "// COMMONS_STUB_END"))
    write(LESSON / "lesson.c", c_student(pick_branch(c, "student")))
    write(LESSON / "solutions" / "lesson_solution.py",
          drop_markers(pick_branch(py, "solution"), "# COMMONS_STUB_END"))
    write(LESSON / "lesson.py", py_student(pick_branch(py, "student")))

    # Gate 7: no student-facing PROSE line may point at a directory the student never gets.
    for name in ("lesson.py", "lesson.c"):
        for n, line in enumerate((LESSON / name).read_text().splitlines(), 1):
            prose = line.lstrip().startswith("#") if name == "lesson.py" else True
            if prose and ("lesson_solution" in line or "solutions/" in line):
                raise SystemExit(f"LEAK {name}:{n}: {line.strip()[:70]}")
    student = (LESSON / "lesson.py").read_text()
    if re.search(r"SELF_CHECK\s*=\s*\{1:\s*\"[abcd]\"", student):
        raise SystemExit("LEAK lesson.py: the self-check answers are in the student file")
    print("  no leaks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
