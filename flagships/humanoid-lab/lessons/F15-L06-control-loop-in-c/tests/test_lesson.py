"""Autograder rubric for F15-L06. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure). Partial credit is the point — a
student who has written the C control law but not yet the loop should see exactly that.

The C exercises are graded by running the compiled binary and checking its output against an
independent reference implemented in THIS file, so the rubric never trusts the student's own
arithmetic to confirm the student's own arithmetic. A binary that exits 2 has an unfinished
stub; run_c turns that into NotImplementedError, which the grader prints as TODO rather than
as an error.
"""
import importlib.util, math, os
from pathlib import Path

import mujoco
import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

_CACHE = {}


def _cached(key, fn):
    """Run the expensive C commands once per grading session, not once per test."""
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def _run_text(ticks=None, kp=None, kd=None, trace_every=None):
    args = ["run", "--model", lesson.MODEL_REL,
            "--kp", str(lesson.KP if kp is None else kp),
            "--kd", str(lesson.KD if kd is None else kd),
            "--ticks", str(lesson.TICKS if ticks is None else ticks)]
    if trace_every is not None:
        args += ["--trace-every", str(trace_every)]
    return lesson.run_c(*args)


def _reference_loop(target, kp, kd, ticks, trace_every=0):
    """An independent Python rollout of the same loop, written against the bindings directly.

    Deliberately does NOT call any helper from lesson.py, so a student who redefines
    py_addresses cannot move this reference.
    """
    m = mujoco.MjModel.from_xml_path(str(lesson.MODEL_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    adr = []
    for i in range(int(m.nu)):
        j = int(m.actuator_trnid[i, 0])
        adr.append((int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])))
    sat = 0
    trace = []
    ctrl = [0.0] * int(m.nu)
    for k in range(ticks):
        on_limit = 0
        for i in range(int(m.nu)):
            qadr, vadr = adr[i]
            tau = kp * (target[i] - float(d.qpos[qadr])) - kd * float(d.qvel[vadr])
            lo, hi = float(m.actuator_ctrlrange[i][0]), float(m.actuator_ctrlrange[i][1])
            if m.actuator_ctrllimited[i]:
                tau = min(max(tau, lo), hi)
            ctrl[i] = tau
            if m.actuator_ctrllimited[i] and (tau >= hi - 1e-12 or tau <= lo + 1e-12):
                on_limit = 1
        for i in range(int(m.nu)):
            d.ctrl[i] = ctrl[i]
        mujoco.mj_step(m, d)
        sat += on_limit
        if trace_every > 0 and k % trace_every == 0:
            row = [float(k), float(d.time)]
            row += [float(d.qpos[adr[i][0]]) for i in range(2)]
            row += [float(d.qvel[adr[i][1]]) for i in range(2)]
            row += [ctrl[i] for i in range(2)]
            trace.append(row)
    worst = max(abs(target[i] - float(d.qpos[adr[i][0]])) for i in range(int(m.nu)))
    return {"ticks_done": ticks, "sim_seconds": float(d.time), "final_abs_err": worst,
            "saturated_ticks": sat, "trace": trace}


# --------------------------------------------------------------------------------------
# The rubric
# --------------------------------------------------------------------------------------

def test_c_builds():
    ok, out = lesson.build(verbose=False)
    assert ok, f"the C source did not compile:\n{out}"


def test_c_selftest_passes():
    code, out = _cached("selftest", lesson.make_test)
    if code == 2:
        raise NotImplementedError("at least one C exercise is still a stub")
    assert code == 0, f"`make test` failed with exit {code}:\n{out}"


def test_addresses_on_pinned_arm():
    text = _cached("addr_pinned",
                   lambda: lesson.run_c("addrdump", "--model", lesson.MODEL_REL))
    head, block = lesson.metrics(text), lesson.rows(text, "addr")
    m = mujoco.MjModel.from_xml_path(str(lesson.MODEL_PATH))
    assert len(block) == int(m.nu), (
        f"addrdump printed {len(block)} rows for {int(m.nu)} actuators")
    for i, (idx, joint, qadr, vadr, _jt) in enumerate(block):
        want_j = int(m.actuator_trnid[i, 0])
        assert int(joint) == want_j, (
            f"actuator {i} drives joint {want_j}, you returned {int(joint)} — the id is "
            "m->actuator_trnid[2*actuator]")
        assert int(qadr) == int(m.jnt_qposadr[want_j]), (
            f"actuator {i}: qpos address {int(qadr)}, expected {int(m.jnt_qposadr[want_j])}")
        assert int(vadr) == int(m.jnt_dofadr[want_j]), (
            f"actuator {i}: qvel address {int(vadr)}, expected {int(m.jnt_dofadr[want_j])}")


def test_addresses_on_floating_arm():
    """The case that separates a real lookup from `qpos[i]` / `qvel[i]`."""
    text = _cached("addr_float",
                   lambda: lesson.run_c("addrdump", "--model", lesson.FLOATING_REL))
    head, block = lesson.metrics(text), lesson.rows(text, "addr")
    m = mujoco.MjModel.from_xml_path(str(lesson.FLOATING_PATH))
    assert head["nq"] == int(m.nq) and head["nv"] == int(m.nv), (
        f"addrdump reported nq={head['nq']:.0f}, nv={head['nv']:.0f}; the model has "
        f"nq={int(m.nq)}, nv={int(m.nv)}")
    assert int(m.nq) - int(m.nv) == 1, "the shipped floating model should have nq = nv + 1"
    for i, (idx, joint, qadr, vadr, _jt) in enumerate(block):
        want_j = int(m.actuator_trnid[i, 0])
        want_q, want_v = int(m.jnt_qposadr[want_j]), int(m.jnt_dofadr[want_j])
        assert (int(qadr), int(vadr)) == (want_q, want_v), (
            f"on the floating arm actuator {i} sits at qpos {want_q} / qvel {want_v}, but you "
            f"returned qpos {int(qadr)} / qvel {int(vadr)}. A free joint takes 7 qpos slots "
            "and 6 qvel rows, so the two addresses are NOT interchangeable — read each from "
            "its own table.")
        assert int(qadr) != int(vadr), (
            f"actuator {i} returned the same number for both addresses on a model with a free "
            "joint; that cannot be right")


def test_pd_ctrl_clamps_to_each_actuators_own_range():
    text = _cached("ctrl_start", lambda: lesson.run_c(
        "ctrldump", "--model", lesson.MODEL_REL, "--kp", "40", "--kd", "3",
        "--target", "0.8,-1.2"))
    block = lesson.rows(text, "ctrl")
    assert len(block) == 2, f"ctrldump printed {len(block)} rows, expected 2"
    for i, (idx, c, lo, hi) in enumerate(block):
        assert lo - 1e-12 <= c <= hi + 1e-12, (
            f"actuator {i} returned {c!r}, outside its ctrlrange [{lo}, {hi}]")
    # 40*0.8 = 32 against a limit of 8; 40*(-1.2) = -48 against -3. Both must pin exactly.
    assert abs(block[0][1] - block[0][3]) < 1e-12, (
        f"the shoulder must clamp to exactly its high bound {block[0][3]!r}, got {block[0][1]!r}")
    assert abs(block[1][1] - block[1][2]) < 1e-12, (
        f"the elbow must clamp to exactly its low bound {block[1][2]!r}, got {block[1][1]!r}")
    # The two limits differ; a single shared clamp would put the elbow at -8 or the shoulder
    # at -3, and this catches either.
    assert block[0][3] != block[1][3], (
        "the shipped model should give the two actuators different ranges, or this test "
        "proves nothing")
    # An UNLIMITED actuator must not be clamped at all. Both shipped models limit both
    # actuators, so without --unlimited (which clears the flag on the loaded model) a law
    # that clamps unconditionally behaves identically to a correct one and no check in this
    # lesson can tell them apart.
    free = lesson.rows(lesson.run_c(
        "ctrldump", "--model", lesson.MODEL_REL, "--kp", "40", "--kd", "3",
        "--target", "0.8,-1.2", "--unlimited"), "ctrl")
    assert len(free) == 2, f"ctrldump --unlimited printed {len(free)} rows, expected 2"
    assert abs(free[0][1] - 32.0) < 1e-9 and abs(free[1][1] - (-48.0)) < 1e-9, (
        f"with every actuator's ctrllimited flag cleared the law must hand back its RAW "
        f"commands 40*0.8 = 32 and 40*(-1.2) = -48; you returned {free[0][1]!r} and "
        f"{free[1][1]!r}. Clamp only when m->actuator_ctrllimited[i] is set — an unlimited "
        "actuator's ctrlrange is meaningless and clamping to it throttles the controller.")


def test_pd_ctrl_arithmetic_and_signs():
    zero = _cached("ctrl_ontarget", lambda: lesson.run_c(
        "ctrldump", "--model", lesson.MODEL_REL, "--kp", "40", "--kd", "3",
        "--target", "0.8,-1.2", "--qpos", "0.8,-1.2", "--qvel", "0,0"))
    for i, (idx, c, _lo, _hi) in enumerate(lesson.rows(zero, "ctrl")):
        assert abs(c) < 1e-12, (
            f"on target and at rest, actuator {i} asks for {c!r} — the error (target - qpos) "
            "is zero there, so a non-zero command means a sign error or a stray gravity term")

    # Position term alone, then position and velocity together, so the two gains are graded
    # separately and a swapped kp/kd cannot pass.
    p_only = lesson.rows(lesson.run_c(
        "ctrldump", "--model", lesson.MODEL_REL, "--kp", "10", "--kd", "2",
        "--target", "0.8,-1.2", "--qpos", "0.7,-1.2", "--qvel", "0,0"), "ctrl")
    assert abs(p_only[0][1] - 1.0) < 1e-9, (
        f"error +0.1 rad at kp = 10 with zero velocity must give 1.0; got {p_only[0][1]!r}")
    both = lesson.rows(lesson.run_c(
        "ctrldump", "--model", lesson.MODEL_REL, "--kp", "10", "--kd", "2",
        "--target", "0.8,-1.2", "--qpos", "0.7,-1.2", "--qvel", "0.2,0"), "ctrl")
    assert abs(both[0][1] - 0.6) < 1e-9, (
        f"error +0.1 rad, velocity +0.2 rad/s, kp = 10, kd = 2 gives 10*0.1 - 2*0.2 = 0.6; "
        f"got {both[0][1]!r}. The damping term is subtracted and reads qvel directly.")
    # A negative error, to catch an abs() or a flipped sign that the positive case misses.
    neg = lesson.rows(lesson.run_c(
        "ctrldump", "--model", lesson.MODEL_REL, "--kp", "10", "--kd", "2",
        "--target", "0.8,-1.2", "--qpos", "0.9,-1.2", "--qvel", "0,0"), "ctrl")
    assert abs(neg[0][1] - (-1.0)) < 1e-9, (
        f"sitting 0.1 rad ABOVE target at kp = 10 must give -1.0, not {neg[0][1]!r} — the "
        "command has to push back down")

    # The damping term must read qvel at the DOF address, not at the qpos address. Every case
    # above runs on the pinned arm, where the two agree, so `d->qvel[qadr]` passes all of
    # them — the exact bug this lesson exists to prevent. On the floating arm the shoulder's
    # qpos address is 7 and its dof address 6, and qvel[7] is the ELBOW's velocity. Both
    # hinges are put exactly on target here, so the kp term is zero and the command IS the
    # damping term. Only actuator 0 is asserted against the wrong-address value in the hint:
    # for actuator 1 the qpos address is 8 against nv = 8, so the wrong read runs off the end
    # of the array and its value is undefined — what is asserted is the correct answer.
    fm = mujoco.MjModel.from_xml_path(str(lesson.FLOATING_PATH))
    j0 = int(fm.actuator_trnid[0, 0])
    q0, v0 = int(fm.jnt_qposadr[j0]), int(fm.jnt_dofadr[j0])
    assert q0 != v0, (
        "the floating model must give actuator 0 two different addresses, or this case "
        "proves nothing")
    onfloat = lesson.rows(lesson.run_c(
        "ctrldump", "--model", lesson.FLOATING_REL, "--kp", "10", "--kd", "2",
        "--target", "0.8,-1.2", "--qpos", "0,0,0.8,1,0,0,0,0.8,-1.2",
        "--qvel", "0,0,0,0,0,0,0.1,0.3"), "ctrl")
    assert len(onfloat) == 2, (
        f"ctrldump on the floating arm printed {len(onfloat)} rows, expected 2")
    assert abs(onfloat[0][1] - (-0.2)) < 1e-9, (
        f"on the floating arm, with both hinges exactly on target, the shoulder's own "
        f"velocity is qvel[{v0}] = 0.1 rad/s, so at kd = 2 the command is -0.2; you returned "
        f"{onfloat[0][1]!r}. Reading qvel[{q0}] — the QPOS address — picks up the elbow's "
        "0.3 rad/s and gives -0.6 instead. The damping term must index qvel with the address "
        "from jnt_dofadr.")
    assert abs(onfloat[1][1] - (-0.6)) < 1e-9, (
        f"the elbow's own velocity is qvel[{int(fm.jnt_dofadr[int(fm.actuator_trnid[1, 0])])}]"
        f" = 0.3 rad/s, so at kd = 2 its command is -0.6; you returned {onfloat[1][1]!r}")


def test_run_loop_steps_exactly_once_per_tick():
    met = lesson.metrics(_cached("run_default", _run_text))
    ticks = lesson.TICKS
    want = ticks * met["timestep"]
    assert met["ticks_done"] == ticks, (
        f"stats_tick ran {met['ticks_done']:.0f} times for {ticks} ticks")
    assert abs(met["sim_seconds"] - want) < 1e-9, (
        f"{ticks} ticks left the clock at {met['sim_seconds']!r}, expected {want!r} — that is "
        f"{met['sim_seconds'] / met['timestep']:.1f} steps, not {ticks}")
    # A different tick count must move the clock proportionally: catches a hard-coded loop.
    half = lesson.metrics(_cached("run_half", lambda: _run_text(ticks=400)))
    assert half["ticks_done"] == 400 and abs(half["sim_seconds"] - 400 * half["timestep"]) < 1e-9, (
        f"asked for 400 ticks, got {half['ticks_done']:.0f} and a clock of "
        f"{half['sim_seconds']!r} — the loop must honour its ticks argument")


def test_run_loop_applies_the_control_it_computes():
    met = lesson.metrics(_cached("run_default", _run_text))
    ref = _reference_loop(lesson.TARGET, lesson.KP, lesson.KD, lesson.TICKS)
    assert met["final_abs_err"] < 0.1, (
        f"the arm finished {met['final_abs_err']:.4f} rad from target, having started about "
        "1.2 rad away — if it barely moved, ctrl never reached d->ctrl")
    gap = abs(met["final_abs_err"] - ref["final_abs_err"])
    assert gap < 1e-12, (
        f"the C loop ended {met['final_abs_err']!r} from target where an independent Python "
        f"loop of the same law ended {ref['final_abs_err']!r} (gap {gap:.3e}) — control from "
        "the current state, then write d->ctrl, then step, exactly once")
    assert met["saturated_ticks"] == ref["saturated_ticks"], (
        f"the C loop counted {met['saturated_ticks']:.0f} saturated ticks against the "
        f"reference's {ref['saturated_ticks']} — the clamp is firing at a different rate")


def test_python_loop_matches_c_bit_for_bit():
    model, data = lesson.fresh()
    mine = lesson.python_pd_loop(model, data, lesson.TARGET, lesson.KP, lesson.KD,
                                 lesson.TICKS, trace_every=25)
    expected = {"ticks_done", "sim_seconds", "final_abs_err", "saturated_ticks", "trace"}
    assert set(mine) == expected, f"keys were {sorted(mine)}, expected {sorted(expected)}"
    ref = _reference_loop(lesson.TARGET, lesson.KP, lesson.KD, lesson.TICKS, trace_every=25)
    assert mine["ticks_done"] == ref["ticks_done"], (
        f"ticks_done was {mine['ticks_done']!r}, expected {ref['ticks_done']!r}")
    assert abs(mine["sim_seconds"] - ref["sim_seconds"]) < 1e-12, (
        f"sim_seconds was {mine['sim_seconds']!r}, expected {ref['sim_seconds']!r}")
    assert abs(mine["final_abs_err"] - ref["final_abs_err"]) < 1e-12, (
        f"final_abs_err was {mine['final_abs_err']!r} against an independent reference of "
        f"{ref['final_abs_err']!r} — set ctrl, then step, then measure")
    assert mine["saturated_ticks"] == ref["saturated_ticks"], (
        f"saturated_ticks was {mine['saturated_ticks']!r}, expected {ref['saturated_ticks']!r} "
        "— count a tick where the APPLIED (clamped) command sits on its bound")
    assert len(mine["trace"]) == len(ref["trace"]), (
        f"recorded {len(mine['trace'])} trace rows, expected {len(ref['trace'])} — one row "
        "when k % trace_every == 0")
    worst = max((abs(a - b) for pr, rr in zip(mine["trace"], ref["trace"])
                 for a, b in zip(pr, rr)), default=0.0)
    assert worst < 1e-12, (
        f"the trace diverges from the reference by {worst:.3e}; rows are "
        "[k, time, q0, q1, v0, v1, c0, c1] recorded after the step")
    # And it must agree with the C binary too, which is the point of the exercise.
    c_trace = lesson.rows(_cached("run_default", _run_text), "trace")
    worst_c = max((abs(a - b) for pr, cr in zip(mine["trace"], c_trace)
                   for a, b in zip(pr, cr)), default=0.0)
    assert worst_c < 1e-12, (
        f"the Python trace and the C trace differ by {worst_c:.3e} — same engine, same start, "
        "same gains: they must agree to the last bit")
    # trace_every = 0 must record nothing, which is what the benchmark relies on.
    model2, data2 = lesson.fresh()
    quiet = lesson.python_pd_loop(model2, data2, lesson.TARGET, lesson.KP, lesson.KD, 50,
                                  trace_every=0)
    assert quiet["trace"] == [], (
        f"with trace_every = 0 the trace must stay empty, got {len(quiet['trace'])} rows")


def test_loop_overhead_arithmetic():
    o = lesson.loop_overhead(1_000_000.0, 200_000.0)
    expected = {"c_tick_us", "py_tick_us", "overhead_us", "ratio", "c_max_hz", "py_max_hz"}
    assert set(o) == expected, f"keys were {sorted(o)}, expected {sorted(expected)}"
    assert o["c_tick_us"] == 1.0, f"c_tick_us was {o['c_tick_us']!r}, expected 1e6/1e6 = 1.0"
    assert o["py_tick_us"] == 5.0, f"py_tick_us was {o['py_tick_us']!r}, expected 5.0"
    assert o["overhead_us"] == 4.0, (
        f"overhead_us was {o['overhead_us']!r}, expected py_tick_us - c_tick_us = 4.0 — "
        "Python's per-tick cost minus C's, in that order")
    assert o["ratio"] == 5.0, f"ratio was {o['ratio']!r}, expected 5.0"
    assert o["c_max_hz"] == 1_000_000 and o["py_max_hz"] == 200_000, (
        f"max_hz came back {o['c_max_hz']!r}/{o['py_max_hz']!r}; one tick per control period "
        "makes the ceiling the throughput itself")
    assert isinstance(o["c_max_hz"], int) and isinstance(o["py_max_hz"], int), (
        f"max_hz must be a whole number of Hz; got {type(o['c_max_hz']).__name__}")

    # Flooring, not rounding: 333333.7 Hz of headroom does not buy a 333334 Hz loop.
    odd = lesson.loop_overhead(1_234_567.8, 333_333.7)
    assert odd["c_max_hz"] == 1_234_567, (
        f"c_max_hz was {odd['c_max_hz']}, expected floor(1234567.8) = 1234567")
    assert odd["py_max_hz"] == 333_333, (
        f"py_max_hz was {odd['py_max_hz']}, expected floor(333333.7) = 333333 — round DOWN")

    # A case where C is slower than Python: overhead must go negative rather than flip sign.
    flipped = lesson.loop_overhead(100_000.0, 400_000.0)
    assert flipped["overhead_us"] == -7.5, (
        f"with C at 10 us/tick and Python at 2.5, overhead_us is 2.5 - 10 = -7.5; you "
        f"returned {flipped['overhead_us']!r}. Do not take an absolute value — the sign is "
        "the information.")
    assert flipped["ratio"] == 0.25, f"ratio was {flipped['ratio']!r}, expected 0.25"


def test_c_outruns_python_on_identical_work():
    c = _cached("bench", lambda: lesson.metrics(lesson.run_c(
        "bench", "--model", lesson.MODEL_REL, "--ticks", str(lesson.BENCH_TICKS))))
    py = lesson.python_throughput(ticks=lesson.BENCH_TICKS)
    assert c["ticks"] == py["ticks"], (
        f"the two benchmarks did not do the same work: C {c['ticks']}, Python {py['ticks']}")
    ratio = c["ticks_per_second"] / py["ticks_per_second"]
    assert ratio > 1.5, (
        f"C managed {c['ticks_per_second']:,.0f} ticks/s against Python's "
        f"{py['ticks_per_second']:,.0f} (ratio {ratio:.2f}) — on the same mj_step calls the "
        "compiled loop should be clearly ahead; if it is not, the Python loop is probably not "
        "doing the same work")
    # Same physics, both languages: the endpoint must agree regardless of speed.
    assert abs(c["final_abs_err"] - py["final_abs_err"]) < 1e-12, (
        f"the benchmark runs ended at different states ({c['final_abs_err']!r} vs "
        f"{py['final_abs_err']!r}) — they are supposed to be the same computation")


RUBRIC = [
    (test_c_builds, 3,
     "run `make` in the lesson directory and read the first compiler error"),
    (test_c_selftest_passes, 5,
     "run `make test` — it names the unfinished exercise and what it expects"),
    (test_addresses_on_pinned_arm, 3,
     "actuator_trnid gives the joint; jnt_qposadr and jnt_dofadr give the two addresses"),
    (test_addresses_on_floating_arm, 6,
     "a free joint takes 7 qpos slots and 6 qvel rows — read each address from its own "
     "table rather than reusing one"),
    (test_pd_ctrl_clamps_to_each_actuators_own_range, 5,
     "clamp with m->actuator_ctrlrange[2*i] and [2*i + 1], per actuator, and only when "
     "m->actuator_ctrllimited[i]"),
    (test_pd_ctrl_arithmetic_and_signs, 5,
     "kp*(target - qpos) - kd*qvel: the error is target minus position, the damping term is "
     "subtracted, and qvel is indexed by the DOF address rather than the qpos one"),
    (test_run_loop_steps_exactly_once_per_tick, 4,
     "one mj_step per tick, and honour the ticks argument rather than a constant"),
    (test_run_loop_applies_the_control_it_computes, 3,
     "copy ctrl[i] into d->ctrl[i] before stepping — d->ctrl is the engine's only input"),
    (test_python_loop_matches_c_bit_for_bit, 6,
     "mirror run_loop exactly: commands from the current state, write ctrl, step once, "
     "record after the step"),
    (test_loop_overhead_arithmetic, 5,
     "us per tick is 1e6/ticks_per_second; overhead is Python's minus C's; max_hz floors"),
    (test_c_outruns_python_on_identical_work, 2,
     "both loops must run the same number of ticks for the ratio to mean anything"),
]
