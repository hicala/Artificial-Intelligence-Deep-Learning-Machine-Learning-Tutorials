# %% [markdown]
# # F15-L06 · The control loop, in C — REFERENCE SOLUTION
#
# This is `lesson.py` with the two Python stubs filled in, driving the reference C build
# rather than the student one. The two files therefore differ only in the bodies of
# `python_pd_loop` and `loop_overhead`, in the build paths set in the setup cell, and in the
# answer key at the foot of this file.
#
# **You will build:** a PD control loop written against MuJoCo's own C API — the address
# arithmetic, the control law and the fixed-step loop — plus the measurement that says how
# much of the gap between C and Python is physics and how much is the trip through Python.
#
# **Time:** ~70 minutes · **Runs on:** a laptop CPU, no GPU, no download
# · **Prerequisites:** F15-L01 (mjModel vs mjData), F15-L03 (PD control in Python)
#
# By the end you will be able to:
# 1. Implement a joint's `qpos` and `qvel` address lookup through `mjModel`'s own tables, and
#    show that the two addresses differ on a model with a free joint.
# 2. Implement a clamped PD control law over the actuator array in C and pass the binary's
#    self-test.
# 3. Implement a fixed-step control loop in C and show it advances simulated time by exactly
#    one timestep per tick.
# 4. Measure the same loop's throughput in C and in Python and report the per-tick overhead
#    that separates them.
# 5. Explain why a PD law leaves a steady-state error, from your own measurement of it.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import math
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

print("mujoco", mujoco.__version__, "· numpy", np.__version__, "· python",
      sys.version.split()[0])

_IS_MAIN = __name__ == "__main__"

# This file sits one directory below the lesson, and drives the reference C build.
try:
    LESSON_DIR = Path(__file__).resolve().parent.parent
except NameError:  # a notebook has no __file__
    LESSON_DIR = Path.cwd().parent

C_SRC = "solutions/lesson.c"
BIN = "solutions/lesson_bin"
MODEL_REL = "assets/arm2.xml"
FLOATING_REL = "assets/arm2_floating.xml"
MODEL_PATH = LESSON_DIR / MODEL_REL
FLOATING_PATH = LESSON_DIR / FLOATING_REL

TARGET = (0.8, -1.2)
KP, KD = 40.0, 3.0
TICKS = 1500
BENCH_TICKS = 40_000
SETTLE_TOL = 0.05

_BUILD = None


def build(verbose: bool = True):
    """Compile the C source with make. Cached: the compiler runs once per session."""
    global _BUILD
    if _BUILD is None:
        proc = subprocess.run(
            ["make", "-C", str(LESSON_DIR), f"PYTHON={sys.executable}",
             f"SRC={C_SRC}", f"BIN={BIN}"],
            capture_output=True, text=True, timeout=600)
        _BUILD = (proc.returncode == 0, (proc.stdout + proc.stderr).strip())
    ok, out = _BUILD
    if verbose:
        print("build OK" if ok else "BUILD FAILED\n" + out)
    return ok, out


def run_c(*args, timeout: int = 300) -> str:
    """Run the compiled binary and return its stdout."""
    ok, out = build(verbose=False)
    if not ok:
        raise RuntimeError("the C build failed; run build() to see the compiler output\n" + out)
    proc = subprocess.run([str(LESSON_DIR / BIN), *args],
                          cwd=LESSON_DIR, capture_output=True, text=True, timeout=timeout)
    if proc.returncode == 2:
        raise NotImplementedError(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"{BIN} {' '.join(args)} exited {proc.returncode}\n{proc.stderr.strip()}")
    return proc.stdout


def metrics(text: str) -> dict:
    """Parse the binary's `@ key=value` report lines into a dict of floats."""
    return {k: float(v) for k, v in
            (line[2:].split("=", 1) for line in text.splitlines() if line.startswith("@ "))}


def rows(text: str, tag: str) -> list:
    """Parse the binary's `<tag> v1 v2 ...` lines into a list of lists of floats."""
    return [[float(v) for v in line.split()[1:]]
            for line in text.splitlines() if line.startswith(tag + " ")]


def make_test():
    """Run `make test` — the C self-test. Returns (exit code, combined output)."""
    proc = subprocess.run(
        ["make", "-C", str(LESSON_DIR), f"PYTHON={sys.executable}",
         f"SRC={C_SRC}", f"BIN={BIN}", "test"],
        capture_output=True, text=True, timeout=600)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


if _IS_MAIN:
    build()

# %% [markdown]
# ## 1. Why this one is in C

# %%
def model_facts() -> dict:
    """What the two shipped models contain. Read from the files, not typed here."""
    out = {}
    for name, path in (("pinned", MODEL_PATH), ("floating", FLOATING_PATH)):
        m = mujoco.MjModel.from_xml_path(str(path))
        out[name] = {"nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
                     "njnt": int(m.njnt), "timestep": float(m.opt.timestep)}
    return out


if _IS_MAIN:
    _facts = model_facts()
    for _n, _f in _facts.items():
        print(f"  {_n:9s} nq={_f['nq']} nv={_f['nv']} nu={_f['nu']} njnt={_f['njnt']} "
              f"timestep={_f['timestep']} s -> {1 / _f['timestep']:.0f} Hz control rate")
    print(f"\n  the pinned arm has nq == nv ({_facts['pinned']['nq']}); the floating one does "
          f"not ({_facts['floating']['nq']} vs {_facts['floating']['nv']}).")

# %%
if _IS_MAIN:
    _code, _out = make_test()
    print(_out)
    print(f"exit code {_code}  (0 = all pass, 2 = something is still a stub, 1 = a real failure)")

# %% [markdown]
# ## 3. Exercise 1 — `actuator_addresses` in the C source

# %%
def _check_addresses() -> None:
    pinned = run_c("addrdump", "--model", MODEL_REL)
    floating = run_c("addrdump", "--model", FLOATING_REL)
    p_head, f_head = metrics(pinned), metrics(floating)
    p_rows, f_rows = rows(pinned, "addr"), rows(floating, "addr")

    assert len(p_rows) == int(p_head["nu"]) and len(f_rows) == int(f_head["nu"]), (
        "addrdump printed a different number of rows than the model has actuators")
    for i, (_, joint, qadr, vadr, _jt) in enumerate(p_rows):
        assert (qadr, vadr) == (i, i), (
            f"on the pinned arm actuator {i} should map to qpos {i} / qvel {i}, but you "
            f"returned qpos {qadr:.0f} / qvel {vadr:.0f}")
    for i, (_, joint, qadr, vadr, _jt) in enumerate(f_rows):
        assert qadr == vadr + 1, (
            f"on the floating arm actuator {i} came back at qpos {qadr:.0f} / qvel "
            f"{vadr:.0f}. The free joint ahead of it takes 7 qpos slots but only 6 qvel rows, "
            "so its qpos address must be exactly one MORE than its qvel address.")
    assert f_head["nq"] - f_head["nv"] == 1, (
        "the floating model should report nq one larger than nv")
    print(f"exercise 1 looks right: pinned arm maps "
          f"{[(int(r[2]), int(r[3])) for r in p_rows]}, floating arm maps "
          f"{[(int(r[2]), int(r[3])) for r in f_rows]} (qpos, qvel)")
    print(f"  the floating model's free joint costs {int(f_head['nq'])} qpos slots against "
          f"{int(f_head['nv'])} qvel rows — that 1 is the whole reason the lookup exists")


if _IS_MAIN:
    _check_addresses()


# %% [markdown]
# ## 4. Exercise 2 — `pd_ctrl` in the C source

# %%
def _check_pd_ctrl() -> None:
    at_start = rows(run_c("ctrldump", "--model", MODEL_REL, "--kp", "40", "--kd", "3",
                          "--target", "0.8,-1.2"), "ctrl")
    for i, (_, c, lo, hi) in enumerate(at_start):
        assert lo - 1e-12 <= c <= hi + 1e-12, (
            f"actuator {i} came back at {c:g}, outside its own ctrlrange [{lo:g}, {hi:g}]")
    assert abs(at_start[0][1] - at_start[0][3]) < 1e-12, (
        f"the shoulder must come back at exactly {at_start[0][3]:g}; got {at_start[0][1]:g}")
    assert abs(at_start[1][1] - at_start[1][2]) < 1e-12, (
        f"the elbow must clamp to exactly {at_start[1][2]:g}; got {at_start[1][1]:g}")

    on_target = rows(run_c("ctrldump", "--model", MODEL_REL, "--kp", "40", "--kd", "3",
                           "--target", "0.8,-1.2", "--qpos", "0.8,-1.2", "--qvel", "0,0"),
                     "ctrl")
    for i, (_, c, _lo, _hi) in enumerate(on_target):
        assert abs(c) < 1e-12, (
            f"with the arm exactly on target and at rest, actuator {i} still asks for {c:g}")

    small = rows(run_c("ctrldump", "--model", MODEL_REL, "--kp", "10", "--kd", "2",
                       "--target", "0.8,-1.2", "--qpos", "0.7,-1.2", "--qvel", "0.2,0"),
                 "ctrl")
    want = 10.0 * 0.1 - 2.0 * 0.2
    assert abs(small[0][1] - want) < 1e-9, (
        f"expected 10*0.1 - 2*0.2 = {want:g}; you returned {small[0][1]:g}")
    print(f"exercise 2 looks right: from the keyframe the law asks for 32 and -48 and gets "
          f"{at_start[0][1]:g} and {at_start[1][1]:g} after clamping; on target it asks for "
          f"exactly 0; a 0.1 rad error at kp = 10 gives {small[0][1]:.3f}")


if _IS_MAIN:
    _check_pd_ctrl()


# %% [markdown]
# ## 5. Exercise 3 — `run_loop` in the C source

# %%
def _check_run_loop() -> None:
    text = run_c("run", "--model", MODEL_REL, "--kp", str(KP), "--kd", str(KD),
                 "--ticks", str(TICKS))
    met = metrics(text)
    want_time = TICKS * met["timestep"]
    assert met["ticks_done"] == TICKS, (
        f"stats_tick ran {met['ticks_done']:.0f} times over {TICKS} ticks")
    assert abs(met["sim_seconds"] - want_time) < 1e-9, (
        f"after {TICKS} ticks the clock reads {met['sim_seconds']:.6f} s, expected "
        f"{want_time:.6f} s")
    assert met["final_abs_err"] < 0.1, (
        f"the arm finished {met['final_abs_err']:.3f} rad from target")
    assert met["settle_tick"] >= 0, "the arm never settled inside the tolerance band"
    assert 0.0 < met["saturated_fraction"] < 0.5, (
        f"{met['saturated_fraction']:.1%} of ticks were saturated")
    print(f"exercise 3 looks right: {met['ticks_done']:.0f} ticks = "
          f"{met['sim_seconds']:.3f} s simulated, settled at tick {met['settle_tick']:.0f} "
          f"({met['settle_seconds']:.2f} s), saturated for "
          f"{met['saturated_fraction']:.1%} of the run")


if _IS_MAIN:
    _check_run_loop()


# %% [markdown]
# ## 6. The droop, and why it is not a bug

# %%
def droop_report() -> list:
    """Per-actuator final error against qfrc_bias / kp, measured from the C run."""
    text = run_c("run", "--model", MODEL_REL, "--kp", str(KP), "--kd", str(KD),
                 "--ticks", str(TICKS))
    out = []
    for i, err, bias, bias_over_kp, ctrl in rows(text, "err"):
        out.append({"actuator": int(i), "final_error": err, "qfrc_bias": bias,
                    "bias_over_kp": bias_over_kp, "ctrl": ctrl,
                    "ratio": err / bias_over_kp if bias_over_kp else float("nan")})
    return out


if _IS_MAIN:
    print(f"  {'act':>3} {'final error':>13} {'qfrc_bias':>11} {'bias/kp':>13} {'ratio':>8}")
    for _r in droop_report():
        print(f"  {_r['actuator']:>3} {_r['final_error']:>13.9f} {_r['qfrc_bias']:>11.4f} "
              f"{_r['bias_over_kp']:>13.9f} {_r['ratio']:>8.4f}")
    print("\n  the last column is the measurement: steady-state error IS the gravitational "
          "load divided by kp.")

# %% [markdown]
# ## 7. Exercise 4 — the same loop, in Python

# %%
def py_addresses(model, i: int):
    """The Python mirror of actuator_addresses() — (joint, qpos address, qvel address)."""
    joint = int(model.actuator_trnid[i, 0])
    return joint, int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def py_at_limit(model, ctrl) -> int:
    """1 if any command is sitting on its own ctrlrange bound. Mirrors at_limit()."""
    for i in range(int(model.nu)):
        if not model.actuator_ctrllimited[i]:
            continue
        lo, hi = model.actuator_ctrlrange[i]
        if ctrl[i] >= hi - 1e-12 or ctrl[i] <= lo + 1e-12:
            return 1
    return 0


def py_max_abs_error(model, data, target) -> float:
    """Largest distance any actuated joint is from its target. Mirrors max_abs_error()."""
    worst = 0.0
    for i in range(int(model.nu)):
        _j, qadr, _v = py_addresses(model, i)
        worst = max(worst, abs(target[i] - float(data.qpos[qadr])))
    return worst


def fresh():
    """A (model, data) pair reset to the model's first keyframe, exactly as reset_start()."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def python_pd_loop(model, data, target, kp: float, kd: float, ticks: int,
                   trace_every: int = 0) -> dict:
    """Run the PD control loop in Python, exactly as run_loop() does in the C source.

    REFERENCE IMPLEMENTATION.

    The order is the whole point, and it mirrors the C exactly: commands from the CURRENT
    state, then write them into data.ctrl, then one mj_step, then record. Doing the same
    arithmetic in the same order is what makes the two trajectories agree bit for bit — which
    in turn is what licenses the claim that the timing difference in section 9 is overhead
    rather than different physics.

    Returns:
        dict with exactly these five keys: ticks_done, sim_seconds, final_abs_err,
        saturated_ticks, trace.
    """
    nu = int(model.nu)
    # Resolve the addresses once, outside the loop. They are a property of mjModel, which is
    # constant, so looking them up every tick would be work for nothing.
    adr = [py_addresses(model, i)[1:] for i in range(nu)]
    limited = [bool(model.actuator_ctrllimited[i]) for i in range(nu)]
    bounds = [(float(model.actuator_ctrlrange[i][0]), float(model.actuator_ctrlrange[i][1]))
              for i in range(nu)]

    ctrl = [0.0] * nu
    saturated = 0
    trace = []
    for k in range(ticks):
        # 1. commands from the current state
        for i in range(nu):
            qadr, vadr = adr[i]
            tau = kp * (target[i] - float(data.qpos[qadr])) - kd * float(data.qvel[vadr])
            if limited[i]:
                lo, hi = bounds[i]
                tau = min(max(tau, lo), hi)
            ctrl[i] = tau
        # 2. hand them to the engine
        for i in range(nu):
            data.ctrl[i] = ctrl[i]
        # 3. advance the physics exactly once
        mujoco.mj_step(model, data)
        # 4. record what just happened
        saturated += py_at_limit(model, ctrl)
        if trace_every > 0 and k % trace_every == 0:
            row = [float(k), float(data.time)]
            row += [float(data.qpos[adr[i][0]]) for i in range(2)]
            row += [float(data.qvel[adr[i][1]]) for i in range(2)]
            row += [ctrl[i] for i in range(2)]
            trace.append(row)

    return {"ticks_done": ticks,
            "sim_seconds": float(data.time),
            "final_abs_err": py_max_abs_error(model, data, target),
            "saturated_ticks": saturated,
            "trace": trace}


def _check_python_loop() -> None:
    model, data = fresh()
    mine = python_pd_loop(model, data, TARGET, KP, KD, TICKS, trace_every=25)
    expected = {"ticks_done", "sim_seconds", "final_abs_err", "saturated_ticks", "trace"}
    assert set(mine) == expected, f"keys were {sorted(mine)}, expected {sorted(expected)}"
    theirs = metrics(run_c("run", "--model", MODEL_REL, "--kp", str(KP), "--kd", str(KD),
                           "--ticks", str(TICKS)))
    assert mine["ticks_done"] == theirs["ticks_done"]
    assert abs(mine["sim_seconds"] - theirs["sim_seconds"]) < 1e-9
    gap = abs(mine["final_abs_err"] - theirs["final_abs_err"])
    assert gap < 1e-12, (
        f"Python ended {mine['final_abs_err']!r} and C ended {theirs['final_abs_err']!r} "
        f"(gap {gap:.2e})")
    assert mine["saturated_ticks"] == theirs["saturated_ticks"]
    c_trace = rows(run_c("run", "--model", MODEL_REL, "--kp", str(KP), "--kd", str(KD),
                         "--ticks", str(TICKS)), "trace")
    assert len(mine["trace"]) == len(c_trace)
    worst = max((abs(a - b) for pr, cr in zip(mine["trace"], c_trace) for a, b in zip(pr, cr)),
                default=0.0)
    assert worst < 1e-12, f"the two trajectories diverge by {worst:.3e}"
    print(f"exercise 4 looks right: {len(c_trace)} sampled ticks agree between C and Python "
          f"to within {worst:.1e}; both end {mine['final_abs_err']:.9f} rad from target")


if _IS_MAIN:
    _check_python_loop()


# %% [markdown]
# ## 8. Exercise 5 — what the gap between them actually is

# %%
def loop_overhead(c_ticks_per_second: float, py_ticks_per_second: float) -> dict:
    """Turn two measured throughputs into the per-tick cost that separates them.

    REFERENCE IMPLEMENTATION.

    The subtraction is the interesting line. A ratio says "C is 8x faster", which does not
    transfer to a different model — put contact in the scene and mj_step gets dearer while the
    Python call overhead stays where it is, so the ratio collapses even though nothing about
    the binding layer changed. The DIFFERENCE is the quantity that travels: a roughly fixed
    number of microseconds per tick, which you can then compare against a control period.

    floor(), not round(): a loop that can sustain 333333.7 ticks per second cannot close a
    333334 Hz control loop, because the tick it cannot finish is a tick it missed.
    """
    c_tick_us = 1e6 / c_ticks_per_second
    py_tick_us = 1e6 / py_ticks_per_second
    return {"c_tick_us": c_tick_us,
            "py_tick_us": py_tick_us,
            "overhead_us": py_tick_us - c_tick_us,
            "ratio": c_ticks_per_second / py_ticks_per_second,
            "c_max_hz": math.floor(c_ticks_per_second),
            "py_max_hz": math.floor(py_ticks_per_second)}


def _check_overhead() -> None:
    o = loop_overhead(1_000_000.0, 200_000.0)
    expected = {"c_tick_us", "py_tick_us", "overhead_us", "ratio", "c_max_hz", "py_max_hz"}
    assert set(o) == expected, f"keys were {sorted(o)}, expected {sorted(expected)}"
    assert o["c_tick_us"] == 1.0 and o["py_tick_us"] == 5.0
    assert o["overhead_us"] == 4.0
    assert o["ratio"] == 5.0
    assert o["c_max_hz"] == 1_000_000 and isinstance(o["c_max_hz"], int)
    odd = loop_overhead(1_234_567.8, 333_333.7)
    assert odd["py_max_hz"] == 333_333
    assert odd["c_max_hz"] == 1_234_567
    print(f"exercise 5 looks right: at 1e6 vs 2e5 ticks/s the per-tick overhead is "
          f"{o['overhead_us']:.1f} us and the ratio is {o['ratio']:.1f}x")


if _IS_MAIN:
    _check_overhead()


# %% [markdown]
# ## 9. The payoff — measure the language, not the folklore

# %%
def python_throughput(ticks: int = BENCH_TICKS) -> dict:
    """Time the Python loop on exactly the work `bench` does in C."""
    model, data = fresh()
    t0 = time.perf_counter()
    out = python_pd_loop(model, data, TARGET, KP, KD, ticks, trace_every=0)
    wall = time.perf_counter() - t0
    return {"ticks": ticks, "wall_seconds": wall, "ticks_per_second": ticks / wall,
            "us_per_tick": wall * 1e6 / ticks, "final_abs_err": out["final_abs_err"]}


def report() -> None:
    run = metrics(run_c("run", "--model", MODEL_REL, "--kp", str(KP), "--kd", str(KD),
                        "--ticks", str(TICKS)))
    print("the control run")
    print(f"  {run['ticks']:.0f} ticks at {run['timestep']} s = {run['sim_seconds']:.2f} s "
          f"simulated, kp = {run['kp']:.0f}, kd = {run['kd']:.0f}")
    print(f"  settled at tick {run['settle_tick']:.0f} ({run['settle_seconds']:.2f} s), "
          f"saturated for {run['saturated_fraction']:.1%} of the run")
    print(f"  finished {run['final_abs_err']:.9f} rad from target")

    c = metrics(run_c("bench", "--model", MODEL_REL, "--ticks", str(BENCH_TICKS)))
    py = python_throughput()
    o = loop_overhead(c["ticks_per_second"], py["ticks_per_second"])
    print(f"\nidentical work, two languages: {int(c['ticks']):,} ticks of the same loop")
    print(f"  C      {c['wall_seconds'] * 1e3:8.1f} ms  {c['ticks_per_second']:12,.0f} ticks/s"
          f"  {c['us_per_tick']:7.3f} us/tick")
    print(f"  Python {py['wall_seconds'] * 1e3:8.1f} ms  {py['ticks_per_second']:12,.0f} ticks/s"
          f"  {py['us_per_tick']:7.3f} us/tick")
    print(f"  ratio {o['ratio']:.1f}x · per-tick overhead {o['overhead_us']:.2f} us")
    print(f"\nthe control rate each loop could close, one tick per period")
    print(f"  C      up to {o['c_max_hz']:,} Hz")
    print(f"  Python up to {o['py_max_hz']:,} Hz")
    print(f"  both agree on the physics: C ended {c['final_abs_err']:.12f} rad from target, "
          f"Python {py['final_abs_err']:.12f}")

    print(f"\nwhere each loop's ceiling actually is, one tick per control period")
    print(f"  {'rate':>12} {'budget/tick':>14} {'C':>10} {'Python':>10}")
    for hz in (1_000, 10_000, 100_000, 500_000, 1_000_000):
        budget_us = 1e6 / hz
        print(f"  {hz:>9,} Hz {budget_us:>11.3f} us "
              f"{'fits' if o['c_tick_us'] <= budget_us else 'MISSES':>10} "
              f"{'fits' if o['py_tick_us'] <= budget_us else 'MISSES':>10}")
    print("\n  Read the top row first: at any rate a real robot actually runs, BOTH loops fit "
          "with\n  room to spare, and choosing C for a 1 kHz joint controller would buy you "
          "nothing.\n  The overhead only decides anything when one control period has to buy "
          "MANY ticks —\n  which is exactly what a sampling planner does, and what F15-L07 is "
          "about.")


# %% [markdown]
# ## 10. Common mistakes — the address confusion, made concrete

# %%
if _IS_MAIN:
    _fm = mujoco.MjModel.from_xml_path(str(FLOATING_PATH))
    _fd = mujoco.MjData(_fm)
    mujoco.mj_resetDataKeyframe(_fm, _fd, 0)
    _fd.qvel[:] = np.arange(1, int(_fm.nv) + 1) / 10.0
    mujoco.mj_forward(_fm, _fd)
    print(f"  the floating arm has nq={int(_fm.nq)} qpos slots but only "
          f"nv={int(_fm.nv)} qvel rows")
    print(f"  qvel = {np.array(_fd.qvel).round(2).tolist()}")
    for _i in range(int(_fm.nu)):
        _j, _q, _v = py_addresses(_fm, _i)
        if _q < int(_fm.nv):
            _wrong = (f"qvel[qadr={_q}] = {_fd.qvel[_q]:.2f}  <- a DIFFERENT joint's velocity")
        else:
            _wrong = (f"qvel[qadr={_q}] is past the end of an {int(_fm.nv)}-row array")
        print(f"  actuator {_i}: correct qvel[{_v}] = {_fd.qvel[_v]:.2f}   but {_wrong}")
    print("\n  two different failures from one mistake: the first actuator silently damps "
          "against\n  another joint, the second indexes past the end of the array entirely.")
    print("  numpy raises on the second. C does not — qvel is a bare double*, so the same "
          "line\n  reads whatever happens to sit after the array and the loop carries on.")

# %% [markdown]
# ## 11. Self-check — answered

# %%
import hashlib

SELF_CHECK = {1: "b", 2: "b", 3: "c", 4: "b", 5: "c"}

_ANSWER_DIGESTS = {1: "f89c23d8b9c4a2fb", 2: "80b50a5e0d4d2a4d", 3: "cf6be7fe6ffd7a1c",
                   4: "9e00f43925593a9b", 5: "c9c745b3f2ba419c"}
_ANSWER_SECTIONS = {
    1: "section 2 — which of the two structs is the robot and which is the moment",
    2: "section 3 and the address table your own code printed for the floating arm",
    3: "section 5 — the four numbered steps, and why their order is the exercise",
    4: "section 8 — what the bindings are documented to be, and section 9's us/tick column",
    5: "section 6 — the ratio column you measured, not the folklore about PD control",
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the five multiple-choice answers in SELF_CHECK, naming where to look again."""
    answers = SELF_CHECK if answers is None else answers
    wrong = []
    for q, digest in sorted(_ANSWER_DIGESTS.items()):
        got = str(answers.get(q, "?")).strip().lower()
        if hashlib.sha256(f"F15-L06-q{q}-{got}".encode()).hexdigest()[:16] != digest:
            wrong.append(q)
    for q in sorted(_ANSWER_DIGESTS):
        note = f"  -> re-read {_ANSWER_SECTIONS[q]}" if q in wrong else ""
        print(f"  q{q}: {'wrong' if q in wrong else 'right'}{note}")
    assert not wrong, (
        f"questions {wrong} are still wrong. Each one names the section that answers it "
        "above — go back to the measurement you ran there rather than guessing a letter.")
    print("self-check: all five right")


# %%
if __name__ == "__main__":
    _check_self_check()
    report()

# %% [markdown]
# ## Self-check answer key
#
# **1 — (b).** One `mjModel`, eight `mjData`. MuJoCo's documentation is explicit that the
# split exists for exactly this: `mjModel` "is expected to remain constant" while `mjData`
# "contains all dynamic variables and intermediate results", and the separation "makes it easy
# to simulate multiple models as well as multiple states and controls for each model, in turn
# facilitating multi-threading for sampling" (`claims.yaml`, `mjmodel-constant-mjdata-scratch`).
# (a) wastes memory duplicating a constant and loses cache sharing. (c) is backwards: the
# shared thing would be the mutable one, so the eight threads would corrupt each other. (d)
# serialises the very thing you parallelised. This is the structure F15-L07 relies on.
#
# **2 — (b).** The free joint takes seven `qpos` slots (three translation plus a four-entry
# quaternion) against six `qvel` rows, so every joint after it sits at two different addresses
# — which your own `addrdump` printed as qpos 7/8 against dof 6/7. Before the base was added,
# `nq == nv` and the two agreed, so the bug was invisible for a year. It is silent because
# reading a neighbour's velocity is not an error, just a wrong number in the damping term.
# (a) would change how well it tracks, not turn it to nonsense. (c) and (d) are not what
# changed. This is why six of this lesson's points ride on the floating model.
#
# **3 — (c).** A one-tick delay. Every command is computed from state *k* and then applied
# across the step from *k+1* to *k+2*, so the loop is acting on information one timestep
# stale. (b) is wrong — `d` is not `const` in `run_loop`, and nothing about the reorder is a
# compile error, which is precisely why this bug ships. (d) is wrong because the number of
# `mj_step` calls does not change; the simulated clock still reads `ticks * timestep`, which
# is what makes this invisible to the clock check. It shows up as ringing once the gains are
# high enough for a tick of delay to matter.
#
# **4 — (b).** Both loops call the same compiled `mj_step` in the same `libmujoco`. MuJoCo's
# bindings are documented as "low-level bindings that are meant to give as close to a direct
# access to the MuJoCo library as possible", providing "access to the raw memory used by
# MuJoCo without copying or buffering" (`claims.yaml`, `mujoco-python-bindings-are-thin`).
# Your own trace comparison proves it: the two trajectories agree to the last bit, so the
# physics cannot be different. What differs is the per-call cost around it, which is why the
# honest report is a near-constant microseconds-per-tick figure rather than a multiplier — put
# contact in the model and `mj_step` gets dearer while the overhead stays put, collapsing the
# ratio without anything about the bindings changing. (d) is backwards: the bindings *release*
# the GIL before calling into MuJoCo.
#
# **5 — (c).** Nothing is wrong. A PD law produces torque only in proportion to error, so to
# hold a gravitational load it must hold a matching error: `error = qfrc_bias / kp`. That is
# the identity your own run printed, and the ratio column came out at 1.000 to twelve
# significant figures — which is what rules out (a), a sign error, and (b), incomplete
# convergence: the arm is not still moving toward the target, it has arrived at the only place
# the law can hold. (d) is unrelated; a smaller timestep integrates the same equilibrium more
# accurately. The three real fixes are a higher gain (which shrinks the error but never
# removes it), an integral term, or feeding `qfrc_bias` forward as gravity compensation, which
# is what F15-L03 measured in Python.
