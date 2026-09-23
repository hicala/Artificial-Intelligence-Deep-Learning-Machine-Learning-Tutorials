"""Autograder rubric for F15-L07. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure). Partial credit is the point — a
student who has written the C++ rollout but not yet the refit should see exactly that.

The C++ exercises are graded by running the compiled binary and checking its output against
an independent Python reference implemented in THIS file, so the rubric never trusts the
student's own arithmetic to confirm the student's own arithmetic. A binary that exits 2 has
an unfinished stub; run_cpp turns that into NotImplementedError, which the grader prints as
TODO rather than as an error.
"""
import importlib.util, math, os, random, tempfile
from pathlib import Path

import mujoco
import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

# The objective, restated here independently of the lesson's own constants. If a student
# edits the weights in lesson.py to make their numbers agree, this reference does not move.
_W_ANGLE, _W_CART, _W_THETADOT, _W_XDOT, _W_CTRL = 10.0, 1.0, 0.05, 0.05, 0.01

_CACHE = {}


def _cached(key, fn):
    """Run the expensive C++ commands once per grading session, not once per test."""
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def _reference_cost(controls) -> float:
    """An independent Python rollout of the same objective from the same keyframe."""
    m = mujoco.MjModel.from_xml_path(str(lesson.MODEL_PATH))
    d, s = mujoco.MjData(m), mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mujoco.mj_copyData(s, m, d)
    total = 0.0
    for u in controls:
        u = float(u)
        s.ctrl[0] = u
        mujoco.mj_step(m, s)
        x, th = float(s.qpos[0]), float(s.qpos[1])
        xd, thd = float(s.qvel[0]), float(s.qvel[1])
        total += (_W_ANGLE * (1.0 - math.cos(th)) + _W_CART * x * x
                  + _W_THETADOT * thd * thd + _W_XDOT * xd * xd + _W_CTRL * u * u)
    return total


def _costof(controls) -> dict:
    """Score a control sequence with the student's C++ rollout_cost."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="f15l07_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(repr(float(v)) for v in controls))
        return lesson.metrics(lesson.run_cpp("costof", "--controls", path))
    finally:
        os.unlink(path)


def _refit(costs: str, offset: str = "0", samples: str = "12", horizon: str = "7",
           lam: str | None = None) -> str:
    args = ["refitdump", "--samples", samples, "--horizon", horizon,
            "--costs", costs, "--offset", offset]
    if lam is not None:
        args += ["--lambda", lam]
    return lesson.run_cpp(*args)


# --------------------------------------------------------------------------------------
# The rubric
# --------------------------------------------------------------------------------------

def test_cpp_builds():
    ok, out = lesson.build(verbose=False)
    assert ok, f"the C++ source did not compile:\n{out}"


def test_cpp_selftest_passes():
    code, out = _cached("selftest", lesson.make_test)
    if code == 2:
        raise NotImplementedError("at least one C++ exercise is still a stub")
    assert code == 0, f"`make test` failed with exit {code}:\n{out}"


def test_sampling_statistics():
    text = _cached("sampledump", lambda: lesson.run_cpp(
        "sampledump", "--samples", "500", "--horizon", "9",
        "--sigma", "0.25", "--offset", "0.1", "--seed", "2718"))
    head = lesson.metrics(text)
    block = np.array(lesson.rows(text, "row"))
    assert block.shape == (500, 9), (
        f"sampledump produced {block.shape}, expected (500, 9) — `out` must be sized to "
        "samples*horizon and laid out row-major")
    assert block.min() >= head["lo"] - 1e-12 and block.max() <= head["hi"] + 1e-12, (
        f"sampled controls ran to [{block.min():.4f}, {block.max():.4f}], outside ctrlrange "
        f"[{head['lo']}, {head['hi']}] — clamp after adding the mean")
    col = block.mean(axis=0)
    assert np.all(np.abs(col - 0.1) < 0.06), (
        f"per-step sample means were {np.round(col, 3).tolist()} but the plan being perturbed "
        "was 0.1 everywhere — the noise must be centred on mean[t]")
    assert abs(float(block.std()) - 0.25) < 0.04, (
        f"the sample spread is {block.std():.4f} against sigma = 0.25 — sigma multiplies a "
        "standard normal; it is a standard deviation, not a variance")
    assert not np.allclose(block[0], block[1]) and not np.allclose(block[0], block[7]), (
        "different samples came out identical — one noise draw is being reused across rows")
    assert float(np.abs(np.diff(block[0])).max()) > 1e-12, (
        "one plan is constant across its horizon — the draw is per (sample, step), not per "
        "sample")
    repeat = np.array(lesson.rows(lesson.run_cpp(
        "sampledump", "--samples", "500", "--horizon", "9",
        "--sigma", "0.25", "--offset", "0.1", "--seed", "2718"), "row"))
    assert np.array_equal(block, repeat), (
        "the same seed produced a different buffer — draw from the rng argument, not a new one")
    other = np.array(lesson.rows(lesson.run_cpp(
        "sampledump", "--samples", "500", "--horizon", "9",
        "--sigma", "0.25", "--offset", "0.1", "--seed", "31415"), "row"))
    assert not np.allclose(block, other), (
        "two different seeds produced the same buffer — the rng is being ignored entirely")

    # Row-major, pinned exactly. 500x9 and 9x500 draw the same 4500 numbers in the same order
    # from the same seed, and row-major storage writes the k-th draw at index k in both, so the
    # flattened buffers must match element for element. A transposed layout permutes one of
    # them and not the other, so this is the check that actually sees out[i*horizon + t].
    tall = np.array(lesson.rows(lesson.run_cpp(
        "sampledump", "--samples", "9", "--horizon", "500",
        "--sigma", "0.25", "--offset", "0.1", "--seed", "2718"), "row"))
    assert np.array_equal(block.ravel(), tall.ravel()), (
        "500 samples of a 9-step horizon and 9 samples of a 500-step horizon drew the same "
        "4500 numbers from the same seed but stored them differently — the control for sample "
        "i at step t must land at out[i * horizon + t], row-major. out[t * samples + i] "
        "transposes the buffer, and every row you hand to rollout_cost is then a column")

    # Sigma well past the actuator range: now most draws are outside it, so clamping is not a
    # detail that a lucky seed can hide.
    hard = lesson.run_cpp("sampledump", "--samples", "200", "--horizon", "6",
                          "--sigma", "3.0", "--offset", "0.4", "--seed", "4242")
    hh = lesson.metrics(hard)
    hb = np.array(lesson.rows(hard, "row"))
    assert hb.min() >= hh["lo"] - 1e-12 and hb.max() <= hh["hi"] + 1e-12, (
        f"at sigma = 3 the samples ran to [{hb.min():.4f}, {hb.max():.4f}] against a ctrlrange "
        f"of [{hh['lo']}, {hh['hi']}] — clamp the FINAL value, mean[t] + sigma*z, not the noise "
        "on its own")
    pinned = float(np.mean((np.abs(hb - hh["lo"]) < 1e-12) | (np.abs(hb - hh["hi"]) < 1e-12)))
    assert pinned > 0.5, (
        f"only {pinned:.0%} of the samples sit exactly on an actuator limit, but with sigma = 3 "
        f"against a range of [{hh['lo']}, {hh['hi']}] about three quarters of the draws fall "
        "outside it — a clamp pins those to the limit; rescaling or wrapping does not")


def test_rollout_matches_hidden_reference():
    rng = random.Random(90210)
    plan = [max(-1.0, min(1.0, rng.gauss(0.0, 0.7))) for _ in range(55)]
    got = _costof(plan)
    ref = _reference_cost(plan)
    assert got["horizon"] == 55, (
        f"costof reported horizon {got['horizon']}, expected 55 — the horizon is the number "
        "of controls supplied")
    rel = abs(got["cost"] - ref) / abs(ref)
    assert rel < 1e-9, (
        f"the C++ rollout scored {got['cost']!r} where an independent Python rollout of the "
        f"same plan from the same keyframe scored {ref!r} (relative gap {rel:.3e}) — score "
        "the state AFTER each mj_step, with the control that produced it, summed over the "
        "whole horizon")


def test_rollout_leaves_the_live_state_alone():
    plan = [0.35] * 30
    got = _costof(plan)
    assert abs(got["start_theta_after"] - math.pi) < 1e-12, (
        f"after scoring, the live hinge angle is {got['start_theta_after']!r} but the keyframe "
        "puts it at pi — you stepped `start` instead of the scratch copy, so the planner is "
        "driving the robot with its own hypotheticals")
    assert got["start_time_after"] == 0.0, (
        f"the live simulation clock reads {got['start_time_after']!r} after scoring; scoring a "
        "plan must not advance it")


def test_rollout_invariants():
    zero25 = _costof([0.0] * 25)
    again = _costof([0.0] * 25)
    assert zero25["cost"] == again["cost"], (
        "scoring the same plan twice gave different answers — scratch state is leaking "
        "between calls; branch it from `start` at the top of every call")
    assert zero25["cost"] > 0.0 and math.isfinite(zero25["cost"]), (
        f"a zero-control rollout from hanging scored {zero25['cost']!r}; every weight is "
        "non-negative and the pole is far from upright, so it must be positive and finite")
    zero50 = _costof([0.0] * 50)
    assert zero50["cost"] > zero25["cost"], (
        "doubling the horizon did not increase the cost — you are returning the final stage's "
        "cost instead of the sum over the horizon")
    driven = _costof([1.0] * 25)
    assert driven["cost"] != zero25["cost"], (
        "full throttle scored exactly the same as zero control — the controls argument is "
        "never reaching scratch->ctrl[0]")


def test_refit_invariants():
    eq = _cached("refit_equal", lambda: _refit("equal"))
    head = lesson.metrics(eq)
    mean_eq = lesson.rows(eq, "mean")[0]
    arith = lesson.rows(eq, "arithmetic")[0]
    assert np.allclose(mean_eq, arith, rtol=0.0, atol=1e-12), (
        "with every cost equal, every weight is equal, so the refit must be the plain "
        "arithmetic mean of the sampled plans — check the division by the sum of the weights")

    sp = _cached("refit_spike", lambda: _refit("spike"))
    assert np.allclose(lesson.rows(sp, "mean")[0], lesson.rows(sp, "first")[0], atol=1e-9), (
        "one sample was astronomically cheaper than every other, so the refit should reduce "
        "to that sample — your weights are not decaying with cost")

    small = lesson.rows(_refit("graded", "0"), "mean")[0]
    big = lesson.rows(_refit("graded", "5e5"), "mean")[0]
    assert all(math.isfinite(v) for v in big), (
        "shifting every cost up by 5e5 produced a non-finite plan — exp(-5e5/lambda) "
        "underflows to exactly zero, so every weight vanished and the normalisation divided "
        "by zero; subtract the minimum cost before exponentiating")
    assert np.allclose(small, big, rtol=0.0, atol=1e-12), (
        "adding the same constant to every cost changed the refitted plan, but that constant "
        "cancels between the weights and their sum — subtract the minimum cost first")
    assert not np.allclose(small, arith, rtol=0.0, atol=1e-9), (
        "with graded costs the refit equalled the unweighted mean — the weights are not "
        "responding to the costs at all")
    assert all(head["lo"] - 1e-12 <= v <= head["hi"] + 1e-12 for v in small), (
        "the refitted plan left the actuator range, which a convex combination of in-range "
        "samples cannot do — the weights are not normalised to sum to one")

    # lambda is the temperature, and every check above is blind to it: equal costs, a spike and
    # a shift all come out the same whatever lambda is. These do not. Graded costs run 0..11,
    # so lambda sets how sharply exp(-cost/lambda) discriminates between them.
    hot = lesson.rows(_refit("graded", "0", lam="1e6"), "mean")[0]
    assert np.allclose(hot, arith, rtol=0.0, atol=1e-4), (
        f"at lambda = 1e6 the refit gave {np.round(hot, 4).tolist()} but the unweighted mean of "
        f"the same samples is {np.round(arith, 4).tolist()} — a huge lambda makes every "
        "exp(-(cost-min)/lambda) ~ 1, so the refit must collapse to the arithmetic mean. Divide "
        "the cost difference BY lambda; dropping lambda leaves you permanently at lambda = 1")
    cold = lesson.rows(_refit("graded", "0", lam="0.01"), "mean")[0]
    assert np.allclose(cold, lesson.rows(sp, "first")[0], rtol=0.0, atol=1e-9), (
        f"at lambda = 0.01 the refit gave {np.round(cold, 4).tolist()} but the cheapest sample "
        "is the first one — a tiny lambda makes the softmax an argmin, so the refit must reduce "
        "to that single plan")
    assert not np.allclose(hot, cold, rtol=0.0, atol=1e-6), (
        "lambda = 1e6 and lambda = 0.01 produced the same plan — the lambda argument is never "
        "being used; it is the temperature, not a cosmetic scale factor")
    assert not np.allclose(small, hot, rtol=0.0, atol=1e-6) and \
        not np.allclose(small, cold, rtol=0.0, atol=1e-6), (
        "the default lambda gave the same plan as lambda = 1e6 or lambda = 0.01 — at lambda = 30 "
        "against costs 0..11 the refit should sit strictly between the arithmetic mean and the "
        "single best sample")


def test_closed_loop_swings_up():
    met = _cached("mpc_default", lambda: lesson.metrics(lesson.run_cpp("mpc")))
    assert met["swung_up"] == 1.0, (
        f"the planner did not swing the pole up: upright fraction "
        f"{met['upright_fraction']:.2f}, final |theta| {met['final_abs_theta']:.3f} rad")
    assert met["upright_fraction"] >= 0.8, (
        f"the pole was upright for only {met['upright_fraction']:.2f} of the final second")
    assert met["final_abs_theta"] <= 0.35, (
        f"the run ended {met['final_abs_theta']:.3f} rad from upright")
    assert met["max_abs_cart"] < 2.0, (
        f"the cart reached {met['max_abs_cart']:.2f} m and hit the rail limit")


def test_closed_loop_generalises_to_an_unseen_seed():
    met = _cached("mpc_101", lambda: lesson.metrics(lesson.run_cpp("mpc", "--seed", "101")))
    assert met["swung_up"] == 1.0 and met["upright_fraction"] >= 0.8, (
        f"the planner swings up on the default seed but not on seed 101 (upright fraction "
        f"{met['upright_fraction']:.2f}, final |theta| {met['final_abs_theta']:.3f} rad) — a "
        "planner that only works for one random stream is not a planner")


def test_python_rollout_matches_cpp():
    rng = random.Random(555001)
    plan = [max(-1.0, min(1.0, rng.gauss(0.0, 0.55))) for _ in range(48)]
    model, data, scratch = lesson.fresh()
    mine = lesson.python_rollout_cost(model, data, scratch, plan)
    # `is float`, not isinstance: np.float64 subclasses float, so isinstance cannot tell a
    # plain Python float from a numpy scalar, and the docstring asks for the former.
    assert type(mine) is float, (
        f"python_rollout_cost returned {type(mine).__name__}; return a plain Python float — "
        "accumulate into a float and return it, or wrap the result in float(...). A numpy "
        "scalar sneaks past isinstance(x, float) and then carries dtype baggage downstream")
    assert abs(float(data.qpos[1]) - math.pi) < 1e-12, (
        "python_rollout_cost moved the live `data` — copy into `scratch` first and step that")
    ref = _reference_cost(plan)
    rel = abs(mine - ref) / abs(ref)
    assert rel < 1e-9, (
        f"the Python rollout scored {mine!r} against an independent reference of {ref!r} "
        f"(relative gap {rel:.3e}) — set ctrl, then step, then score the resulting state")


def test_rollout_budget_arithmetic():
    b = lesson.rollout_budget(200, 40, 100.0, 1_000_000.0)
    expected = {"steps_per_tick", "steps_per_second_required", "tick_seconds",
                "plan_seconds_per_tick", "realtime", "max_samples"}
    assert set(b) == expected, f"keys were {sorted(b)}, expected {sorted(expected)}"
    assert b["steps_per_tick"] == 8000, f"steps_per_tick was {b['steps_per_tick']}, expected 8000"
    assert b["steps_per_second_required"] == 800_000.0, (
        f"steps_per_second_required was {b['steps_per_second_required']}, expected 800000")
    assert b["tick_seconds"] == 0.01, f"tick_seconds was {b['tick_seconds']}, expected 0.01"
    assert abs(b["plan_seconds_per_tick"] - 0.008) < 1e-15, (
        f"plan_seconds_per_tick was {b['plan_seconds_per_tick']}, expected 8000/1e6 = 0.008")
    assert b["realtime"] is True, "8 ms of planning fits a 10 ms tick, so realtime is True"
    assert b["max_samples"] == 250, f"max_samples was {b['max_samples']}, expected 250"

    # 8000.0 == 8000 and 250.0 == 250, so every equality above is blind to the type. Both of
    # these count things that cannot come in fractions.
    assert type(b["steps_per_tick"]) is int, (
        f"steps_per_tick came back as {type(b['steps_per_tick']).__name__} "
        f"({b['steps_per_tick']!r}) — it counts mj_step calls, so it is an int; a float here "
        "compares equal to 8000 and still says you can run half a step")
    assert type(b["max_samples"]) is int, (
        f"max_samples came back as {type(b['max_samples']).__name__} ({b['max_samples']!r}) — "
        "250.0 is not a whole number of samples; wrap the floor division in int()")

    slow = lesson.rollout_budget(200, 40, 100.0, 300_000.0)
    assert slow["realtime"] is False, "26.7 ms of planning cannot fit a 10 ms tick"
    assert slow["max_samples"] == 75, f"max_samples was {slow['max_samples']}, expected 75"

    # max_samples is inversely proportional to the horizon: double it, halve the samples.
    wide = lesson.rollout_budget(200, 80, 100.0, 1_000_000.0)
    assert wide["max_samples"] == 125, (
        f"at horizon 80 the affordable sample count is {wide['max_samples']}, expected 125 — "
        "it scales as steps_per_second / (control_hz * horizon)")
    assert wide["steps_per_tick"] == 16000, "200 samples of an 80-step horizon is 16000 steps"

    fast = lesson.rollout_budget(50, 10, 200.0, 2_000_000.0)
    assert fast["steps_per_tick"] == 500 and fast["tick_seconds"] == 0.005, (
        "check steps_per_tick = samples*horizon and tick_seconds = 1/control_hz")
    assert fast["realtime"] is True and fast["max_samples"] == 1000, (
        f"expected 1000 affordable samples, got {fast['max_samples']}")

    # Every case above divides out to a whole number, so none of them can tell flooring from
    # rounding. These two can: 1234567/(100*40) = 308.64 and 999999/(50*30) = 666.666.
    odd = lesson.rollout_budget(200, 40, 100.0, 1_234_567.0)
    assert odd["max_samples"] == 308, (
        f"max_samples was {odd['max_samples']}, expected floor(1234567/(100*40)) = 308 — "
        "308.64 affordable samples is 308 samples, not 309; round DOWN, because a partial "
        "sample still costs a whole rollout")
    odder = lesson.rollout_budget(10, 30, 50.0, 999_999.0)
    assert odder["max_samples"] == 666, (
        f"max_samples was {odder['max_samples']}, expected floor(999999/(50*30)) = 666")

    # Exactly filling the tick counts as fitting: the spec says <=, not <.
    exact = lesson.rollout_budget(100, 40, 100.0, 400_000.0)
    assert exact["plan_seconds_per_tick"] == exact["tick_seconds"] == 0.01, (
        f"4000 steps at 400k steps/s is exactly one 10 ms tick, but plan_seconds_per_tick was "
        f"{exact['plan_seconds_per_tick']} against a tick of {exact['tick_seconds']}")
    assert exact["realtime"] is True, (
        "a planner that exactly fills its control tick still fits — compare "
        "plan_seconds_per_tick <= tick_seconds, not <")

    # A budget you cannot compute is not a budget. Dividing the work by a throughput of zero
    # raises all by itself, so the straightforward implementation passes this without trying;
    # only a try/except that invents a default gets past it.
    try:
        bad = lesson.rollout_budget(200, 40, 100.0, 0.0)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError(
            f"rollout_budget(200, 40, 100.0, 0.0) returned {bad} instead of letting the "
            "ZeroDivisionError through — you wrapped the arithmetic in try/except and returned "
            "a default. A zeroed dict reads like a real answer (realtime False, max_samples 0) "
            "and hides the bad throughput that produced it")


def test_cpp_outruns_python_on_identical_work():
    cpp = _cached("bench600", lambda: lesson.metrics(
        lesson.run_cpp("bench", "--samples", "600", "--horizon", "40")))
    py = lesson.python_throughput(rollouts=600, horizon=40)
    assert cpp["steps"] == py["steps"], (
        f"the two benchmarks did not do the same work: C++ {cpp['steps']}, Python {py['steps']}")
    ratio = cpp["steps_per_second"] / py["steps_per_second"]
    assert ratio > 1.5, (
        f"C++ managed {cpp['steps_per_second']:,.0f} steps/s against Python's "
        f"{py['steps_per_second']:,.0f} (ratio {ratio:.2f}) — on identical mj_step calls the "
        "compiled loop should be clearly ahead; if it is not, the Python loop is probably not "
        "doing the same work")


RUBRIC = [
    (test_cpp_builds, 3,
     "run `make` in the lesson directory and read the first compiler error"),
    (test_cpp_selftest_passes, 5,
     "run `make test` — it names the unfinished exercise and what it expects"),
    (test_sampling_statistics, 5,
     "draw a fresh standard normal for every (sample, step), scale by sigma, add mean[t], "
     "then clamp"),
    (test_rollout_matches_hidden_reference, 6,
     "set ctrl, then mj_step, then score the state you just arrived at — in that order"),
    (test_rollout_leaves_the_live_state_alone, 3,
     "load_state(m, scratch, start) at the top of rollout_cost, then step `scratch` only"),
    (test_rollout_invariants, 3,
     "accumulate stage_cost over the whole horizon rather than returning the last one"),
    (test_refit_invariants, 5,
     "subtract the minimum cost before exp, and divide by the sum of the weights"),
    (test_closed_loop_swings_up, 6,
     "all three pieces must be right at once; `make test` isolates which one is not"),
    (test_closed_loop_generalises_to_an_unseen_seed, 4,
     "if it works on one seed only, the refit is probably collapsing onto a single sample"),
    (test_python_rollout_matches_cpp, 5,
     "mj_copyData(scratch, model, data) first; the C++ version is the specification"),
    (test_rollout_budget_arithmetic, 5,
     "plan_seconds_per_tick = samples*horizon / steps_per_second; max_samples floors "
     "steps_per_second / (control_hz * horizon)"),
    (test_cpp_outruns_python_on_identical_work, 3,
     "both loops must run the same number of mj_step calls for the ratio to mean anything"),
]
