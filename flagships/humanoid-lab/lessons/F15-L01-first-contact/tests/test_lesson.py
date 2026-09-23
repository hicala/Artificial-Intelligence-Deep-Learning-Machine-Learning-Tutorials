"""Autograder rubric for F15-L01. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

The cases here deliberately differ from the notebook's public `_check_*` functions: other
durations, other keyframes, and structural probes (does the physics clock actually advance?
does the answer respond to posture?) that a plausible-looking wrong implementation fails.
"""
import importlib.util, math, os
from pathlib import Path

import mujoco

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

EPS = 1e-9


def _expected_steps(model, seconds: float) -> int:
    """Whole steps needed to cover `seconds` without undershooting."""
    return max(math.ceil(seconds / model.opt.timestep - EPS), 0)


# --- exercise 1: step_for -----------------------------------------------------------------

def test_step_for_exact_multiple():
    """A duration that is a whole number of timesteps must cost exactly that many steps."""
    model, data = lesson.load_humanoid()
    for seconds in (0.25, 0.5, 1.0):
        fresh_model, fresh_data = lesson.load_humanoid()
        got = lesson.step_for(fresh_model, fresh_data, seconds)
        want = _expected_steps(fresh_model, seconds)
        assert got == want, f"step_for(..., {seconds}) returned {got}, expected {want}"
    assert lesson.step_for(model, data, 0.1) == _expected_steps(model, 0.1)


def test_step_for_rounds_up():
    """Fractional timesteps round up: the contract is to never undershoot."""
    model, _ = lesson.load_humanoid()
    dt = model.opt.timestep
    for seconds, want in (
        (dt * 1.4, 2),          # the docstring's worked case
        (dt * 0.4, 1),          # less than one step still costs a step
        (dt * 2.000001, 3),     # a hair over two
        (dt * 3.0, 3),          # exactly three, and float drift must not add a fourth
    ):
        _, data = lesson.load_humanoid()
        got = lesson.step_for(model, data, seconds)
        assert got == want, (
            f"step_for(..., {seconds!r}) returned {got}, expected {want} — truncation "
            "undershoots and an over-eager epsilon overshoots"
        )


def test_step_for_advances_the_physics_clock():
    """Counting is not simulating: data.time and data.qpos must both actually move."""
    model, data = lesson.load_humanoid()
    before_qpos = data.qpos.copy()
    steps = lesson.step_for(model, data, 0.3)
    assert steps > 0, "step_for(..., 0.3) took no steps at all"
    assert abs(data.time - steps * model.opt.timestep) < 1e-6, (
        f"data.time is {data.time} after {steps} steps of {model.opt.timestep}s — the clock "
        "did not advance by one timestep per step, so mj_step was not called each time"
    )
    assert not (data.qpos == before_qpos).all(), (
        "qpos is unchanged after stepping — the state never moved, so no physics ran"
    )


def test_step_for_zero_negative_and_relative():
    """Zero and negative cost nothing; repeated calls each advance by the same amount."""
    model, data = lesson.load_humanoid()
    assert lesson.step_for(model, data, 0.0) == 0, "0.0 seconds must take no steps"
    assert lesson.step_for(model, data, -1.0) == 0, "a negative duration must take no steps"
    assert data.time == 0.0, "the clock moved while taking zero steps"
    runs = [lesson.step_for(model, data, 0.2) for _ in range(3)]
    assert len(set(runs)) == 1, (
        f"three consecutive 0.2s calls gave {runs} — the target must be relative to the "
        "clock as found, not compared against `seconds` absolutely"
    )
    assert abs(data.time - sum(runs) * model.opt.timestep) < 1e-6, (
        "total elapsed simulated time does not match the total number of steps taken"
    )


# --- exercise 2: com_height ---------------------------------------------------------------

def test_com_height_matches_mujoco_reference():
    """The mass-weighted mean must equal MuJoCo's own whole-body subtree COM."""
    model, data = lesson.load_humanoid()
    mujoco.mj_forward(model, data)
    got = lesson.com_height(model, data)
    assert isinstance(got, float), f"expected a float, got {type(got).__name__}"
    reference = float(data.subtree_com[0][2])
    mass = model.body_mass
    assert abs(got - reference) < 1e-9, (
        f"got {got:.6f}, MuJoCo's whole-body COM is {reference:.6f}. Each near miss names a "
        f"different mistake, and all three are computed from this very pose: "
        f"{float(data.qpos[2]):.6f} is the root height in qpos[2]; "
        f"{float((mass * data.xpos[:, 2]).sum() / mass.sum()):.6f} weights data.xpos, the "
        f"frame origins, where data.xipos belongs; "
        f"{float(data.xipos[:, 2].mean()):.6f} is an unweighted mean of xipos, with "
        "model.body_mass left out"
    )


def test_com_height_tracks_every_keyframe():
    """Across all shipped poses the answer must follow MuJoCo, and differ from the root."""
    model, _ = lesson.load_humanoid()
    assert model.nkey > 0, "this model should ship named keyframes"
    gaps = []
    for k in range(model.nkey):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, k)
        mujoco.mj_forward(model, data)
        got = lesson.com_height(model, data)
        reference = float(data.subtree_com[0][2])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k)
        assert abs(got - reference) < 1e-9, (
            f"keyframe {name!r}: got {got:.9f}, expected {reference:.9f}"
        )
        gaps.append(abs(got - float(data.qpos[2])))
    assert max(gaps) > 0.01, (
        "the COM height never differs from the root height by even a centimetre in any pose "
        "— you are reading qpos[2] rather than the mass-weighted body positions"
    )


def test_com_height_reflects_written_state():
    """Lifting the whole model in qpos must lift the COM by the same amount."""
    model, data = lesson.load_humanoid()
    mujoco.mj_forward(model, data)
    base = lesson.com_height(model, data)
    data.qpos[2] += 0.5
    mujoco.mj_forward(model, data)
    lifted = lesson.com_height(model, data)
    assert abs((lifted - base) - 0.5) < 1e-6, (
        f"COM went from {base:.6f} to {lifted:.6f} after raising the root by 0.5 m; a rigid "
        "translation must move the centre of mass by exactly the same 0.5 m"
    )


# --- exercise 3: real_time_factor ---------------------------------------------------------

def test_real_time_factor_shape_and_internal_consistency():
    """The four keys must be present, typed, and consistent with each other.

    The duration is deliberately NOT a whole number of timesteps. On an exact multiple
    `steps * timestep` equals the `seconds` argument, so returning the argument — which the
    docstring expressly forbids — is indistinguishable from the correct answer.
    """
    model, data = lesson.load_humanoid()
    asked = 0.207                      # 41.4 timesteps at this model's 0.005 s, so 42 steps
    report = lesson.real_time_factor(model, data, asked)
    assert isinstance(report, dict), f"expected a dict, got {type(report).__name__}"
    assert set(report) == {"steps", "sim_seconds", "wall_seconds", "real_time_factor"}, (
        f"keys were {sorted(report)}"
    )
    assert report["steps"] == _expected_steps(model, asked), (
        f"steps was {report['steps']}, expected {_expected_steps(model, asked)}"
    )
    assert not isinstance(report["steps"], float), (
        f"steps came back as {type(report['steps']).__name__} ({report['steps']!r}) — it "
        "counts mj_step calls, so it must be a whole number. math.ceil returns an int; "
        "numpy's np.ceil returns a float that compares equal here and then breaks range() "
        "and every index built from it downstream"
    )
    assert abs(report["sim_seconds"] - report["steps"] * model.opt.timestep) < 1e-9, (
        f"sim_seconds was {report['sim_seconds']!r} for {asked} s asked; it must be "
        f"steps * timestep = {report['steps'] * model.opt.timestep!r}, derived from the steps "
        "actually taken rather than echoing the duration requested"
    )
    assert report["wall_seconds"] > 0.0, f"wall_seconds was {report['wall_seconds']}"
    ratio = report["sim_seconds"] / report["wall_seconds"]
    assert abs(report["real_time_factor"] - ratio) < 1e-6, (
        f"real_time_factor {report['real_time_factor']:.6f} != sim/wall {ratio:.6f}"
    )


def test_real_time_factor_is_measured_and_right_way_up():
    """It must exceed 1 on any CPU this lesson targets, and scale with the work requested."""
    model, data = lesson.load_humanoid()
    short = lesson.real_time_factor(model, data, 0.1)
    longer = lesson.real_time_factor(model, data, 0.4)
    assert short["real_time_factor"] > 2.0, (
        f"real_time_factor was {short['real_time_factor']:.4f}; this model steps far faster "
        "than real time on a CPU, so a value at or below 1 means the ratio is inverted"
    )
    assert longer["steps"] > short["steps"], (
        f"0.4 s of simulation took {longer['steps']} steps against {short['steps']} for 0.1 s "
        "— the measurement is not responding to the duration requested"
    )
    assert longer["wall_seconds"] > 0.0 and short["wall_seconds"] > 0.0
    assert data.time > 0.0, (
        "the data handed in was never stepped — reset it, then step it, do not simulate a "
        "throwaway copy"
    )
    # Two calls have now run on one `data`. If the reset happened, the clock reads exactly
    # what the second call's own steps bought; if it did not, it still carries the first.
    assert abs(data.time - longer["steps"] * model.opt.timestep) < 1e-9, (
        f"after two measurements on the same data, data.time is {data.time} rather than the "
        f"{longer['steps'] * model.opt.timestep} that the last call's {longer['steps']} steps "
        "bought — the clock still carries the previous run, so mujoco.mj_resetData never "
        "ran. Every measurement after the first then times an already-fallen, contact-heavy "
        "pose, which is a different benchmark from the one you reported"
    )


RUBRIC = [
    (test_step_for_exact_multiple, 4,
     "steps = seconds / model.opt.timestep, and never a hard-coded timestep"),
    (test_step_for_rounds_up, 4,
     "int() and // truncate; round the step count up so you never undershoot"),
    (test_step_for_advances_the_physics_clock, 3,
     "the count must come from real mujoco.mj_step calls, not from arithmetic"),
    (test_step_for_zero_negative_and_relative, 2,
     "a while loop that tests before stepping, against a target relative to data.time"),
    (test_com_height_matches_mujoco_reference, 4,
     "sum(model.body_mass * data.xipos[:, 2]) / model.body_mass.sum()"),
    (test_com_height_tracks_every_keyframe, 3,
     "read the live pose from data.xipos; qpos[2] is only the root body"),
    (test_com_height_reflects_written_state, 2,
     "a mass-weighted mean of world-frame positions translates with the whole body"),
    (test_real_time_factor_shape_and_internal_consistency, 2,
     "return exactly the four documented keys, sim_seconds = steps * timestep, steps a whole "
     "number"),
    (test_real_time_factor_is_measured_and_right_way_up, 2,
     "mj_resetData first, then simulated seconds over wall seconds, timed with perf_counter"),
]
