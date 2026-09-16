# %% [markdown]
# # F15-L01 · First contact with a humanoid — reference solution
#
# The student notebook with every stub filled in. Runs top to bottom in a fresh interpreter
# and prints real measured output. Self-check answers are in the comment block at the foot.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import hashlib
import math
import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np

print("mujoco", mujoco.__version__, "· numpy", np.__version__)

MODEL_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/humanoid/humanoid.xml"
)
MODEL_FILENAME = "humanoid.xml"

TIME_EPS = 1e-9


def humanoid_xml_path() -> Path:
    """Return the path to the cached humanoid XML, downloading it only if it is missing."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:  # a notebook has no __file__
        here = Path.cwd()
    candidates = [
        here / "assets" / MODEL_FILENAME,
        here.parent / "assets" / MODEL_FILENAME,
        Path.cwd() / "assets" / MODEL_FILENAME,
        Path.cwd().parent / "assets" / MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    target = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"cached model absent; fetching once from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, target)  # noqa: S310 - pinned https URL above
    return target


def load_humanoid():
    """Compile the humanoid XML and hand back a fresh (model, data) pair."""
    model = mujoco.MjModel.from_xml_path(str(humanoid_xml_path()))
    return model, mujoco.MjData(model)


MODEL, DATA = load_humanoid()
print("compiled:", humanoid_xml_path().name)

# %% [markdown]
# ## 1. Two objects, and the difference is the whole lesson

# %%
print("mjModel — constant description")
for _name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nkey"):
    print(f"  {_name:7s} = {getattr(MODEL, _name)}")
print(f"  timestep = {MODEL.opt.timestep} s")
print(f"  gravity  = {MODEL.opt.gravity[2]:.2f} m/s^2 (z)")
print(f"  mass     = {MODEL.body_mass.sum():.3f} kg total")

print("\nmjData — live state")
print(f"  time = {DATA.time} s")
print(f"  qpos holds {DATA.qpos.shape[0]} numbers, qvel holds {DATA.qvel.shape[0]}")

# %%
_left, _right = mujoco.MjData(MODEL), mujoco.MjData(MODEL)
for _ in range(10):
    mujoco.mj_step(MODEL, _left)
print(f"left.time={_left.time:.3f}s  right.time={_right.time:.3f}s  "
      f"(the model they share was never modified)")

# %% [markdown]
# ## 2. Why `nq` is not `nv`

# %%
def joint_table(model, limit: int = 6) -> list:
    """Rows of (name, joint type, qpos address, qvel address) for the first `limit` joints."""
    rows = []
    for j in range(min(model.njnt, limit)):
        rows.append((
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
            mujoco.mjtJoint(model.jnt_type[j]).name,
            int(model.jnt_qposadr[j]),
            int(model.jnt_dofadr[j]),
        ))
    return rows


print(f"{'joint':18s} {'type':12s} {'qpos@':>6s} {'qvel@':>6s}")
for _jname, _jtype, _qadr, _vadr in joint_table(MODEL):
    print(f"{_jname:18s} {_jtype:12s} {_qadr:6d} {_vadr:6d}")
print(f"\nnq - nv = {MODEL.nq - MODEL.nv}  (one quaternion's worth of extra position slots)")
print(f"free joint qpos block: {np.array(MODEL.qpos0[:7])}")
print("                       ^ x y z, then quaternion w x y z")

# %% [markdown]
# ## 3. One step of physics

# %%
_probe = mujoco.MjData(MODEL)
_t_before, _z_before = _probe.time, float(_probe.qpos[2])
mujoco.mj_step(MODEL, _probe)
print(f"time   {_t_before} -> {_probe.time}  (+{_probe.time - _t_before} s)")
print(f"root z {_z_before:.6f} -> {float(_probe.qpos[2]):.6f}  (it has begun to fall)")

# %% [markdown]
# ## 4. Exercise 1 — `step_for(model, data, seconds)`
#
# The loop tests before it steps, so `seconds <= 0` costs nothing. The target is relative to
# the clock as found, so repeated calls each advance by the same amount. Subtracting
# `TIME_EPS` stops accumulated float drift (which reaches a few parts in 1e14 over hundreds
# of steps) from buying one extra step.

# %%
def step_for(model, data, seconds: float) -> int:
    """Step `data` until at least `seconds` of simulated time has passed; return step count.

    Measure elapsed time from `data.time` as it is when you are called, so the function also
    works on a `data` that has already been stepped. Use `TIME_EPS` as the slack in your
    comparison, so float drift in the accumulated clock cannot buy an extra step.

    Example (this model's timestep is 0.005 s, so 0.007 s is 1.4 timesteps, rounded up to 2):
        >>> model, data = load_humanoid()
        >>> step_for(model, data, 0.007)
        2
        >>> round(data.time, 3)
        0.01

    Returns:
        The number of mujoco.mj_step calls made. Zero is a valid answer for seconds <= 0.
    """
    start = data.time
    steps = 0
    while data.time - start < seconds - TIME_EPS:
        mujoco.mj_step(model, data)
        steps += 1
    return steps


def _check_step_for() -> None:
    model, data = load_humanoid()
    n = step_for(model, data, 1.0)
    expected = math.ceil(1.0 / model.opt.timestep - TIME_EPS)
    assert n == expected, (
        f"step_for(1.0) returned {n}, expected {expected} — divide the request by "
        "model.opt.timestep; do not assume any particular timestep value."
    )
    assert abs(data.time - n * model.opt.timestep) < 1e-6, (
        f"data.time is {data.time} after {n} steps — you counted steps without calling "
        "mujoco.mj_step, so no physics actually happened."
    )

    model2, data2 = load_humanoid()
    partial = step_for(model2, data2, 0.007)
    assert partial == 2, (
        f"step_for(0.007) returned {partial}, expected 2 — int() and // truncate, and the "
        "contract is to never undershoot, so the step count rounds up."
    )

    model3, data3 = load_humanoid()
    assert step_for(model3, data3, 0.0) == 0, (
        "step_for(model, data, 0.0) must take no steps — your loop runs at least once, so "
        "make it a while loop that tests before it steps."
    )

    model4, data4 = load_humanoid()
    first = step_for(model4, data4, 0.5)
    second = step_for(model4, data4, 0.5)
    assert first == second, (
        f"stepping 0.5 s twice gave {first} then {second} — you compared data.time against "
        "`seconds` absolutely instead of against the time when the call started."
    )
    print(f"exercise 1 looks right: {n} steps buy {data.time:.3f} s of simulated time")


# %% [markdown]
# ## 5. Exercise 2 — `com_height(model, data)`
#
# `body_mass` and `xipos[:, 2]` are both length `nbody`, so numpy does the whole weighted sum
# in one expression. The world body sits at index 0 with zero mass and drops out on its own.

# %%
def com_height(model, data) -> float:
    """Return the z coordinate of the whole-body centre of mass, in metres.

    Mass-weighted mean over all bodies: sum(mass_i * z_i) / sum(mass_i), where z_i is
    `data.xipos[i][2]`. The world body carries zero mass, so including it changes nothing.

    `data.xipos` is only valid after MuJoCo has run kinematics — after an `mj_step`, an
    `mj_forward`, or a keyframe reset followed by `mj_forward`. This function does not
    refresh it for you; the caller does.

    Example (the default pose of this model stands taller at the root than at the COM):
        >>> model, data = load_humanoid()
        >>> mujoco.mj_forward(model, data)
        >>> com_height(model, data) < float(data.qpos[2])
        True

    Returns:
        A single float, not an array.
    """
    mass = model.body_mass
    return float((mass * data.xipos[:, 2]).sum() / mass.sum())


def _check_com_height() -> None:
    model, data = load_humanoid()
    mujoco.mj_forward(model, data)
    got = com_height(model, data)
    assert isinstance(got, float), (
        f"com_height returned {type(got).__name__} — index the z component and cast with "
        "float(); do not return the whole 3-vector."
    )
    reference = float(data.subtree_com[0][2])
    mass = model.body_mass
    assert abs(got - reference) < 1e-9, (
        f"com_height gave {got:.6f}, MuJoCo's own whole-body COM is {reference:.6f}. Each "
        "near miss names a different mistake, and all three are measured from this pose: "
        f"{float(data.qpos[2]):.6f} is the root height in qpos[2]; "
        f"{float((mass * data.xpos[:, 2]).sum() / mass.sum()):.6f} weights data.xpos, the "
        "frame origins, where data.xipos belongs; "
        f"{float(data.xipos[:, 2].mean()):.6f} is an unweighted mean of xipos, with "
        "model.body_mass left out."
    )
    squat = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, squat, 0)
    mujoco.mj_forward(model, squat)
    crouched = com_height(model, squat)
    assert crouched < got - 0.1, (
        f"the crouched keyframe gave {crouched:.6f} against {got:.6f} standing — a constant, "
        "or anything computed from the model alone, cannot respond to a pose change; the "
        "pose lives in data.xipos."
    )
    print(f"exercise 2 looks right: standing COM {got:.3f} m, crouched {crouched:.3f} m")


# %% [markdown]
# ## 6. Exercise 3 — `real_time_factor(model, data, seconds)`
#
# `mj_resetData` puts the clock and the pose back to `qpos0`, so the measurement is
# repeatable. `perf_counter` brackets only the stepping loop: no model compilation, no reset,
# no printing inside the timed region.

# %%
def real_time_factor(model, data, seconds: float) -> dict:
    """Measure how much faster than reality this machine simulates `seconds` of motion.

    In order: reset `data`, read `time.perf_counter()`, call your `step_for`, read the
    counter again.

    Example:
        >>> model, data = load_humanoid()
        >>> report = real_time_factor(model, data, 0.05)
        >>> sorted(report)
        ['real_time_factor', 'sim_seconds', 'steps', 'wall_seconds']
        >>> report["sim_seconds"] == report["steps"] * model.opt.timestep
        True

    Returns:
        dict with exactly these four keys:
          "steps"            int, from step_for
          "sim_seconds"      float, steps * model.opt.timestep
          "wall_seconds"     float, measured elapsed real time
          "real_time_factor" float, sim_seconds / wall_seconds
    """
    mujoco.mj_resetData(model, data)
    t0 = time.perf_counter()
    steps = step_for(model, data, seconds)
    wall_seconds = time.perf_counter() - t0
    sim_seconds = steps * model.opt.timestep
    return {
        "steps": steps,
        "sim_seconds": float(sim_seconds),
        "wall_seconds": float(wall_seconds),
        "real_time_factor": float(sim_seconds / wall_seconds),
    }


def _check_real_time_factor() -> None:
    model, data = load_humanoid()
    # 0.507 s is deliberately not a whole number of timesteps: on an exact multiple,
    # steps * timestep equals the seconds you were handed, and returning the argument
    # instead of the measured step count would slip through unnoticed.
    report = real_time_factor(model, data, 0.507)
    expected_keys = {"steps", "sim_seconds", "wall_seconds", "real_time_factor"}
    assert set(report) == expected_keys, (
        f"keys were {sorted(report)}, expected {sorted(expected_keys)} — return a dict with "
        "exactly those four names, spelled as in the docstring."
    )
    assert report["wall_seconds"] > 0.0, (
        "wall_seconds is not positive — you read perf_counter twice before stepping, or "
        "subtracted the two readings the wrong way round."
    )
    assert abs(report["sim_seconds"] - report["steps"] * model.opt.timestep) < 1e-9, (
        "sim_seconds does not equal steps * timestep — derive it from the step count you "
        "actually took, not from the `seconds` argument."
    )
    ratio = report["sim_seconds"] / report["wall_seconds"]
    assert abs(report["real_time_factor"] - ratio) < 1e-6, (
        f"real_time_factor is {report['real_time_factor']:.4f} but sim/wall is {ratio:.4f} — "
        "the ratio is simulated over wall; inverting it reports a fraction near zero."
    )
    assert report["real_time_factor"] > 1.0, (
        f"real_time_factor came out at {report['real_time_factor']:.4f} — a value that small "
        "means the ratio is upside down, or that you timed a reload instead of the stepping."
    )
    assert not isinstance(report["steps"], float), (
        f"steps came back as {type(report['steps']).__name__} — it counts mj_step calls, so "
        "it must be a whole number. math.ceil gives an int; numpy's np.ceil gives a float "
        "that still compares equal here and then breaks range() downstream."
    )
    # A second measurement on the same `data`: if the reset ran, the clock reads exactly what
    # this call's own steps bought, and nothing of the previous call survives.
    again = real_time_factor(model, data, 0.207)
    assert abs(data.time - again["steps"] * model.opt.timestep) < 1e-9, (
        f"a second call left data.time at {data.time}, not the "
        f"{again['steps'] * model.opt.timestep} its own {again['steps']} steps bought — the "
        "clock still carries the first run, so mujoco.mj_resetData never ran and every "
        "measurement after the first times an already-fallen pose."
    )
    print(f"exercise 3 looks right: {report['sim_seconds']:.3f} s simulated in "
          f"{report['wall_seconds']:.4f} s wall, RTF {report['real_time_factor']:.1f}x")


# %% [markdown]
# ## 7. The payoff: a fall, measured

# %%
def fall_profile(samples: int = 10, interval: float = 0.2) -> list:
    """Return [(simulated time, COM height)] sampled every `interval` simulated seconds."""
    model, data = load_humanoid()
    mujoco.mj_forward(model, data)
    trace = [(data.time, com_height(model, data))]
    for _ in range(samples):
        step_for(model, data, interval)
        trace.append((data.time, com_height(model, data)))
    return trace


def _check_fall_profile() -> None:
    trace = fall_profile()
    start, end = trace[0][1], trace[-1][1]
    print(f"{'sim t (s)':>10s} {'COM z (m)':>10s}  profile")
    for t, z in trace:
        print(f"{t:10.2f} {z:10.4f}  {'#' * max(int(round(z * 40)), 0)}")
    assert end < start, (
        f"COM went from {start:.3f} to {end:.3f} — an uncontrolled humanoid under gravity "
        "must end lower than it started; check that step_for really calls mj_step."
    )
    print(f"\nfell {start - end:.3f} m of centre-of-mass height in {trace[-1][0]:.1f} s of "
          "simulated time, with zero control applied")


# %% [markdown]
# ## 8. Extra readout: the same posture sensor across every keyframe
#
# The shipped model carries several named poses. Reading the COM for each is a one-line check
# that the sensor responds to posture and not to the model alone.

# %%
def keyframe_report() -> list:
    """Return [(keyframe name, COM height, root height)] for every keyframe in the model."""
    model, _ = load_humanoid()
    rows = []
    for k in range(model.nkey):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, k)
        mujoco.mj_forward(model, data)
        rows.append((
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k),
            com_height(model, data),
            float(data.qpos[2]),
        ))
    return rows


def _report_keyframes() -> None:
    print(f"{'keyframe':20s} {'COM z (m)':>10s} {'root z (m)':>11s} {'gap (m)':>9s}")
    for name, com_z, root_z in keyframe_report():
        print(f"{name:20s} {com_z:10.4f} {root_z:11.4f} {root_z - com_z:9.4f}")
    print("\nthe gap is never zero: the root body is not the centre of mass")


# %% [markdown]
# ## 9. Two mistakes, demonstrated
#
# The notebook lists these in its common-mistakes section; here they are as measured output.
# Writing `qpos` marks nothing dirty, so every derived array still describes the old pose
# until `mj_forward` runs — which is exactly what self-check question 4 asks about.

# %%
_stale = mujoco.MjData(MODEL)
mujoco.mj_forward(MODEL, _stale)
_frame_gap = np.abs(_stale.xpos[:, 2] - _stale.xipos[:, 2])
print(f"xpos vs xipos in z: worst body {_frame_gap.max():.4f} m, mean "
      f"{_frame_gap[_frame_gap > 0].mean():.4f} m over the "
      f"{int((_frame_gap > 0).sum())} bodies where the two differ at all")
_torso_before = float(_stale.xipos[1][2])
_stale.qpos[2] += 1.0                    # lift the robot a metre, then read derived state at once
print(f"wrote qpos[2] += 1.0 -> torso xipos z still reads {float(_stale.xipos[1][2]):.4f} "
      f"(it was {_torso_before:.4f})")
mujoco.mj_forward(MODEL, _stale)         # the one line that makes derived state true again
print(f"after mujoco.mj_forward it reads            {float(_stale.xipos[1][2]):.4f}  "
      f"(+{float(_stale.xipos[1][2]) - _torso_before:.4f} m, the lift you asked for)")
print(f"\nperf_counter monotonic={time.get_clock_info('perf_counter').monotonic}, "
      f"time() monotonic={time.get_clock_info('time').monotonic}"
      "  <- only one of these cannot go backwards mid-measurement")

# %% [markdown]
# ## 10. Self-check answers
#
# The four letters, marked by the same digest checker the student notebook ships. The
# reasoning is in the comment block at the foot of this file.

# %%
SELF_CHECK = {1: "c", 2: "b", 3: "c", 4: "c"}

_ANSWER_DIGESTS = {1: "d260b0fbba30524f", 2: "0c940df6b5fea42b",
                   3: "9230a39fdad2bf8e", 4: "f22b9958b9c74bec"}
_ANSWER_SECTIONS = {
    1: "section 1 — which of the two objects changed while the other one stayed put",
    2: "section 2 — the joint table you printed, and the free joint's 7-number qpos block",
    3: "section 6 — what your own real_time_factor measured, and what it says nothing about",
    # Section 8 in the student notebook; the reference carries an extra keyframe readout
    # ahead of it, so the same demonstration is section 9 in this file.
    4: "section 9 — the stale-xipos demonstration you just ran",
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the four multiple-choice answers in SELF_CHECK, naming where to look again."""
    answers = SELF_CHECK if answers is None else answers
    wrong = []
    for q, digest in sorted(_ANSWER_DIGESTS.items()):
        got = str(answers.get(q, "?")).strip().lower()
        if hashlib.sha256(f"F15-L01-q{q}-{got}".encode()).hexdigest()[:16] != digest:
            wrong.append(q)
    for q in sorted(_ANSWER_DIGESTS):
        note = f"  -> re-read {_ANSWER_SECTIONS[q]}" if q in wrong else ""
        print(f"  q{q}: {'wrong' if q in wrong else 'right'}{note}")
    assert not wrong, (
        f"questions {wrong} are still wrong. Each one names the section that answers it "
        "above — go back to the cell you ran there rather than guessing another letter."
    )
    print("self-check: all four right")


# %%
if __name__ == "__main__":
    _check_step_for()
    _check_com_height()
    _check_real_time_factor()
    _check_fall_profile()
    print()
    _report_keyframes()
    print()
    _check_self_check()

# %% [markdown]
# ## Self-check answers
#
# **1 — (c) mass in `mjModel`, velocity in `mjData`.** Mass is part of the compiled
# description and does not change while you simulate, so it is `model.body_mass`. A joint
# velocity is state that changes every step, so it is `data.qvel` at that joint's `dofadr`.
# The split is the organising principle of the whole API: constants in the model, everything
# that moves in the data.
#
# **2 — (b) the free joint stores a 4-number quaternion but only a 3-number angular
# velocity.** Every hinge contributes one slot to each vector, so hinges cannot create a
# mismatch. The single free joint contributes 7 to `qpos` (x y z, then w x y z) and 6 to
# `qvel` (linear, then angular), and that one extra slot is the entire difference. This is
# also why `qvel` is not an element-wise derivative of `qpos`: the orientation part needs
# quaternion algebra. (a) confuses actuators with degrees of freedom, and neither (c) nor (d)
# describes anything MuJoCo does — time lives in `data.time`, and the world body has no
# joint at all.
#
# **3 — (c) the model is cheap; accuracy is a separate question.** The real-time factor is a
# throughput measurement: simulated seconds divided by wall seconds. A high value says this
# model is small and this CPU is fast, nothing more. Accuracy is governed by the timestep and
# the integrator, which the factor never touches — (a) and (b) both invent a coupling that
# does not exist, and MuJoCo never skips a step behind your back. (d) is backwards: a
# *smaller* timestep means more steps per simulated second, which pushes the factor down.
#
# **4 — (c) `data.xipos` is derived and nothing has recomputed it.** `qpos` is the
# independent state; `xipos` is a result of forward kinematics over it. Assigning to `qpos`
# marks nothing dirty and triggers no recomputation, so every derived array still describes
# the previous pose until `mj_forward` (or the next `mj_step`) runs. This is the single most
# common source of confusing MuJoCo output, and the fix is one line: call `mj_forward` after
# you write state and before you read anything derived from it.
