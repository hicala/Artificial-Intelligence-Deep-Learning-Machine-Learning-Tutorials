# %% [markdown]
# # F15-L08 · Sim-to-real, and what a free tier cannot teach you — reference solution
#
# The student notebook with every stub filled in. Runs top to bottom in a fresh interpreter
# and prints real measured output. Self-check answers are in the comment block at the foot.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import hashlib
import math
import time
from pathlib import Path

import mujoco
import numpy as np

print("mujoco", mujoco.__version__, "· numpy", np.__version__)

MODEL_FILENAME = "balancer.xml"


def balancer_xml_path() -> Path:
    """Locate the balancer model that ships next to this notebook."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:  # a notebook has no __file__
        here = Path.cwd()
    for candidate in (here / "assets" / MODEL_FILENAME,
                      here.parent / "assets" / MODEL_FILENAME,
                      Path.cwd() / "assets" / MODEL_FILENAME,
                      Path.cwd().parent / "assets" / MODEL_FILENAME):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{MODEL_FILENAME} not found next to this lesson. It ships in assets/ and is never "
        "downloaded; restore it from the lesson directory."
    )


def load_balancer():
    """Compile the balancer and hand back a fresh (model, data) pair."""
    model = mujoco.MjModel.from_xml_path(str(balancer_xml_path()))
    return model, mujoco.MjData(model)


MODEL, DATA = load_balancer()
TORSO_ID = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_BODY, "torso")
BASE_TORSO_MASS = float(MODEL.body_mass[TORSO_ID])
CTRL_RANGE = MODEL.actuator_ctrlrange.copy()

HORIZON_STEPS = 500
FALL_ANGLE = 0.6
NOMINAL = {"mass_scale": 1.0, "frictionloss": 0.0, "delay_steps": 0, "noise_std": 0.0,
           "seed": 0}

print(f"model: nq={MODEL.nq} nv={MODEL.nv} nu={MODEL.nu} dt={MODEL.opt.timestep} s")
print(f"torso mass {BASE_TORSO_MASS:.1f} kg, total {MODEL.body_mass.sum():.1f} kg")
print(f"torque limits (N*m): ankle {CTRL_RANGE[0]}, hip {CTRL_RANGE[1]}")
print(f"horizon {HORIZON_STEPS} steps = {HORIZON_STEPS * MODEL.opt.timestep:.1f} s "
      f"of simulated time; a lean past {FALL_ANGLE} rad counts as fallen")

# %% [markdown]
# ## 1. The gap is not one thing, it is four

# %%
BASELINE_GAINS = np.array([346.8, 52.6, 27.0, 6.7, -18.9, 54.9, -6.2, 4.1])


def reset_to_lean(model, data) -> None:
    """Put the machine back on the one keyframe the model ships: leaning and falling."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)


reset_to_lean(MODEL, DATA)
print(f"start pose: ankle {DATA.qpos[0]:+.3f} rad, hip {DATA.qpos[1]:+.3f} rad, "
      f"ankle rate {DATA.qvel[0]:+.3f} rad/s")
print("it is already falling forwards; the controller has to catch it")

# %% [markdown]
# ## 2. Exercise 1 — `rollout`, the instrument everything else is built on

# %%
def rollout(model, data, gains, delay_steps: int = 0, noise_std: float = 0.0,
            seed: int = 0) -> dict:
    """Run the balancing controller for one episode and report how it went."""
    reset_to_lean(model, data)
    rng = np.random.default_rng(seed)
    gain_matrix = np.asarray(gains, dtype=float).reshape(2, 4)
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]

    # The pipeline starts FULL of zero commands, so the first `delay_steps` steps apply a
    # stale (zero) command. Start it empty and the delay silently does not exist.
    pipeline = [np.zeros(model.nu) for _ in range(delay_steps)]
    squared_lean = 0.0
    saturated_steps = 0

    for step in range(HORIZON_STEPS):
        true_state = np.concatenate([data.qpos, data.qvel])
        # Noise corrupts the controller's VIEW of the world, never the world.
        observed = true_state + rng.normal(0.0, noise_std, size=4) if noise_std else true_state
        command = -gain_matrix @ observed

        pipeline.append(command)
        applied = pipeline.pop(0)

        # Measure saturation on the RAW command, before MuJoCo silently clamps it.
        if np.any(applied < low) or np.any(applied > high):
            saturated_steps += 1

        data.ctrl[:] = applied
        mujoco.mj_step(model, data)

        lean = float(data.qpos[0])
        squared_lean += lean * lean
        if abs(lean) > FALL_ANGLE:
            taken = step + 1
            return {"survived": False, "steps": taken, "rms_lean": float("inf"),
                    "saturated_fraction": saturated_steps / taken}

    return {"survived": True, "steps": HORIZON_STEPS,
            "rms_lean": math.sqrt(squared_lean / HORIZON_STEPS),
            "saturated_fraction": saturated_steps / HORIZON_STEPS}


def _check_rollout() -> None:
    model, data = load_balancer()
    r = rollout(model, data, BASELINE_GAINS)
    assert set(r) == {"survived", "steps", "rms_lean", "saturated_fraction"}, (
        f"keys were {sorted(r)} — return a dict with exactly those four names."
    )
    assert r["survived"] and r["steps"] == HORIZON_STEPS, (
        f"the baseline gains survived={r['survived']} for {r['steps']} steps with every gap "
        "switched off. They are known good, so the loop is wrong: the usual cause is "
        "forgetting reset_to_lean, or applying +K @ x instead of -K @ x."
    )
    assert 0.0 < r["rms_lean"] < 0.2, (
        f"rms_lean was {r['rms_lean']} — it is the root MEAN square of data.qpos[0] over the "
        "steps taken, so divide by the step count before taking the square root."
    )
    assert 0.0 < r["saturated_fraction"] < 1.0, (
        f"saturated_fraction was {r['saturated_fraction']} — at these gains the ankle spends "
        "part, but not all, of the episode against its 40 N*m limit. Exactly 0.0 means you "
        "compared the CLAMPED command against the range instead of the raw one."
    )
    took = data.time / model.opt.timestep
    assert abs(took - HORIZON_STEPS) < 1e-6, (
        f"you reported {r['steps']} steps, but MuJoCo's own clock says {took:.0f} steps were "
        f"simulated. The loop runs range({HORIZON_STEPS}) — check the bound — and `steps` is "
        "the number of steps actually taken, not a label."
    )
    slow = rollout(model, data, BASELINE_GAINS, delay_steps=2)
    assert slow["rms_lean"] > r["rms_lean"] * 1.3, (
        f"8 ms of latency changed rms_lean from {r['rms_lean']:.4f} to {slow['rms_lean']:.4f} "
        "— that is too little. Your pipeline is not actually delaying: it must START with "
        "delay_steps zeros, so the first commands applied are stale."
    )
    quiet = rollout(model, data, BASELINE_GAINS, noise_std=0.05, seed=1)
    again = rollout(model, data, BASELINE_GAINS, noise_std=0.05, seed=1)
    assert quiet["rms_lean"] == again["rms_lean"], (
        "the same seed gave two different answers — build the Generator once, before the "
        "loop, not inside it."
    )
    print(f"exercise 1 looks right: clean rms {r['rms_lean']:.4f}, "
          f"saturated {r['saturated_fraction']:.0%} of the episode")


# %% [markdown]
# ## 3. Saturation: the lie the controller is never told

# %%
_m, _d = load_balancer()
_d.ctrl[:] = [1e6, 1e6]
mujoco.mj_step(_m, _d)
print(f"wrote ctrl      = {_d.ctrl}")
print(f"motor delivered = {_d.actuator_force}   <- clamped to ctrlrange, silently")
print(f"the ankle's real ceiling is {CTRL_RANGE[0][1]:.0f} N*m, and the controller "
      "cannot tell from data.ctrl that it ever hit it")

# %% [markdown]
# ## 4. Exercise 2 — `apply_parameters`, and the hazard of a mutable model

# %%
def apply_parameters(model, data, mass_scale: float = 1.0,
                     frictionloss: float = 0.0) -> None:
    """Set the torso mass and the joint dry friction on a compiled model, in place."""
    # Set FROM the stored baseline. `*=` would compound on every call.
    model.body_mass[TORSO_ID] = BASE_TORSO_MASS * mass_scale
    model.dof_frictionloss[:] = frictionloss
    # Rebuild the quantities MuJoCo DERIVED from the old mass (body_subtreemass, dof_M0,
    # body_invweight0, ...). On this model the rollout is identical either way — the mass
    # matrix is rebuilt every step — so the honest statement is that the derived bookkeeping
    # goes stale, not that the physics breaks. Section 11's cell shows it.
    mujoco.mj_setConst(model, data)


def _check_apply_parameters() -> None:
    model, data = load_balancer()
    apply_parameters(model, data, mass_scale=1.5)
    got = float(model.body_mass[TORSO_ID])
    assert abs(got - BASE_TORSO_MASS * 1.5) < 1e-9, (
        f"torso mass is {got:.3f}, expected {BASE_TORSO_MASS * 1.5:.3f}."
    )
    apply_parameters(model, data, mass_scale=1.5)
    twice = float(model.body_mass[TORSO_ID])
    assert abs(twice - got) < 1e-9, (
        f"calling it twice with the same scale gave {got:.3f} then {twice:.3f} — you are "
        "multiplying the current mass instead of setting it from BASE_TORSO_MASS."
    )
    apply_parameters(model, data, frictionloss=3.0)
    assert np.allclose(model.dof_frictionloss, 3.0), (
        f"dof_frictionloss is {model.dof_frictionloss} — set every entry, and note this call "
        "also restores mass_scale to its default of 1.0."
    )
    assert abs(float(model.body_mass[TORSO_ID]) - BASE_TORSO_MASS) < 1e-9, (
        "that last call left mass_scale at its default of 1.0, so the torso mass should be "
        "back at the baseline; it is not, so you are not setting from the baseline."
    )
    print("exercise 2 looks right: mass and friction set from baseline, no compounding")


# %% [markdown]
# ## 5. Exercise 3 — `evaluate`, one condition end to end

# %%
def evaluate(model, data, gains, condition: dict) -> dict:
    """Apply one condition to the model and run one rollout under it."""
    # Two keys are MODEL edits...
    apply_parameters(model, data,
                     mass_scale=condition["mass_scale"],
                     frictionloss=condition["frictionloss"])
    # ...and three are LOOP arguments.
    return rollout(model, data, gains,
                   delay_steps=condition["delay_steps"],
                   noise_std=condition["noise_std"],
                   seed=condition["seed"])


def survival_rate(model, data, gains, conditions: list) -> float:
    """Fraction of `conditions` the controller survives. Provided; uses your `evaluate`."""
    if not conditions:
        return 0.0
    return sum(bool(evaluate(model, data, gains, c)["survived"])
               for c in conditions) / len(conditions)


def _check_evaluate() -> None:
    model, data = load_balancer()
    assert evaluate(model, data, BASELINE_GAINS, NOMINAL)["survived"], (
        "the nominal condition is the one the baseline gains were tuned on; it must survive. "
        "Check you are passing noise_std and delay_steps through rather than dropping them."
    )
    heavy = evaluate(model, data, BASELINE_GAINS, dict(NOMINAL, mass_scale=1.3))
    assert not heavy["survived"], (
        "a 30% heavier torso must topple these gains — if it survived, mass_scale is not "
        "reaching apply_parameters."
    )
    back = evaluate(model, data, BASELINE_GAINS, NOMINAL)
    assert back["survived"], (
        "the nominal condition failed straight after a heavy one: the model kept the heavy "
        "mass. apply_parameters must set from the baseline on every call."
    )
    print("exercise 3 looks right: conditions apply, and they do not leak into each other")


# %% [markdown]
# ## 6. Where each gap actually bites

# %%
def sweep(name: str, key: str, values, model=None, data=None) -> list:
    """Turn one knob across `values`, holding the rest at NOMINAL, and print the result."""
    if model is None:
        model, data = load_balancer()
    rows = []
    print(f"\n{name}")
    print(f"  {'value':>10s}  {'survived':>8s} {'steps':>6s} {'rms lean':>9s} {'saturated':>9s}")
    for v in values:
        r = evaluate(model, data, BASELINE_GAINS, dict(NOMINAL, **{key: v}))
        rows.append((v, r))
        rms = "fell" if r["rms_lean"] == float("inf") else f"{r['rms_lean']:.4f}"
        print(f"  {v:>10} {str(r['survived']):>9s} {r['steps']:>6d} {rms:>9s} "
              f"{r['saturated_fraction']:>8.0%}")
    return rows


def _check_sweeps() -> None:
    dt_ms = MODEL.opt.timestep * 1000
    lat = sweep(f"LATENCY (one step = {dt_ms:.0f} ms)", "delay_steps", [0, 2, 4, 8, 12, 16, 20])
    mass = sweep("TORSO MASS (multiplier)", "mass_scale", [0.8, 1.0, 1.1, 1.2, 1.3])
    fric = sweep("DRY FRICTION (N*m)", "frictionloss", [0.0, 5.0, 10.0, 20.0, 30.0])
    noise = sweep("SENSOR NOISE (rad, std)", "noise_std", [0.0, 0.02, 0.08, 0.12, 0.20])

    def first_fall(rows):
        for v, r in rows:
            if not r["survived"]:
                return v
        return None

    print(f"\nfirst latency that topples it : {first_fall(lat)} steps "
          f"({(first_fall(lat) or 0) * dt_ms:.0f} ms)")
    print(f"first mass multiplier that does: {first_fall(mass)}")
    print(f"first friction that does       : {first_fall(fric)}  "
          "<- dry friction degrades tracking without toppling it")
    print(f"first noise level that does    : {first_fall(noise)}")
    assert first_fall(lat) is not None, "latency should eventually topple it"
    assert first_fall(mass) is not None, "a heavy enough torso should topple it"
    assert first_fall(noise) is not None, "enough sensor noise should topple it"
    # The sentence printed below is a claim about this machine, so it is asserted rather
    # than merely typed: if dry friction ever did topple it, the prose would be a lie.
    assert first_fall(fric) is None, (
        f"dry friction toppled the machine at {first_fall(fric)} N*m, so the conclusion "
        "printed below no longer holds — re-read the friction table before trusting it"
    )
    assert fric[0][1]["rms_lean"] < fric[-1][1]["rms_lean"], (
        "friction should make the wobble worse even where it does not cause a fall"
    )
    wobble_ratio = fric[-1][1]["rms_lean"] / fric[0][1]["rms_lean"]
    print(f"\nread that again: friction never topples it, and still multiplies the wobble by "
          f"{wobble_ratio:.1f}x while pinning the ankle at its limit for "
          f"{fric[-1][1]['saturated_fraction']:.0%} of the episode."
          "\nSurvival is not the only thing worth measuring, and it is often the last to move.")


# %% [markdown]
# ## 7. Exercise 4 — the domain-randomisation wrapper

# %%
DR_SPEC = {
    "mass_scale": (0.85, 1.30),
    "frictionloss": (0.0, 15.0),
    "delay_steps": (0, 9),        # integers, upper bound exclusive
    "noise_std": (0.0, 0.03),
}
HELD_OUT_SEED, HELD_OUT_N = 4242, 40
DR_TRAIN_SEED, DR_TRAIN_N = 707, 10
TUNE_SEED = 1


def sample_conditions(n: int, seed: int, spec: dict = DR_SPEC) -> list:
    """Draw `n` random conditions from `spec`. This is the domain-randomisation wrapper."""
    rng = np.random.default_rng(seed)
    conditions = []
    for _ in range(n):
        # The draw ORDER is the contract: it is what makes a seed reproducible.
        conditions.append({
            "mass_scale": float(rng.uniform(*spec["mass_scale"])),
            "frictionloss": float(rng.uniform(*spec["frictionloss"])),
            "delay_steps": int(rng.integers(*spec["delay_steps"])),
            "noise_std": float(rng.uniform(*spec["noise_std"])),
            # A fresh rollout seed per condition, from the same generator.
            "seed": int(rng.integers(0, 10000)),
        })
    return conditions


def ensemble_cost(model, data, gains, conditions: list) -> float:
    """Mean cost over conditions: wobble if it survived, a graded penalty if it fell."""
    total = 0.0
    for c in conditions:
        r = evaluate(model, data, gains, c)
        total += r["rms_lean"] if r["survived"] else 10.0 - 8.0 * (r["steps"] / HORIZON_STEPS)
    return total / len(conditions)


def cem_tune(conditions: list, seed: int = TUNE_SEED, iterations: int = 8,
             population: int = 40, elite: int = 8) -> np.ndarray:
    """Cross-entropy search for gains that do well across `conditions`. Provided."""
    model, data = load_balancer()
    rng = np.random.default_rng(seed)
    mu = np.array([150.0, 30.0, 45.0, 8.0, -10.0, 50.0, -4.0, 10.0])
    sd = np.abs(mu) * 0.7 + 4.0
    for _ in range(iterations):
        candidates = rng.normal(mu, sd, size=(population, 8))
        scores = np.array([ensemble_cost(model, data, g, conditions) for g in candidates])
        keep = candidates[np.argsort(scores)[:elite]]
        mu, sd = keep.mean(axis=0), keep.std(axis=0) + 1e-3
    return mu


def _check_sample_conditions() -> None:
    got = sample_conditions(5, seed=0)
    assert len(got) == 5, f"asked for 5 conditions, got {len(got)}"
    assert sorted(got[0]) == sorted(NOMINAL), (
        f"a condition has keys {sorted(got[0])}, expected {sorted(NOMINAL)}"
    )
    assert got == sample_conditions(5, seed=0), (
        "the same seed produced different conditions — build the Generator once, from the "
        "seed you were given, and draw in the documented order."
    )
    assert got != sample_conditions(5, seed=1), "different seeds must give different draws"
    for c in got:
        assert DR_SPEC["mass_scale"][0] <= c["mass_scale"] <= DR_SPEC["mass_scale"][1], (
            f"mass_scale {c['mass_scale']} is outside {DR_SPEC['mass_scale']}"
        )
        assert isinstance(c["delay_steps"], int), (
            f"delay_steps is {type(c['delay_steps']).__name__}; it indexes a list, so make "
            "it an int with int(), not a numpy integer from rng.integers"
        )
    assert len({c["seed"] for c in got}) > 1, (
        "every condition got the same rollout seed — draw a fresh one per condition from the "
        "same generator, do not reuse the argument `seed`"
    )
    print("exercise 4 looks right: reproducible, in range, one rollout seed per condition")


# %% [markdown]
# ## 8. Does it actually narrow the gap?

# %%
def compare_nominal_vs_dr() -> dict:
    """Tune two controllers, score both on held-out conditions, return the comparison."""
    model, data = load_balancer()
    held_out = sample_conditions(HELD_OUT_N, seed=HELD_OUT_SEED)
    train_dr = sample_conditions(DR_TRAIN_N, seed=DR_TRAIN_SEED)

    t0 = time.perf_counter()
    gains_nominal = cem_tune([NOMINAL])
    t1 = time.perf_counter()
    gains_dr = cem_tune(train_dr)
    t2 = time.perf_counter()

    out = {
        "nominal_on_nominal": survival_rate(model, data, gains_nominal, [NOMINAL]),
        "nominal_on_held_out": survival_rate(model, data, gains_nominal, held_out),
        "dr_on_held_out": survival_rate(model, data, gains_dr, held_out),
        "tune_seconds_nominal": t1 - t0,
        "tune_seconds_dr": t2 - t1,
    }
    out["gap_closed"] = out["dr_on_held_out"] - out["nominal_on_held_out"]
    return out


def _check_comparison() -> None:
    r = compare_nominal_vs_dr()
    print(f"tuned on the nominal model alone   ({r['tune_seconds_nominal']:.1f}s)")
    print(f"tuned on {DR_TRAIN_N} randomised models     ({r['tune_seconds_dr']:.1f}s)")
    print(f"\n  nominal-tuned, scored on the model it was tuned on : "
          f"{r['nominal_on_nominal']:.0%}")
    print(f"  nominal-tuned, scored on {HELD_OUT_N} held-out models      : "
          f"{r['nominal_on_held_out']:.0%}")
    print(f"  DR-tuned,      scored on the same held-out models  : {r['dr_on_held_out']:.0%}")
    print(f"\n  gap closed by domain randomisation: {r['gap_closed']:+.1%}")
    assert r["nominal_on_nominal"] == 1.0, (
        "the nominally-tuned controller should be perfect on the model it was tuned on — "
        "that is exactly what makes its held-out score so instructive"
    )
    assert r["dr_on_held_out"] > r["nominal_on_held_out"], (
        f"DR scored {r['dr_on_held_out']:.0%} against nominal's {r['nominal_on_held_out']:.0%} "
        "on held-out conditions. Expected DR to win — check that cem_tune is receiving the "
        "randomised list and not [NOMINAL]."
    )
    print("\n  100% on the model you tuned on is not a result. It is the definition of the"
          "\n  model you tuned on. Only the held-out column is evidence of anything.")


# %% [markdown]
# ## 9. Exercise 5 — what this course cannot give you

# %%
ISAAC_MIN = {
    "min_vram_gib": 16,
    "requires_rt_cores": True,
    "reference_gpu": "GeForce RTX 4080",
    "source": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html",
}

GPU_LEDGER = [
    {"name": "GeForce RTX 4080", "vram_gib": 16, "rt_cores": True, "named_by_nvidia": True,
     "found_on": "a workstation you bought"},
    {"name": "RTX PRO 6000 Blackwell", "vram_gib": 48, "rt_cores": True,
     "named_by_nvidia": True, "found_on": "a workstation you really bought"},
    {"name": "NVIDIA T4", "vram_gib": 16, "rt_cores": True, "named_by_nvidia": False,
     "found_on": "Colab free tier, when you are given one at all"},
    {"name": "NVIDIA L4", "vram_gib": 24, "rt_cores": True, "named_by_nvidia": False,
     "found_on": "cloud notebook tiers"},
    {"name": "NVIDIA A100", "vram_gib": 40, "rt_cores": False, "named_by_nvidia": False,
     "found_on": "research clusters; named by NVIDIA as unsupported"},
    {"name": "this machine's CPU", "vram_gib": 0, "rt_cores": False, "named_by_nvidia": False,
     "found_on": "where this entire flagship runs"},
]


def can_run_isaac_sim(gpu: dict, requirement: dict = ISAAC_MIN) -> dict:
    """Decide whether one GPU meets NVIDIA's published Isaac Sim requirements."""
    reasons = []
    # Read `requirement`, not the ISAAC_MIN global: the grader relaxes it to check.
    if requirement["requires_rt_cores"] and not gpu["rt_cores"]:
        reasons.append("no RT Cores, which NVIDIA lists as unsupported")
    if gpu["vram_gib"] < requirement["min_vram_gib"]:
        # Note the >= semantics: 16 GiB MEETS a 16 GiB floor. The T4 turns on this.
        reasons.append(f"{gpu['vram_gib']} GiB VRAM, below the "
                       f"{requirement['min_vram_gib']} GiB minimum")
    if not gpu["named_by_nvidia"]:
        reasons.append("not named in NVIDIA's requirements table (minimum listed: "
                       f"{requirement['reference_gpu']})")
    return {"ok": not reasons, "reasons": reasons}


def _check_can_run_isaac_sim() -> None:
    print(f"{'GPU':<26s} {'VRAM':>5s} {'RT':>6s} {'runs Isaac Sim?':>16s}   why not")
    for gpu in GPU_LEDGER:
        v = can_run_isaac_sim(gpu)
        why = "" if v["ok"] else "; ".join(v["reasons"])
        print(f"{gpu['name']:<26s} {gpu['vram_gib']:>4d}G {str(gpu['rt_cores']):>6s} "
              f"{str(v['ok']):>16s}   {why}")
    assert can_run_isaac_sim(GPU_LEDGER[0])["ok"], "the RTX 4080 IS the documented minimum"
    t4 = can_run_isaac_sim(GPU_LEDGER[2])
    assert not t4["ok"] and len(t4["reasons"]) == 1, (
        f"the T4 produced {len(t4['reasons'])} reasons: {t4['reasons']}. It has RT Cores and "
        "16 GiB, so it must fail exactly one check — the 'not named' one. If you gave it a "
        "VRAM reason, your comparison is > rather than >=."
    )
    a100 = can_run_isaac_sim(GPU_LEDGER[4])
    assert len(a100["reasons"]) == 2, (
        f"the A100 produced {len(a100['reasons'])} reasons: {a100['reasons']}. It has 40 GiB "
        "but no RT Cores and is not named, so it fails exactly two."
    )
    print("\nexercise 5 looks right. Note the T4 row: it fails for being unnamed, NOT for"
          "\nlacking RT Cores. Check the claim, not the folklore.")


# %% [markdown]
# ## 10. The hardware ladder, with prices that carry their receipts

# %%
HARDWARE_LADDER = [
    {"rung": "a 6-DoF arm, servo kit", "cost_usd": 249.90, "kind": "buy",
     "what_it_teaches": "real actuators, real backlash, real latency",
     "source": "https://www.seeedstudio.com/SO-ARM101-Low-Cost-AI-Arm-Kit-p-6426.html"},
    {"rung": "Reachy Mini Lite", "cost_usd": 399.00, "kind": "buy",
     "what_it_teaches": "an open-source robot with a real sensing loop",
     "source": "https://huggingface.co/blog/reachy-mini"},
    {"rung": "Unitree Go2 quadruped", "cost_usd": 2800.00, "kind": "buy",
     "what_it_teaches": "contact, slip and balance on hardware",
     "source": "https://shop.unitree.com/products/unitree-go2"},
    {"rung": "Berkeley Humanoid Lite (build it)", "cost_usd": 5000.00, "kind": "build",
     "what_it_teaches": "a whole open humanoid, printed on a desktop printer",
     "source": "https://arxiv.org/abs/2504.17249"},
    {"rung": "Unitree G1 humanoid", "cost_usd": 13500.00, "kind": "buy",
     "what_it_teaches": "the real thing, and the real repair bill",
     "source": "https://shop.unitree.com/products/unitree-g1"},
    {"rung": "rent an RTX 4090, per hour", "cost_usd": 0.74, "kind": "rent",
     "what_it_teaches": "Isaac Sim, without owning the card",
     "source": "https://www.runpod.io/pricing"},
    {"rung": "rent an RTX 5090, per hour", "cost_usd": 0.99, "kind": "rent",
     "what_it_teaches": "the same, faster",
     "source": "https://www.runpod.io/pricing"},
]


def ladder_table(ladder: list = HARDWARE_LADDER) -> None:
    """Print the ladder, and compute how much rental time each purchase price would buy."""
    hourly = min(r["cost_usd"] for r in ladder if r["kind"] == "rent")
    print(f"\n{'rung':<36s} {'USD':>9s}  {'kind':<6s} what it teaches")
    for r in sorted(ladder, key=lambda row: row["cost_usd"]):
        print(f"{r['rung']:<36s} {r['cost_usd']:>9,.2f}  {r['kind']:<6s} "
              f"{r['what_it_teaches']}")
    print(f"\nCheapest rented RTX hour on the ladder: ${hourly:.2f}.")
    for r in sorted(ladder, key=lambda row: row["cost_usd"]):
        if r["kind"] != "rent":
            print(f"  the {r['rung']} budget instead buys {r['cost_usd'] / hourly:,.0f} "
                  f"GPU-hours of Isaac Sim")
    print("\nEvery source URL is in claims.yaml with the date it was read. Prices move; the"
          "\nmethod of checking them before you repeat them does not.")


# %% [markdown]
# ## 11. Common mistakes, and the one you can watch happen

# %%
_stale_model, _stale_data = load_balancer()
print(f"total mass on file          : {_stale_model.body_subtreemass[0]:.1f} kg")
_stale_model.body_mass[TORSO_ID] = BASE_TORSO_MASS * 2.0      # the edit, without mj_setConst
print(f"after doubling the torso    : {_stale_model.body_subtreemass[0]:.1f} kg   <- stale, "
      f"though body_mass now reads {_stale_model.body_mass[TORSO_ID]:.1f} kg")
mujoco.mj_setConst(_stale_model, _stale_data)
print(f"after mujoco.mj_setConst    : {_stale_model.body_subtreemass[0]:.1f} kg   <- rebuilt")

# %% [markdown]
# ## 12. Self-check

# %%
SELF_CHECK = {1: "c", 2: "b", 3: "b", 4: "c"}

_ANSWER_DIGESTS = {1: "6f62e7f0082bcf7c", 2: "c5c3917d88602bae",
                   3: "03e5e02cc8ff791f", 4: "df9689b878adb253"}
_ANSWER_SECTIONS = {
    1: "section 8 — what a held-out score is for, and what a training score is not",
    2: "section 7 — which of the two randomisation papers this lesson actually follows",
    3: "section 9 — the T4 row of the ledger you computed, not the folklore about it",
    4: "section 6 — the saturation column of the latency sweep you ran",
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the four multiple-choice answers in SELF_CHECK, naming where to look again."""
    answers = SELF_CHECK if answers is None else answers
    wrong = []
    for q, digest in sorted(_ANSWER_DIGESTS.items()):
        got = str(answers.get(q, "?")).strip().lower()
        if hashlib.sha256(f"F15-L08-q{q}-{got}".encode()).hexdigest()[:16] != digest:
            wrong.append(q)
    for q in sorted(_ANSWER_DIGESTS):
        note = f"  -> re-read {_ANSWER_SECTIONS[q]}" if q in wrong else ""
        print(f"  q{q}: {'wrong' if q in wrong else 'right'}{note}")
    assert not wrong, (
        f"questions {wrong} are still wrong. Each one names the section that answers it "
        "above — go back to the measurement you ran there rather than guessing a letter."
    )
    print("self-check: all four right")


# %% [markdown]
# ## What you built, and where it goes next

# %%
if __name__ == "__main__":
    _check_rollout()
    _check_apply_parameters()
    _check_evaluate()
    _check_sweeps()
    _check_sample_conditions()
    _check_comparison()
    _check_can_run_isaac_sim()
    ladder_table()
    _check_self_check()

# %% [markdown]
# ## Self-check answers
#
# **1 — (c).** The score on the conditions you tuned on is 100% by construction; it is the
# definition of the set you optimised against, not evidence about anything else. Only the
# held-out column carries information. (a) is the reflex of every overfitted model and is
# unfalsifiable as stated; (b) misreads a measurement of the gap as a postponement of it;
# (d) is the specific wrong move — higher gains are what saturate the actuator sooner, and
# section 6 shows saturation already at 100% before survival moves at all.
#
# **2 — (b).** Tobin et al. (2017) randomised *appearance* — textures, lighting, camera pose —
# for a vision model. Peng et al. (2017) randomised *dynamics* — mass, friction, damping — for
# a control policy. This controller never sees a pixel; it reads four joint numbers. Only the
# dynamics can possibly matter to it, so this is the Peng variant. (a) inverts the two; (c) is
# simply false; (d) collapses a distinction that decides what you should randomise.
#
# **3 — (b).** The popular explanation is factually wrong, which is why the lesson makes you
# compute the verdict instead of repeating it. NVIDIA's own product page says the T4 "features
# multi-precision Turing Tensor Cores and new RT Cores", and its 16 GB meets the documented
# 16 GB floor exactly — so a `>=` comparison passes it. It fails the third check: it is not a
# GPU NVIDIA names in the requirements table, whose stated minimum is a GeForce RTX 4080. The
# second, independent obstacle is that Google's own FAQ says the available GPU types vary over
# time and are not guaranteed, so you cannot build a course on being handed one.
#
# **4 — (c).** Saturation is the leading indicator and survival is the lagging one. Once the
# controller is against its torque limit for the whole episode, it has no authority left in
# reserve: the next disturbance — a heavier payload, a little more latency, a gust — has
# nothing to draw on. The measurement in section 6 is exactly this, and it is why the rubric
# scores saturation separately from survival. (a) mistakes a surviving episode for a healthy
# one; (b) is wrong on the mechanics — a stale command is a larger command by the time it
# lands; (d) treats the instrument as the problem, and a torque limit you raise in simulation
# is a motor you do not own.
