# %% [markdown]
# # T03-L03 · The CPU slice of nanolm — REFERENCE SOLUTION
#
# The complete implementation. Every stub in `lesson.py` is filled in here, the file runs top
# to bottom in a fresh interpreter, and every number it prints is measured by the code.
#
# Self-check answers are in the comment block at the foot of this file.

# %%
# Setup. Everything the lesson needs, in one cell, with versions printed by the code itself.
import contextlib
import io
import sys
import time
from typing import Callable

import numpy as np

print("python", sys.version.split()[0])
print("numpy ", np.__version__)

try:  # noqa: SIM105 - the failure is the point, so it is reported rather than swallowed
    import torch

    _TORCH = torch.__version__
except ImportError:
    _TORCH = "not installed — which is exactly why you are about to differentiate by hand"
print("torch ", _TORCH)

# Every check and every demo below runs through `_try`, so pressing Run all before you have
# written a line reaches the progress board at the foot of the notebook instead of stopping at
# the first stub.
# The six exercises, in the order the notebook meets them: (label, function, where).
_EXERCISES = [("exercise 1", "softmax", "section 3"), ("exercise 2", "cross_entropy", "section 4"),
              ("exercise 3", "attention_forward", "section 6"),
              ("exercise 4", "cross_entropy_backward", "section 8"),
              ("exercise 5", "attention_backward", "section 9"),
              ("exercise 6", "make_batch", "section 12")]
_FUNCTION = {label: function for label, function, _ in _EXERCISES}
_FOR_GRADIENTS = ("exercise 1", "exercise 2", "exercise 3", "exercise 4", "exercise 5")
_STATUS: dict = {}       # label -> "passed" | "failed" | "not started", for the progress board
_WAITING_ON: dict = {}   # label -> the stub whose NotImplementedError stopped it


def _unfinished(exc: BaseException) -> str:
    """The function that raised NotImplementedError: the innermost frame of the traceback.

    Exercise 3 calls your exercise 1, so an unfilled `softmax` stops exercise 3's check too —
    and naming the stub that actually raised says which one to go back to.
    """
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return tb.tb_frame.f_code.co_name if tb is not None else ""


def _named(labels: list) -> str:
    """["exercise 3"] -> "exercise 3 (`attention_forward`)"; several -> "exercises 1, 3 and 5"."""
    if len(labels) == 1:
        return f"{labels[0]} (`{_FUNCTION[labels[0]]}`)"
    nums = [label.split()[-1] for label in labels]
    return "exercises " + ", ".join(nums[:-1]) + " and " + nums[-1]


def _try(label: str, check: Callable[[], None], needs: tuple = ()) -> None:
    """Run a check, or a demo that depends on your code, without derailing the notebook.

    A stub you have not filled in yet simply says so, by name. A wrong answer prints the
    check's own message — which names the likely mistake — and the notebook carries on, so one
    broken exercise never hides the feedback on the others. A demo names the exercises it
    `needs`: until each has passed its check, the demo says which one it is waiting for and
    skips, so it never runs on an answer a check has just rejected. Nothing is swallowed:
    every outcome is recorded for the progress board, and a script run exits non-zero there if
    any check came back wrong.
    """
    waiting = [name for name, _, _ in _EXERCISES   # in the order you meet them
               if name in needs and _STATUS.get(name) != "passed"]
    if waiting:
        _STATUS[label] = "not started"
        print(f"{label}: skipped — needs {_named(waiting)} to pass first.")
        return
    try:
        check()
    except NotImplementedError as exc:
        stub = _unfinished(exc)
        _STATUS[label], _WAITING_ON[label] = "not started", stub
        owner = [name for name, function in _FUNCTION.items() if function == stub]
        if owner and owner[0] != label:
            print(f"{label}: skipped — needs {_named(owner)} first; it still raises "
                  "NotImplementedError.")
        else:
            todo = f"`{stub}`" if stub else "a stub"
            print(f"{label}: not implemented yet — {todo} still raises NotImplementedError. "
                  "Fill it in, then re-run this cell.")
    except AssertionError as exc:
        _STATUS[label] = "failed"
        print(f"{label}: FAILED — {exc}")
    except Exception as exc:  # a half-finished implementation raising something else
        _STATUS[label] = "failed"
        print(f"{label}: raised {type(exc).__name__}: {exc}")
    else:
        _STATUS[label] = "passed"

DTYPE = np.float64

# %% [markdown]
# ## 1. The corpus

# %%
CORPUS = """From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou, contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thyself thy foe, to thy sweet self too cruel:
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And tender churl mak'st waste in niggarding:
    Pity the world, or else this glutton be,
    To eat the world's due, by the grave and thee.

II

When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a tatter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.

III

Look in thy glass and tell the face thou viewest
Now is the time that face should form another;
Whose fresh repair if now thou not renewest,
Thou dost beguile the world, unbless some mother.
For where is she so fair whose unear'd womb
Disdains the tillage of thy husbandry?
Or who is he so fond will be the tomb,
Of his self-love to stop posterity?
Thou art thy mother's glass and she in thee
Calls back the lovely April of her prime;
So thou through windows of thine age shalt see,
Despite of wrinkles this thy golden time.
    But if thou live, remember'd not to be,
    Die single and thine image dies with thee.

IV

Unthrifty loveliness, why dost thou spend
Upon thyself thy beauty's legacy?
Nature's bequest gives nothing, but doth lend,
And being frank she lends to those are free:
Then, beauteous niggard, why dost thou abuse
The bounteous largess given thee to give?
Profitless usurer, why dost thou use
So great a sum of sums, yet canst not live?
For having traffic with thyself alone,
Thou of thyself thy sweet self dost deceive:
Then how when nature calls thee to be gone,
What acceptable audit canst thou leave?
    Thy unused beauty must be tombed with thee,
    Which, used, lives th' executor to be.

V

Those hours, that with gentle work did frame
The lovely gaze where every eye doth dwell,
Will play the tyrants to the very same
And that unfair which fairly doth excel;
For never-resting time leads summer on
To hideous winter, and confounds him there;
Sap checked with frost, and lusty leaves quite gone,
Beauty o'er-snowed and bareness every where:
Then were not summer's distillation left,
A liquid prisoner pent in walls of glass,
Beauty's effect with beauty were bereft,
Nor it, nor no remembrance what it was:
    But flowers distill'd, though they with winter meet,
    Leese but their show; their substance still lives sweet.


VI

Then let not winter's ragged hand deface,
In thee thy summer, ere thou be distill'd:
Make sweet some vial; treasure thou some place
With beauty's treasure ere it be self-kill'd.
That use is not forbidden usury,
Which happies those that pay the willing loan;
That's for thyself to breed another thee,
Or ten times happier, be it ten for one;
Ten times thyself were happier than thou art,
If ten of thine ten times refigur'd thee:
Then what could death do if thou shouldst depart,
Leaving thee living in posterity?
    Be not self-will'd, for thou art much too fair
    To be death's conquest and make worms thine heir.

VII

Lo! in the orient when the gracious light
Lifts up his burning head, each under eye
Doth homage to his new-appearing sight,
Serving with looks his sacred majesty;
And having climb'd the steep-up heavenly hill,
Resembling strong youth in his middle age,
Yet mortal looks adore his beauty still,
Attending on his golden pilgrimage:
But when from highmost pitch, with weary car,
Like feeble age, he reeleth from the day,
The eyes, 'fore duteous, now converted are
From his low tract, and look another way:
    So thou, thyself outgoing in thy noon:
    Unlook'd, on diest unless thou get a son.

VIII

Music to hear, why hear'st thou music sadly?
Sweets with sweets war not, joy delights in joy:
Why lov'st thou that which thou receiv'st not gladly,
Or else receiv'st with pleasure thine annoy?
If the true concord of well-tuned sounds,
By unions married, do offend thine ear,
They do but sweetly chide thee, who confounds
In singleness the parts that thou shouldst bear.
Mark how one string, sweet husband to another,
Strikes each in each by mutual ordering;
Resembling sire and child and happy mother,
Who, all in one, one pleasing note do sing:
    Whose speechless song being many, seeming one,
    Sings this to thee: 'Thou single wilt prove none.'

IX

Is it for fear to wet a widow's eye,
That thou consum'st thyself in single life?
Ah! if thou issueless shalt hap to die,
The world will wail thee like a makeless wife;
The world will be thy widow and still weep
That thou no form of thee hast left behind,
When every private widow well may keep
By children's eyes, her husband's shape in mind:
Look! what an unthrift in the world doth spend
Shifts but his place, for still the world enjoys it;
But beauty's waste hath in the world an end,
And kept unused the user so destroys it.
    No love toward others in that bosom sits
    That on himself such murd'rous shame commits.

X

For shame! deny that thou bear'st love to any,
Who for thyself art so unprovident.
Grant, if thou wilt, thou art belov'd of many,
But that thou none lov'st is most evident:
For thou art so possess'd with murderous hate,
That 'gainst thyself thou stick'st not to conspire,
Seeking that beauteous roof to ruinate
Which to repair should be thy chief desire.
"""

VOCAB = sorted(set(CORPUS))
VOCAB_SIZE = len(VOCAB)
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for ch, i in STOI.items()}


def encode(text: str) -> np.ndarray:
    """Map a string to an int64 array of character ids."""
    return np.array([STOI[ch] for ch in text], dtype=np.int64)


def decode(ids) -> str:
    """Map a sequence of character ids back to a string."""
    return "".join(ITOS[int(i)] for i in ids)


DATA = encode(CORPUS)
SPLIT = int(0.9 * len(DATA))
TRAIN_DATA, VAL_DATA = DATA[:SPLIT], DATA[SPLIT:]

print(f"corpus      {len(CORPUS)} characters")
print(f"vocabulary  {VOCAB_SIZE} distinct characters")
print(f"            {VOCAB!r}")
print(f"train/val   {len(TRAIN_DATA)} / {len(VAL_DATA)} characters")
print(f"round-trip  {decode(encode(CORPUS[:40]))!r}")

# %% [markdown]
# ## 2. Two baselines you must beat, both measured

# %%
def unigram_entropy(data: np.ndarray, vocab_size: int) -> float:
    """Entropy in nats of the character distribution of `data`. The no-context baseline."""
    counts = np.bincount(data, minlength=vocab_size).astype(DTYPE)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


UNIFORM_BASELINE = float(np.log(VOCAB_SIZE))
UNIGRAM_BASELINE = unigram_entropy(TRAIN_DATA, VOCAB_SIZE)

print(f"uniform baseline        {UNIFORM_BASELINE:.4f} nats/char  (ln {VOCAB_SIZE})")
print(f"unigram entropy (train) {UNIGRAM_BASELINE:.4f} nats/char  (counts only, no context)")
print(f"context is worth at most{UNIFORM_BASELINE - UNIGRAM_BASELINE:8.4f} nats before a model "
      "looks at a single neighbour")

# %% [markdown]
# ## 3. Exercise 1 — `softmax(x)`
#
# Subtract the row maximum, then exponentiate. `exp(x - m)/sum(exp(x - m))` equals
# `exp(x)/sum(exp(x))` exactly, because the `exp(m)` factors cancel, but the largest exponent
# is now `exp(0) == 1` so nothing can overflow. `-inf` entries survive as exactly 0.

# %%
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax over the LAST axis, computed so it cannot overflow.

    Example:
        >>> softmax(np.array([0.0, 0.0, 0.0]))
        array([0.33333333, 0.33333333, 0.33333333])
    """
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _check_softmax() -> None:
    flat = softmax(np.array([0.0, 0.0, 0.0]))
    assert flat.shape == (3,), f"softmax changed the shape to {flat.shape}; it must not"
    assert abs(float(flat.sum()) - 1.0) < 1e-12, (
        f"softmax([0,0,0]) sums to {float(flat.sum())}, not 1 — you normalised over the wrong "
        "axis, or forgot to divide by the sum at all."
    )

    batched = softmax(np.arange(24, dtype=DTYPE).reshape(2, 3, 4))
    sums = batched.sum(axis=-1)
    assert batched.shape == (2, 3, 4), f"expected shape (2,3,4), got {batched.shape}"
    assert np.allclose(sums, 1.0), (
        f"row sums were {sums.ravel()} — normalise over the LAST axis only, and use "
        "keepdims=True so the divide broadcasts back over that axis."
    )

    huge = softmax(np.array([10000.0, 10000.0, 9999.0]))
    assert np.isfinite(huge).all(), (
        "softmax overflowed to nan/inf on logits of 1e4 — you called np.exp on the raw input. "
        "Subtract the max along the last axis (keepdims=True) before exponentiating."
    )
    assert abs(float(huge.sum()) - 1.0) < 1e-12, f"huge-logit row sums to {float(huge.sum())}"

    masked = softmax(np.array([0.0, -np.inf, 1.0]))
    assert masked[1] == 0.0, (
        f"a -inf entry produced {masked[1]} instead of exactly 0.0 — do not clip -inf to a "
        "large negative number; exp(-inf) is already exactly 0."
    )

    shifted = softmax(np.array([1.0, 2.0, 3.0]))
    unshifted = softmax(np.array([1.0, 2.0, 3.0]) + 500.0)
    assert np.allclose(shifted, unshifted), (
        "adding a constant to every score changed the answer — softmax is shift-invariant, so "
        "if these differ you subtracted something that varies along the wrong axis."
    )
    print(f"exercise 1 looks right: uniform row {flat}, 1e4 logits stayed finite")


# %% [markdown]
# ## 4. Exercise 2 — `cross_entropy(logits, targets)`
#
# `loss = logsumexp(logits) - logits[target]`, with the row max pulled out of the logsumexp
# and added back outside the log. The tiny probability is never formed, so nothing underflows.

# %%
def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean negative log-likelihood in nats, averaged over every position.

    Example:
        >>> round(cross_entropy(np.zeros((1, 2, 4)), np.array([[0, 3]])), 6)
        1.386294
    """
    row_max = logits.max(axis=-1, keepdims=True)
    log_sum_exp = row_max[..., 0] + np.log(np.exp(logits - row_max).sum(axis=-1))
    chosen = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return float((log_sum_exp - chosen).mean())


def _check_cross_entropy() -> None:
    uniform = cross_entropy(np.zeros((2, 5, VOCAB_SIZE)), np.zeros((2, 5), dtype=np.int64))
    assert isinstance(uniform, float), (
        f"cross_entropy returned {type(uniform).__name__}; wrap the result in float()."
    )
    assert abs(uniform - UNIFORM_BASELINE) < 1e-12, (
        f"all-zero logits scored {uniform:.6f}, but a uniform distribution over {VOCAB_SIZE} "
        f"symbols must score ln({VOCAB_SIZE}) = {UNIFORM_BASELINE:.6f}. If you got "
        f"{uniform / 10:.6f} you summed instead of averaging over the batch."
    )

    confident = cross_entropy(
        np.array([[[0.0, 100.0, 0.0], [100.0, 0.0, 0.0]]]), np.array([[1, 0]])
    )
    assert confident < 1e-9, (
        f"a model that puts essentially all mass on the right answer scored {confident:.6f}, "
        "which should be ~0 — check you are selecting the TARGET's logit, not the largest one."
    )

    wrong_way = cross_entropy(
        np.array([[[0.0, 100.0, 0.0], [100.0, 0.0, 0.0]]]), np.array([[0, 1]])
    )
    assert wrong_way > 10.0, (
        f"a confidently WRONG model scored {wrong_way:.6f}; loss must be large when the "
        "target's logit is the low one. A near-zero answer means you ignored `targets`."
    )

    big = cross_entropy(np.full((1, 3, 4), 1e4), np.zeros((1, 3), dtype=np.int64))
    assert np.isfinite(big), (
        "logits of 1e4 produced nan/inf — you exponentiated before subtracting the row max. "
        "Use the log-sum-exp form with the max pulled out."
    )
    assert abs(big - np.log(4.0)) < 1e-9, (
        f"equal logits of 1e4 over 4 symbols scored {big:.6f}; it is still a uniform "
        f"distribution, so the answer is ln(4) = {np.log(4.0):.6f}."
    )

    scale_free = cross_entropy(np.zeros((4, 7, VOCAB_SIZE)), np.zeros((4, 7), dtype=np.int64))
    assert abs(scale_free - uniform) < 1e-12, (
        f"a (4,7) batch scored {scale_free:.6f} against {uniform:.6f} for a (2,5) batch — you "
        "divided by B or by T alone. Average over every one of the B*T positions."
    )
    print(f"exercise 2 looks right: uniform {uniform:.4f} nats, confident {confident:.2e}, "
          f"1e4 logits stayed finite at {big:.4f}")


# %% [markdown]
# ## 5. Causal masking: the mask goes *before* the softmax

# %%
def causal_mask(t: int) -> np.ndarray:
    """(T, T) boolean array: True where position i is ALLOWED to attend to position j."""
    return np.tril(np.ones((t, t), dtype=bool))


_m4 = causal_mask(4)
print("causal mask (True = allowed):")
print(_m4.astype(int))

_scores = np.array([[0.0, 1.0, 2.0, 3.0]])
_right = np.exp(np.where(_m4[0], _scores, -np.inf))
_right = _right / _right.sum()
_wrong = np.exp(_scores) / np.exp(_scores).sum()
_wrong = np.where(_m4[0], _wrong, 0.0)
print(f"\nrow 0, masked before softmax: {_right}  sums to {_right.sum():.6f}")
print(f"row 0, masked after  softmax: {_wrong}  sums to {_wrong.sum():.6f}")
print("the second row is not a probability distribution at all")


def sentinel_leak(sentinel) -> float:
    """Largest future attention weight surviving when the mask writes `sentinel`.

    Example:
        >>> sentinel_leak(-np.inf)
        0.0
    """
    rng = np.random.default_rng(0)
    b, t, d = 2, 6, 4
    x = rng.normal(0, 1, (b, t, d))
    wq, wk = (rng.normal(0, 0.5, (d, d)) for _ in range(2))
    raw = (x @ wq) @ (x @ wk).transpose(0, 2, 1) / np.sqrt(d)
    scores = np.where(causal_mask(t)[None], raw, sentinel)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    a = e / e.sum(axis=-1, keepdims=True)
    return float(np.abs(a[:, np.triu(np.ones((t, t), dtype=bool), k=1)]).max())


print(f"\n{'mask sentinel':>14s} {'largest surviving future weight':>33s}")
for _s in (-np.inf, -1e9, -1e3, -100.0, -50.0, -30.0):
    print(f"{_s:14.4g} {sentinel_leak(_s):33.3e}")
print(f"\nin float64 np.exp underflows to exactly 0 below about "
      f"{np.log(np.nextafter(0.0, 1.0)):.1f}, so any sentinel\npast that cliff is "
      "indistinguishable from -inf HERE — and a merely large negative one\nis not. -np.inf is "
      "the only choice that stays correct at any score scale and in any\nprecision; the "
      "cliff moves as soon as you leave float64.")

# %% [markdown]
# ## 6. Exercise 3 — `attention_forward(X, Wq, Wk, Wv)`
#
# Project, score, scale, mask, softmax, average. The scale is read from `x.shape[-1]`, and
# the mask is applied to the softmax's INPUT with `-np.inf` so the future weights are
# exactly zero rather than merely small.

# %%
def attention_forward(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray):
    """Causal single-head self-attention. Returns (context, cache)."""
    t, d = x.shape[1], x.shape[2]
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scale = 1.0 / np.sqrt(d)
    mask = causal_mask(t)
    scores = np.where(mask[None, :, :], (q @ k.transpose(0, 2, 1)) * scale, -np.inf)
    a = softmax(scores)
    context = a @ v
    cache = {"X": x, "Wq": wq, "Wk": wk, "Wv": wv,
             "Q": q, "K": k, "V": v, "A": a, "mask": mask, "scale": scale}
    return context, cache


ATTENTION_CACHE_KEYS = {"X", "Wq", "Wk", "Wv", "Q", "K", "V", "A", "mask", "scale"}


def _check_attention_forward() -> None:
    rng = np.random.default_rng(0)
    b, t, d = 2, 5, 4
    x = rng.normal(0, 1, (b, t, d))
    wq, wk, wv = (rng.normal(0, 0.5, (d, d)) for _ in range(3))
    ctx, cache = attention_forward(x, wq, wk, wv)

    assert ctx.shape == (b, t, d), f"context came back {ctx.shape}, expected {(b, t, d)}"
    assert isinstance(cache, dict), f"the cache must be a dict, got {type(cache).__name__}"
    missing = ATTENTION_CACHE_KEYS - set(cache)
    assert not missing, (
        f"the cache is missing {sorted(missing)} — exercise 5 needs every one of "
        f"{sorted(ATTENTION_CACHE_KEYS)}."
    )

    a = cache["A"]
    assert a.shape == (b, t, t), f"A came back {a.shape}, expected {(b, t, t)}"
    assert np.allclose(a.sum(axis=-1), 1.0), (
        f"attention rows sum to {a.sum(axis=-1).ravel()[:4]} instead of 1 — if some rows sum "
        "to less than 1 you masked AFTER the softmax instead of before it."
    )

    future = a[:, np.triu(np.ones((t, t), dtype=bool), k=1)]
    assert np.all(future == 0.0), (
        f"the largest strictly-upper-triangular attention weight is {future.max():.3e}, not "
        "exactly 0 — either you masked AFTER the softmax, or your mask sentinel is a finite "
        "negative number too small in magnitude for exp to underflow to zero. Use -np.inf."
    )

    assert abs(float(cache["scale"]) - 1.0 / np.sqrt(d)) < 1e-12, (
        f"scale was {cache['scale']}, expected 1/sqrt({d}) = {1 / np.sqrt(d):.6f} — read D "
        "from x.shape[-1]."
    )

    x2 = x.copy()
    x2[:, -1, :] += 100.0
    ctx2, _ = attention_forward(x2, wq, wk, wv)
    drift = float(np.abs(ctx[:, :-1] - ctx2[:, :-1]).max())
    assert drift == 0.0, (
        f"changing the LAST position moved earlier outputs by {drift:.3e} — the head is "
        "reading the future. Check the mask orientation: np.tril keeps the past, np.triu "
        "keeps the future, and transposing the scores swaps the two."
    )
    print(f"exercise 3 looks right: rows sum to 1, future weights exactly 0, perturbing the "
          f"last position moved earlier outputs by {drift:.1f}")


# %% [markdown]
# ## 7. The whole model, assembled

# %%
D_MODEL = 48       # width of the residual stream
CONTEXT = 32       # how many characters the model may look back over
BATCH = 32         # sequences per training step
INIT_SCALE = 0.02  # standard deviation of the initial weights


def init_params(seed: int = 1234, scale: float = INIT_SCALE) -> dict:
    """Fresh parameters. Every array is float64 and drawn from a seeded generator."""
    rng = np.random.default_rng(seed)
    return {
        "E": rng.normal(0, scale, (VOCAB_SIZE, D_MODEL)),
        "P": rng.normal(0, scale, (CONTEXT, D_MODEL)),
        "Wq": rng.normal(0, scale, (D_MODEL, D_MODEL)),
        "Wk": rng.normal(0, scale, (D_MODEL, D_MODEL)),
        "Wv": rng.normal(0, scale, (D_MODEL, D_MODEL)),
        "Wo": rng.normal(0, scale, (D_MODEL, D_MODEL)),
        "W1": rng.normal(0, scale, (D_MODEL, VOCAB_SIZE)),
        "b1": np.zeros(VOCAB_SIZE, dtype=DTYPE),
    }


def model_forward(params: dict, inputs: np.ndarray, targets=None):
    """Full forward pass. Returns (loss, cache); loss is None when targets is None."""
    t = inputs.shape[1]
    x = params["E"][inputs] + params["P"][None, :t, :]
    context, att_cache = attention_forward(x, params["Wq"], params["Wk"], params["Wv"])
    h = x + context @ params["Wo"]
    logits = h @ params["W1"] + params["b1"]
    loss = None if targets is None else cross_entropy(logits, targets)
    return loss, {"inputs": inputs, "targets": targets, "X": x, "context": context,
                  "att": att_cache, "H": h, "logits": logits}


def _check_model_forward() -> None:
    params = init_params()
    inputs = TRAIN_DATA[: CONTEXT * 2].reshape(2, CONTEXT)
    targets = TRAIN_DATA[1 : CONTEXT * 2 + 1].reshape(2, CONTEXT)
    loss, cache = model_forward(params, inputs, targets)
    n_params = sum(v.size for v in params.values())
    assert cache["logits"].shape == (2, CONTEXT, VOCAB_SIZE), cache["logits"].shape
    assert abs(loss - UNIFORM_BASELINE) < 0.15, (
        f"an untrained model scored {loss:.4f} but should start near the uniform baseline "
        f"{UNIFORM_BASELINE:.4f}. A much larger value means the initial weights are too big."
    )
    print(f"model has {n_params} parameters and starts at {loss:.4f} nats/char, against a "
          f"uniform baseline of {UNIFORM_BASELINE:.4f}")


# %% [markdown]
# ## 8. Exercise 4 — `cross_entropy_backward(logits, targets)`
#
# `(softmax(logits) - onehot(targets)) / (B*T)`. The `exp` and the `log` cancel, which is the
# entire reason softmax and cross-entropy are always paired.

# %%
def cross_entropy_backward(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Gradient of `cross_entropy(logits, targets)` with respect to `logits`.

    Example:
        >>> cross_entropy_backward(np.zeros((1, 1, 2)), np.array([[0]]))
        array([[[-0.5,  0.5]]])
    """
    probs = softmax(logits)
    one_hot = np.zeros_like(probs)
    np.put_along_axis(one_hot, targets[..., None], 1.0, axis=-1)
    return (probs - one_hot) / targets.size


def _check_cross_entropy_backward() -> None:
    g = cross_entropy_backward(np.zeros((1, 1, 2)), np.array([[0]]))
    assert g.shape == (1, 1, 2), f"expected shape (1,1,2), got {g.shape}"
    assert np.allclose(g, [[[-0.5, 0.5]]]), (
        f"got {g.ravel()}, expected [-0.5, 0.5] — subtract the one-hot from the softmax in "
        "that order, then divide by the number of positions."
    )

    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, (3, 4, VOCAB_SIZE))
    targets = rng.integers(0, VOCAB_SIZE, (3, 4))
    grad = cross_entropy_backward(logits, targets)
    rowsums = grad.sum(axis=-1)
    assert np.allclose(rowsums, 0.0, atol=1e-12), (
        f"rows of the gradient sum to {rowsums.ravel()[:4]} instead of 0 — that only happens "
        "if the one-hot was not subtracted, or was subtracted in the wrong place."
    )
    assert abs(float(np.abs(grad).sum()) - 2.0 * 3 * 4 / (3 * 4)) < 1.0, (
        "the gradient magnitude looks wrong by a large factor — did you divide by B*T?"
    )

    probe = (1, 2, 7)
    h = 1e-6
    up, down = logits.copy(), logits.copy()
    up[probe] += h
    down[probe] -= h
    numeric = (cross_entropy(up, targets) - cross_entropy(down, targets)) / (2 * h)
    analytic = float(grad[probe])
    assert abs(numeric - analytic) < 1e-7, (
        f"finite differences say {numeric:.9f} at logit {probe}, your formula says "
        f"{analytic:.9f}. A ratio of about {analytic / numeric:.1f} between them is the "
        "tell-tale of a missing or extra division by B*T."
    )
    print(f"exercise 4 looks right: rows sum to 0, and at one probed logit finite differences "
          f"give {numeric:.9f} against your {analytic:.9f}")


# %% [markdown]
# ## 9. Exercise 5 — `attention_backward(d_context, cache)`
#
# Every forward line reversed. The two things that matter: the softmax Jacobian couples the
# row, so it is `A * (dA - (dA*A).sum(-1, keepdims=True))` and not `A * dA`; and `dX`
# accumulates the query, key and value paths rather than being assigned once.

# %%
def attention_backward(d_context: np.ndarray, cache: dict):
    """Backward pass of `attention_forward`. Returns (dX, dWq, dWk, dWv)."""
    x, wq, wk, wv = cache["X"], cache["Wq"], cache["Wk"], cache["Wv"]
    q, k, v, a = cache["Q"], cache["K"], cache["V"], cache["A"]
    mask, scale = cache["mask"], cache["scale"]
    d_model = x.shape[-1]

    # context = A @ V
    d_a = d_context @ v.transpose(0, 2, 1)
    d_v = a.transpose(0, 2, 1) @ d_context

    # A = softmax(scores): the rows are coupled, so subtract the row's own weighted sum.
    d_scores = a * (d_a - (d_a * a).sum(axis=-1, keepdims=True))

    # the mask contributed nothing, so no gradient flows through the masked entries
    d_scores = np.where(mask[None, :, :], d_scores, 0.0) * scale

    # scores = (Q @ K^T) * scale
    d_q = d_scores @ k
    d_k = d_scores.transpose(0, 2, 1) @ q

    # Q = X @ Wq, K = X @ Wk, V = X @ Wv — a weight is shared over batch and time
    x_flat = x.reshape(-1, d_model)
    d_wq = x_flat.T @ d_q.reshape(-1, d_model)
    d_wk = x_flat.T @ d_k.reshape(-1, d_model)
    d_wv = x_flat.T @ d_v.reshape(-1, d_model)

    # X fed all three projections: three gradients, and they add
    d_x = d_q @ wq.T + d_k @ wk.T + d_v @ wv.T
    return d_x, d_wq, d_wk, d_wv


def _check_attention_backward() -> None:
    rng = np.random.default_rng(2)
    b, t, d = 2, 4, 3
    x = rng.normal(0, 0.6, (b, t, d))
    wq, wk, wv = (rng.normal(0, 0.6, (d, d)) for _ in range(3))
    weight = rng.normal(0, 1, (b, t, d))

    ctx, cache = attention_forward(x, wq, wk, wv)
    dx, dwq, dwk, dwv = attention_backward(weight, cache)

    for name, got, want in (("dX", dx, x.shape), ("dWq", dwq, wq.shape),
                            ("dWk", dwk, wk.shape), ("dWv", dwv, wv.shape)):
        assert got.shape == want, f"{name} came back {got.shape}, expected {want}"

    def objective(x_, wq_, wk_, wv_) -> float:
        c, _ = attention_forward(x_, wq_, wk_, wv_)
        return float((c * weight).sum())

    h = 1e-6
    for name, arr, analytic in (("X", x, dx), ("Wq", wq, dwq), ("Wk", wk, dwk),
                                ("Wv", wv, dwv)):
        flat = arr.reshape(-1)
        worst = 0.0
        for q in range(0, flat.size, max(1, flat.size // 4)):
            old = flat[q]
            flat[q] = old + h
            up = objective(x, wq, wk, wv)
            flat[q] = old - h
            down = objective(x, wq, wk, wv)
            flat[q] = old
            numeric = (up - down) / (2 * h)
            worst = max(worst, abs(numeric - float(analytic.reshape(-1)[q])))
        assert worst < 1e-5, (
            f"d{name} disagrees with finite differences by {worst:.3e}. If dWq and dWk are "
            "the ones failing while dWv is fine, the softmax Jacobian is the suspect: it is "
            "A * (dA - (dA*A).sum(-1, keepdims=True)), not A * dA. If dX alone is wrong, you "
            "assigned instead of accumulating its three contributions."
        )
    print("exercise 5 looks right: dX, dWq, dWk and dWv all match central finite differences")


# %% [markdown]
# ## 10. The full gradient check

# %%
FD_STEP = 1e-5
FD_ATOL = 1e-9
FD_RTOL = 1e-5
GRADCHECK_SCALE = 0.2  # see section 11 — deliberately NOT the training init scale


def model_backward(params: dict, cache: dict) -> dict:
    """Gradients of the loss with respect to every parameter. Returns a dict keyed like params."""
    d_model = cache["H"].shape[-1]
    d_logits = cross_entropy_backward(cache["logits"], cache["targets"])
    grads = {
        "W1": cache["H"].reshape(-1, d_model).T @ d_logits.reshape(-1, VOCAB_SIZE),
        "b1": d_logits.sum(axis=(0, 1)),
    }
    d_h = d_logits @ params["W1"].T
    grads["Wo"] = cache["context"].reshape(-1, d_model).T @ d_h.reshape(-1, d_model)
    d_context = d_h @ params["Wo"].T

    d_x_attn, grads["Wq"], grads["Wk"], grads["Wv"] = attention_backward(d_context, cache["att"])
    d_x = d_h + d_x_attn  # d_h is the residual path: H = X + context @ Wo

    grads["P"] = d_x.sum(axis=0)
    d_e = np.zeros_like(params["E"])
    np.add.at(d_e, cache["inputs"], d_x)  # accumulate: a character can occur many times
    grads["E"] = d_e
    return grads


def finite_difference_gradient(params, name, indices, inputs, targets, h=FD_STEP):
    """Central-difference derivative of the loss at the flat `indices` of params[name]."""
    flat = params[name].reshape(-1)
    out = []
    for q in indices:
        old = flat[q]
        flat[q] = old + h
        up, _ = model_forward(params, inputs, targets)
        flat[q] = old - h
        down, _ = model_forward(params, inputs, targets)
        flat[q] = old
        out.append((up - down) / (2 * h))
    return np.array(out)


def gradient_report(scale=GRADCHECK_SCALE, probes=6, seed=900, verbose=True):
    """Check every parameter's analytic gradient against finite differences. Returns rows."""
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed, scale=scale)
    params["b1"] = rng.normal(0, scale * 0.1, VOCAB_SIZE)  # a zero bias is a degenerate probe
    starts = rng.integers(0, len(TRAIN_DATA) - CONTEXT - 1, size=4)
    inputs = np.stack([TRAIN_DATA[s : s + CONTEXT] for s in starts])
    targets = np.stack([TRAIN_DATA[s + 1 : s + 1 + CONTEXT] for s in starts])

    _, cache = model_forward(params, inputs, targets)
    grads = model_backward(params, cache)

    rows = []
    for name in ("E", "P", "Wq", "Wk", "Wv", "Wo", "W1", "b1"):
        idx = rng.integers(0, params[name].size, size=probes)
        numeric = finite_difference_gradient(params, name, idx, inputs, targets)
        analytic = grads[name].reshape(-1)[idx]
        err = np.abs(numeric - analytic)
        tol = FD_ATOL + FD_RTOL * np.maximum(np.abs(numeric), np.abs(analytic))
        rows.append((name, float(err.max()), float((err / tol).max()),
                     float(np.sqrt((grads[name] ** 2).mean()))))
    if verbose:
        print(f"{'param':6s} {'grad RMS':>10s} {'max |num-ana|':>14s} {'fraction of tol':>16s}")
        for name, abs_err, slack, rms in rows:
            print(f"{name:6s} {rms:10.3e} {abs_err:14.3e} {slack:16.3f}"
                  f"{'  OK' if slack <= 1.0 else '  FAIL'}")
    return rows


def _check_gradients() -> None:
    rows = gradient_report()
    bad = [(n, s) for n, _, s, _ in rows if s > 1.0]
    assert not bad, (
        f"these parameters failed the finite-difference check: {bad}. The number shown is the "
        "error as a fraction of tolerance, so anything above 1.0 is a real disagreement."
    )
    worst = max(s for _, _, s, _ in rows)
    print(f"\nall 8 parameter gradients agree with central finite differences; the worst was "
          f"{worst:.3f} of the allowed tolerance")


# %% [markdown]
# ## 11. Why the probe point decides whether your check means anything

# %%
def probe_scale_comparison(scales=(INIT_SCALE, GRADCHECK_SCALE)) -> None:
    """Print attention-gradient magnitude at each init scale. Pure measurement, no claims."""
    print(f"{'init scale':>11s} {'|dWq| RMS':>12s} {'|dWv| RMS':>12s} {'last-row max':>14s}")
    for scale in scales:
        params = init_params(seed=900, scale=scale)
        starts = np.random.default_rng(3).integers(0, len(TRAIN_DATA) - CONTEXT - 1, size=4)
        inputs = np.stack([TRAIN_DATA[s : s + CONTEXT] for s in starts])
        targets = np.stack([TRAIN_DATA[s + 1 : s + 1 + CONTEXT] for s in starts])
        _, cache = model_forward(params, inputs, targets)
        grads = model_backward(params, cache)
        print(f"{scale:11.3f} {np.sqrt((grads['Wq'] ** 2).mean()):12.3e} "
              f"{np.sqrt((grads['Wv'] ** 2).mean()):12.3e} "
              f"{float(cache['att']['A'][:, -1, :].max()):14.4f}")
    print(f"\nthe last column is the largest weight in the FINAL attention row, which may "
          f"look back\nover all {CONTEXT} positions; perfectly uniform attention would put "
          f"{1.0 / CONTEXT:.4f} there.\nAt the smaller scale that row is essentially uniform "
          "and |dWq| is orders of magnitude\nsmaller than every other gradient, so finite "
          "differences have no signal to find and a\nbroken query/key gradient passes the "
          "check by accident.")


def _show_probe_tables() -> None:
    """Both tables, printed once they are complete, so an unfinished exercise skips cleanly
    instead of leaving a table header with nothing under it."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        probe_scale_comparison()
        print()
        step_size_sweep()
    print(buffer.getvalue(), end="")


def step_size_sweep() -> None:
    """Show the U-curve: truncation error at large h, round-off error at small h."""
    params = init_params(seed=900, scale=GRADCHECK_SCALE)
    starts = np.random.default_rng(4).integers(0, len(TRAIN_DATA) - CONTEXT - 1, size=2)
    inputs = np.stack([TRAIN_DATA[s : s + CONTEXT] for s in starts])
    targets = np.stack([TRAIN_DATA[s + 1 : s + 1 + CONTEXT] for s in starts])
    _, cache = model_forward(params, inputs, targets)
    analytic = model_backward(params, cache)["Wv"].reshape(-1)
    idx = np.random.default_rng(5).integers(0, params["Wv"].size, size=5)
    print(f"{'h':>10s} {'max relative error in dWv':>28s}")
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10):
        numeric = finite_difference_gradient(params, "Wv", idx, inputs, targets, h=h)
        rel = np.abs(numeric - analytic[idx]) / np.maximum(np.abs(analytic[idx]), 1e-30)
        print(f"{h:10.0e} {float(rel.max()):28.3e}")
    print("\nboth ends are bad: large h measures a chord instead of a tangent, small h\n"
          "subtracts two nearly equal numbers and keeps only the round-off")


# %% [markdown]
# ## 12. Exercise 6 — `make_batch(data, batch_size, context_len, rng)`
#
# The target window starts one character later than the input window. The exclusive upper
# bound `len(data) - context_len` guarantees that final target character exists.

# %%
def make_batch(data: np.ndarray, batch_size: int, context_len: int, rng):
    """Cut `batch_size` random (input, target) windows out of `data`.

    Example:
        >>> toy = np.arange(10)
        >>> xb, yb = make_batch(toy, 2, 4, np.random.default_rng(0))
        >>> bool((yb[:, :-1] == xb[:, 1:]).all())
        True
    """
    starts = rng.integers(0, len(data) - context_len, size=batch_size)
    inputs = np.stack([data[s : s + context_len] for s in starts])
    targets = np.stack([data[s + 1 : s + 1 + context_len] for s in starts])
    return inputs, targets


def _check_make_batch() -> None:
    toy = np.arange(20)
    xb, yb = make_batch(toy, 3, 5, np.random.default_rng(0))
    assert xb.shape == (3, 5) and yb.shape == (3, 5), (
        f"expected two (3,5) arrays, got {xb.shape} and {yb.shape}"
    )
    assert (yb == xb + 1).all(), (
        f"inputs start {xb[0]} and targets start {yb[0]} — on this toy corpus each target "
        "must be its input plus one. If they are identical you sliced the same window twice; "
        "the target window starts one character later."
    )

    xb2, yb2 = make_batch(TRAIN_DATA, BATCH, CONTEXT, np.random.default_rng(1))
    assert xb2.dtype == np.int64, f"inputs came back {xb2.dtype}; they index an array, so int64"
    assert xb2.max() < VOCAB_SIZE and yb2.max() < VOCAB_SIZE, (
        "a character id landed outside the vocabulary — you indexed past the end of the corpus"
    )
    assert (yb2[:, :-1] == xb2[:, 1:]).all(), (
        "on the real corpus the target window is not the input window shifted by one"
    )
    assert len({tuple(row) for row in xb2}) > 1, (
        "every row of the batch is identical — draw one start position PER row, by passing "
        "size=batch_size to rng.integers."
    )
    print(f"exercise 6 looks right: {xb2.shape} inputs, targets shifted by exactly one")


# %% [markdown]
# ## 13. Train it

# %%
TRAIN_STEPS = 3000
LEARNING_RATE = 3e-3
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
MIN_MARGIN_NATS = 0.35  # how far below the unigram entropy held-out loss must land


def adam_init(params: dict) -> dict:
    """First and second moment accumulators, one pair per parameter."""
    return {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in params.items()}


def adam_step(params, grads, state, step, lr=LEARNING_RATE):
    """One Adam update, in place, with the bias correction applied to both moments."""
    for k, p in params.items():
        m, v = state[k]
        m *= ADAM_BETA1
        m += (1 - ADAM_BETA1) * grads[k]
        v *= ADAM_BETA2
        v += (1 - ADAM_BETA2) * grads[k] ** 2
        m_hat = m / (1 - ADAM_BETA1 ** step)
        v_hat = v / (1 - ADAM_BETA2 ** step)
        p -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


def evaluate(params, data, batches=30, seed=5) -> float:
    """Mean loss over `batches` freshly drawn windows. Used for held-out measurement."""
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        inputs, targets = make_batch(data, BATCH, CONTEXT, rng)
        loss, _ = model_forward(params, inputs, targets)
        losses.append(loss)
    return float(np.mean(losses))


def train(steps=TRAIN_STEPS, seed=0, report_every=500):
    """Train from a fresh init. Returns (params, history) with history in (step, loss) pairs."""
    params = init_params()
    state = adam_init(params)
    rng = np.random.default_rng(seed)
    history, started = [], time.perf_counter()
    for step in range(1, steps + 1):
        inputs, targets = make_batch(TRAIN_DATA, BATCH, CONTEXT, rng)
        loss, cache = model_forward(params, inputs, targets)
        adam_step(params, model_backward(params, cache), state, step)
        if step % report_every == 0 or step == 1:
            history.append((step, loss))
            bar = "#" * int(40 * loss / UNIFORM_BASELINE)
            print(f"  step {step:5d}  loss {loss:6.4f}  {bar}")
    print(f"  trained {steps} steps in {time.perf_counter() - started:.1f} s")
    return params, history


def _run_training() -> None:
    """Train, and keep the model for the sample and the recap — only if training succeeded."""
    global trained
    trained = _check_training()


def _check_training():
    print("training (no GPU, no autodiff, one attention head):")
    params, history = train()
    train_loss = evaluate(params, TRAIN_DATA, seed=6)
    val_loss = evaluate(params, VAL_DATA, seed=7)
    margin = UNIGRAM_BASELINE - val_loss

    print(f"\n  uniform baseline        {UNIFORM_BASELINE:.4f} nats/char")
    print(f"  unigram entropy         {UNIGRAM_BASELINE:.4f} nats/char  <- the bar")
    print(f"  final training loss     {train_loss:.4f} nats/char")
    print(f"  final held-out loss     {val_loss:.4f} nats/char")
    print(f"  margin over unigram     {margin:+.4f} nats  (must be >= {MIN_MARGIN_NATS})")
    print(f"  overfitting gap         {val_loss - train_loss:+.4f} nats")

    assert history[-1][1] < history[0][1], (
        f"loss went from {history[0][1]:.4f} to {history[-1][1]:.4f} — it must fall. If it "
        "rose, a gradient sign is flipped; the finite-difference check in section 10 finds it."
    )
    assert margin >= MIN_MARGIN_NATS, (
        f"held-out loss {val_loss:.4f} beat the unigram entropy {UNIGRAM_BASELINE:.4f} by only "
        f"{margin:.4f} nats, short of the {MIN_MARGIN_NATS} required. The model is not using "
        "context: check the off-by-one in make_batch, and that the causal mask is not hiding "
        "every position including the present one."
    )
    return params


# %% [markdown]
# ## 14. What it writes

# %%
def sample(params, n_chars=400, seed=3, temperature=1.0, prompt=None) -> str:
    """Generate `n_chars` characters, one at a time, feeding each back in as context."""
    rng = np.random.default_rng(seed)
    context = list(encode(prompt if prompt is not None else CORPUS[:CONTEXT]))
    out = []
    for _ in range(n_chars):
        window = np.array(context[-CONTEXT:], dtype=np.int64)[None, :]
        _, cache = model_forward(params, window)
        probs = softmax(cache["logits"][0, -1] / temperature)
        nxt = int(rng.choice(VOCAB_SIZE, p=probs))
        context.append(nxt)
        out.append(nxt)
    return decode(out)


# %% [markdown]
# ## 15. Common mistakes, one of them reproduced and measured

# %%
def _demo_wrong_jacobian() -> None:
    """Break the softmax Jacobian on purpose, and measure how far wrong it goes."""
    rng = np.random.default_rng(7)
    b, t, d = 2, 5, 4
    x = rng.normal(0, 0.8, (b, t, d))
    wq, wk, wv = (rng.normal(0, 0.8, (d, d)) for _ in range(3))
    w = rng.normal(0, 1, (b, t, d))
    _, cache = attention_forward(x, wq, wk, wv)
    good = attention_backward(w, cache)[1]  # the correct dWq

    d_a = w @ cache["V"].transpose(0, 2, 1)  # and now the mistake, one line changed
    d_scores = np.where(cache["mask"][None], cache["A"] * d_a, 0.0) * cache["scale"]
    bad = x.reshape(-1, d).T @ (d_scores @ cache["K"]).reshape(-1, d)

    flat, h, numeric = wq.reshape(-1), 1e-6, []
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + h
        up, _ = attention_forward(x, wq, wk, wv)
        flat[i] = old - h
        down, _ = attention_forward(x, wq, wk, wv)
        flat[i] = old
        numeric.append((float((up * w).sum()) - float((down * w).sum())) / (2 * h))
    numeric = np.array(numeric)
    print(f"  correct dWq    max |analytic - finite difference|  "
          f"{np.abs(good.reshape(-1) - numeric).max():.3e}")
    print(f"  `A * dA` dWq   max |analytic - finite difference|  "
          f"{np.abs(bad.reshape(-1) - numeric).max():.3e}")
    print("\nthe second is the error the list above warns about. It is glaring once finite\n"
          "differences are pointed at it, and completely invisible without them.")


# %% [markdown]
# ## 16. Self-check answers
#
# See the comment block at the foot of this file.

# %%
def _show_heldout() -> None:
    print(f"  held-out after training  {evaluate(trained, VAL_DATA, seed=7):.4f} nats/char")


# The progress board. The six exercises it lists are the ones `_EXERCISES` names in the setup
# cell.
_MARKS = {"passed": "✅ passed     ", "failed": "❌ failed     ", "not started": "⏳ not started"}


def _progress_board() -> None:
    """One line per exercise, from what the checks recorded — then how many are complete."""
    print("\nyour progress")
    done, next_up = 0, None
    for label, function, where in _EXERCISES:
        status = _STATUS.get(label, "not started")
        done += status == "passed"
        waiting = _WAITING_ON.get(label)
        note = (f"   (waiting on `{waiting}`)"
                if status == "not started" and waiting not in (None, "", function) else "")
        print(f"  {_MARKS[status]}  {label} · {function}{note}")
        if status != "passed" and next_up is None:
            next_up = (label, function, where)
    print(f"\n{done} of {len(_EXERCISES)} exercises complete")
    if next_up:
        print(f"next: {next_up[0]} — `{next_up[1]}`, {next_up[2]}. Its cell has hints if you "
              "are stuck.")


def _finish() -> None:
    """The board, then the verdict. A stub you have not reached yet is not a failure; a check
    that ran and came back wrong is. In a script or under CI that ends the run non-zero, so a
    green exit code can never paper over it. In a notebook kernel it is a printed line."""
    _progress_board()
    failed = [label for label, status in _STATUS.items() if status == "failed"]
    if failed and "ipykernel" not in sys.modules:
        raise SystemExit("checks failed: " + ", ".join(failed))
    if failed:
        print("checks failed: " + ", ".join(failed) + " — each one printed its likely mistake "
              "where it ran.")


if __name__ == "__main__":
    _try("exercise 1", _check_softmax)
    _try("exercise 2", _check_cross_entropy)
    _try("exercise 3", _check_attention_forward)
    _try("the assembled model", _check_model_forward,
         needs=("exercise 1", "exercise 2", "exercise 3"))
    _try("exercise 4", _check_cross_entropy_backward)
    _try("exercise 5", _check_attention_backward)

    print("\n--- section 10: full gradient check ---")
    _try("the full gradient check", _check_gradients, needs=_FOR_GRADIENTS)

    print("\n--- section 11: why the probe point matters ---")
    _try("the probe-point tables", _show_probe_tables, needs=_FOR_GRADIENTS)

    print("\n--- section 12/13: batching and training ---")
    _try("exercise 6", _check_make_batch)
    print()
    trained = None
    _try("training", _run_training, needs=tuple(_FUNCTION))

    print("\n--- section 14: a sample ---")
    if trained is None:
        print("the sample: skipped — it needs the model section 13 trains, and that needs all "
              "six exercises.")
    else:
        _try("the sample", lambda: print(sample(trained)))

    print("\n--- section 15: watch the `A * dA` mistake get caught ---")
    _try("the `A * dA` demonstration", _demo_wrong_jacobian,
         needs=("exercise 1", "exercise 3", "exercise 5"))

    print("\n--- what you built, measured ---")
    print(f"  parameters               {sum(v.size for v in init_params().values())}")
    print(f"  uniform baseline         {UNIFORM_BASELINE:.4f} nats/char")
    print(f"  unigram entropy (train)  {UNIGRAM_BASELINE:.4f} nats/char")
    if trained is None:
        print("  held-out after training  not measured yet — section 13 needs all six exercises")
    else:
        _try("the recap", _show_heldout)
    print("  every number above was computed by the code you just ran")
    _finish()

# %% [markdown]
# ## Self-check answers
#
# **1 — (a) nothing at all; it is exactly the same distribution, just computable.** Softmax is
# shift-invariant: multiplying numerator and denominator by `exp(-m)` cancels, so
# `exp(x-m)/sum(exp(x-m))` and `exp(x)/sum(exp(x))` are the same function of `x`. What changes
# is only the arithmetic: after the shift the largest exponent is `exp(0) == 1`, so nothing can
# overflow. (b) and (c) both imagine the shift alters the distribution's shape — it cannot,
# because a constant added to every score is divided straight back out. (d) is closer but still
# wrong in kind: the identity is exact algebra, not an approximation that happens to fall below
# float64 resolution. The `_check_softmax` assertion comparing a row against the same row plus
# 500 is exactly this property, measured.
#
# **2 — (b) the rows no longer sum to 1.** This is the one that bites hardest because the code
# looks right and the model still trains, just worse. Softmax normalises over whatever it is
# given; if the future is included in that normalisation, the future's probability mass is
# already baked into the denominator. Zeroing those entries afterwards removes the numerators
# but not the denominator they contributed to, so row `t` ends up summing to some value below 1
# that depends on how interesting the future looked — and every position gets silently scaled
# by a different, data-dependent factor. Masking before the softmax makes `exp(-inf) == 0`, so
# the surviving entries normalise among themselves. (d) is the tempting wrong answer: no
# information actually leaks, because the zeroed weights genuinely are zero; the damage is the
# broken normalisation, not leakage. Section 5 prints both rows so the failure is visible
# rather than asserted.
#
# **3 — (c) at that scale the true `dWq` is vanishingly small.** This is the deepest point in
# the lesson. At the training initialisation the attention scores are far too small to move the
# softmax, so every row is indistinguishable from uniform. A uniform attention row is a *stationary* point of the
# softmax with respect to the scores: moving `Wq` a little barely moves `A` at all, and the
# effect on the loss is second-order small. The analytic gradient is therefore genuinely
# minuscule, and the finite-difference estimate — a difference of two losses that agree to
# fourteen digits — is pure round-off. Both are approximately zero, so they agree, and a `dWq`
# that returns literal zeros passes. (b) is exactly backwards: a *tighter* tolerance would not
# help, because the numerical estimate carries no signal to compare against at any tolerance.
# (a) is false, and (d) confuses a correctness check with an optimiser. The fix is the one this
# lesson uses: probe at a scale where the attention is actually doing something, which
# `probe_scale_comparison` measures side by side.
#
# **4 — (b) dot products grow with `D`, and large scores flatten the softmax's gradient.** The
# sum defining a dot product has `D` terms, so its magnitude grows like `sqrt(D)` for random
# vectors; dividing by `sqrt(D)` holds the score variance roughly constant as the model widens.
# Without it, scores at realistic widths are large enough to saturate the softmax into a near
# one-hot, and a saturated softmax has almost no gradient — which is the reason Vaswani et al.
# give, quoted in `claims.yaml`. (a) is wrong because softmax normalises whatever scale it is
# handed; the rows sum to 1 either way. (c) confuses the scaling with the sequence length,
# which never enters. (d) is the belief this question exists to kill: it is not cosmetic, and
# removing it at width 48 is already measurable.
#
# **5 — (b) `make_batch` returned targets equal to the inputs.** If the target at position `t`
# is the character *at* `t` rather than at `t+1`, the model is being asked to reproduce
# something it can already see through the residual path, so the loss collapses towards zero
# almost immediately — far below the unigram entropy, which no honest next-character model on
# this corpus reaches. The samples stay gibberish because nothing about the actual prediction
# task was ever learned. (a) produces the opposite symptom: a high learning rate makes the loss
# spike or oscillate, not fall to zero. (c) would show as a gap between training and held-out
# loss with both still above zero, which is ordinary overfitting and visible in this lesson's
# own output. (d) would shift the baseline slightly and change nothing about the training
# curve. The `_check_make_batch` assertion `(yb == xb + 1).all()` on a toy `arange` corpus is
# designed so this failure is impossible to miss.
