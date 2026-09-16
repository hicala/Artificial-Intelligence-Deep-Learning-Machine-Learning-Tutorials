"""Autograder rubric for T03-L03. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These cases deliberately differ from the notebook's public `_check_*` functions: different
shapes, different seeds, different probe points, and structural probes that a plausible-looking
wrong implementation fails. In particular:

  * softmax is probed with -inf entries and with a 1e4 offset, not just a uniform row;
  * the causality test perturbs a MIDDLE position, not the last one, so an off-by-one in the
    mask is caught as well as a transposed one;
  * the backward tests build their cache from this file's OWN reference forward pass, so a
    student whose attention_forward is broken can still earn the backward marks, and a student
    whose backward merely agrees with their own broken forward cannot.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


# --- reference implementations, used only to generate expected values -----------------------

def _ref_softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _ref_cross_entropy(logits, targets):
    m = logits.max(axis=-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(logits - m).sum(axis=-1))
    chosen = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return float((lse - chosen).mean())


def _ref_attention(x, wq, wk, wv):
    t, d = x.shape[1], x.shape[2]
    q, k, v = x @ wq, x @ wk, x @ wv
    scale = 1.0 / np.sqrt(d)
    mask = np.tril(np.ones((t, t), dtype=bool))
    scores = np.where(mask[None], (q @ k.transpose(0, 2, 1)) * scale, -np.inf)
    a = _ref_softmax(scores)
    return a @ v, {"X": x, "Wq": wq, "Wk": wk, "Wv": wv, "Q": q, "K": k, "V": v,
                   "A": a, "mask": mask, "scale": scale}


def _rand_case(seed, b=3, t=6, d=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.7, (b, t, d))
    wq, wk, wv = (rng.normal(0, 0.7, (d, d)) for _ in range(3))
    return x, wq, wk, wv, rng


# --- exercise 1: softmax --------------------------------------------------------------------

def test_softmax_matches_reference():
    """Correct values on ranks 1, 2 and 3, including a row containing -inf."""
    rng = np.random.default_rng(11)
    for shape in ((7,), (4, 9), (2, 5, 6)):
        x = rng.normal(0, 3, shape)
        got = lesson.softmax(x)
        assert got.shape == x.shape, f"shape {shape}: returned {got.shape}"
        assert np.allclose(got, _ref_softmax(x), atol=1e-12), (
            f"values differ from the reference on shape {shape}"
        )
        assert np.allclose(got.sum(axis=-1), 1.0, atol=1e-12), (
            f"rows do not sum to 1 on shape {shape} — normalise the LAST axis with keepdims"
        )
    masked = lesson.softmax(np.array([[0.0, -np.inf, 2.0, -np.inf]]))
    assert masked[0, 1] == 0.0 and masked[0, 3] == 0.0, (
        f"-inf entries gave {masked[0, 1]} and {masked[0, 3]}, both must be exactly 0.0"
    )
    assert abs(float(masked.sum()) - 1.0) < 1e-12, "a partly-masked row must still sum to 1"


def test_softmax_survives_huge_logits():
    """1e4-scale logits must stay finite, and shifting a row must not change it."""
    big = lesson.softmax(np.array([1e4, 1e4 - 1.0, 1e4 - 2.0]))
    assert np.isfinite(big).all(), "overflowed to nan/inf — subtract the row max before np.exp"
    assert abs(float(big.sum()) - 1.0) < 1e-12, f"sums to {float(big.sum())}"
    assert np.allclose(big, _ref_softmax(np.array([0.0, -1.0, -2.0])), atol=1e-12), (
        "the shifted values are wrong, so the max was subtracted along the wrong axis"
    )
    tiny = lesson.softmax(np.array([-1e4, -1e4 - 1.0]))
    assert np.isfinite(tiny).all() and abs(float(tiny.sum()) - 1.0) < 1e-12, (
        "large NEGATIVE logits underflowed; subtracting the max fixes this direction too"
    )


# --- exercise 2: cross_entropy --------------------------------------------------------------

def test_cross_entropy_matches_reference():
    """Correct value on random logits, and averaged over B*T rather than summed."""
    rng = np.random.default_rng(12)
    for b, t, v in ((2, 3, 7), (4, 5, 11), (1, 1, 3)):
        logits = rng.normal(0, 2.5, (b, t, v))
        targets = rng.integers(0, v, (b, t))
        got = lesson.cross_entropy(logits, targets)
        want = _ref_cross_entropy(logits, targets)
        assert isinstance(got, float), f"returned {type(got).__name__}, expected float"
        assert abs(got - want) < 1e-10, (
            f"got {got:.9f}, expected {want:.9f} on a ({b},{t},{v}) batch; a ratio of "
            f"{got / want if want else float('nan'):.2f} suggests summing instead of averaging"
        )
    v = 6
    doubled = np.zeros((8, 4, v))
    single = np.zeros((1, 1, v))
    assert abs(lesson.cross_entropy(doubled, np.zeros((8, 4), dtype=np.int64))
               - lesson.cross_entropy(single, np.zeros((1, 1), dtype=np.int64))) < 1e-12, (
        "the loss changed when the batch grew — average over every one of the B*T positions"
    )


def test_cross_entropy_survives_huge_logits():
    """Must not overflow, and must still be exact, at logit magnitude 1e4."""
    v = 5
    flat = lesson.cross_entropy(np.full((2, 3, v), 1e4), np.zeros((2, 3), dtype=np.int64))
    assert np.isfinite(flat), "nan/inf at 1e4 logits — use log-sum-exp with the max pulled out"
    assert abs(flat - np.log(v)) < 1e-9, (
        f"equal logits over {v} symbols scored {flat:.9f}, but that is still uniform, so the "
        f"answer is ln({v}) = {np.log(v):.9f}"
    )
    rng = np.random.default_rng(13)
    logits = rng.normal(0, 2, (2, 3, v))
    targets = rng.integers(0, v, (2, 3))
    base = lesson.cross_entropy(logits, targets)
    offset = lesson.cross_entropy(logits + 1e4, targets)
    assert abs(base - offset) < 1e-8, (
        f"adding 1e4 to every logit changed the loss from {base:.9f} to {offset:.9f}; "
        "cross-entropy is shift-invariant, so this is a stability bug"
    )


# --- exercise 3: attention_forward ----------------------------------------------------------

def test_attention_cache_contract():
    """The cache must carry every documented key, with the right shapes and scale."""
    x, wq, wk, wv, _ = _rand_case(21)
    _, cache = lesson.attention_forward(x, wq, wk, wv)
    assert isinstance(cache, dict), f"cache is {type(cache).__name__}, expected dict"
    required = {"X", "Wq", "Wk", "Wv", "Q", "K", "V", "A", "mask", "scale"}
    missing = required - set(cache)
    assert not missing, f"cache is missing {sorted(missing)}"
    b, t, d = x.shape
    for key in ("Q", "K", "V"):
        assert np.asarray(cache[key]).shape == (b, t, d), (
            f"cache[{key!r}] is {np.asarray(cache[key]).shape}, expected {(b, t, d)}"
        )
    assert np.asarray(cache["A"]).shape == (b, t, t), "cache['A'] must be (B, T, T)"
    assert abs(float(cache["scale"]) - 1.0 / np.sqrt(d)) < 1e-12, (
        f"scale {cache['scale']} should be 1/sqrt({d}); read D from x.shape[-1]"
    )


def test_attention_forward_matches_reference():
    """Context and attention weights must equal the reference on several random cases."""
    for seed in (22, 23, 24):
        x, wq, wk, wv, _ = _rand_case(seed)
        ctx, cache = lesson.attention_forward(x, wq, wk, wv)
        want_ctx, want_cache = _ref_attention(x, wq, wk, wv)
        assert ctx.shape == want_ctx.shape, f"context {ctx.shape}, expected {want_ctx.shape}"
        assert np.allclose(cache["A"], want_cache["A"], atol=1e-11), (
            "attention weights differ from the reference; check the 1/sqrt(D) scale and that "
            "the mask is applied to the softmax INPUT"
        )
        assert np.allclose(ctx, want_ctx, atol=1e-11), (
            "context differs from the reference even though A may be close; check context = A @ V"
        )


def test_attention_cannot_see_the_future():
    """Perturbing a MIDDLE position must leave every earlier position bit-identical."""
    x, wq, wk, wv, _ = _rand_case(25, b=2, t=7, d=4)
    ctx, cache = lesson.attention_forward(x, wq, wk, wv)
    a = np.asarray(cache["A"])
    t = x.shape[1]
    future = a[:, np.triu(np.ones((t, t), dtype=bool), k=1)]
    assert np.all(future == 0.0), (
        f"largest future attention weight is {float(np.abs(future).max()):.3e}, not exactly "
        "0.0 — either the mask was applied after the softmax, or its sentinel is a finite "
        "negative number too small in magnitude for exp to underflow to zero. Use -np.inf."
    )
    assert np.allclose(a.sum(axis=-1), 1.0, atol=1e-12), (
        "attention rows do not sum to 1 — masking happened after the softmax"
    )
    cut = 3
    x2 = x.copy()
    x2[:, cut, :] += 50.0
    ctx2, _ = lesson.attention_forward(x2, wq, wk, wv)
    drift = float(np.abs(ctx[:, :cut] - ctx2[:, :cut]).max())
    assert drift == 0.0, (
        f"changing position {cut} moved positions 0..{cut - 1} by {drift:.3e}; the head is "
        "reading the future, so the mask is transposed or off by one"
    )
    assert float(np.abs(ctx[:, cut:] - ctx2[:, cut:]).max()) > 0.0, (
        f"changing position {cut} left positions {cut}.. unchanged too, so the head is "
        "ignoring its input entirely rather than merely masking correctly"
    )


# --- exercise 4: cross_entropy_backward -----------------------------------------------------

def test_cross_entropy_backward_matches_finite_differences():
    """Against central differences of the REFERENCE loss, so it is independent of exercise 2."""
    rng = np.random.default_rng(31)
    b, t, v = 2, 4, 6
    logits = rng.normal(0, 1.5, (b, t, v))
    targets = rng.integers(0, v, (b, t))
    grad = lesson.cross_entropy_backward(logits, targets)
    assert grad.shape == logits.shape, f"returned {grad.shape}, expected {logits.shape}"
    assert np.allclose(grad.sum(axis=-1), 0.0, atol=1e-12), (
        f"rows sum to {grad.sum(axis=-1).ravel()[:3]}, not 0 — the one-hot was not subtracted"
    )
    h = 1e-6
    worst = 0.0
    for probe in ((0, 0, 0), (1, 3, 5), (0, 2, 4), (1, 1, 2)):
        up, down = logits.copy(), logits.copy()
        up[probe] += h
        down[probe] -= h
        numeric = (_ref_cross_entropy(up, targets) - _ref_cross_entropy(down, targets)) / (2 * h)
        worst = max(worst, abs(numeric - float(grad[probe])))
    assert worst < 1e-7, (
        f"worst disagreement with finite differences was {worst:.3e}; a clean factor of "
        f"{b * t} between the two means the division by B*T is missing"
    )


# --- exercise 5: attention_backward (the key gate) -------------------------------------------

def _backward_fd_error(seed, b=2, t=5, d=4):
    """Max |analytic - central difference| over X, Wq, Wk, Wv for objective sum(context*R)."""
    x, wq, wk, wv, rng = _rand_case(seed, b, t, d)
    weight = rng.normal(0, 1, (b, t, d))
    _, ref_cache = _ref_attention(x, wq, wk, wv)   # canonical cache: isolates the backward
    dx, dwq, dwk, dwv = lesson.attention_backward(weight, ref_cache)

    for name, got, want in (("dX", dx, x.shape), ("dWq", dwq, wq.shape),
                            ("dWk", dwk, wk.shape), ("dWv", dwv, wv.shape)):
        assert np.asarray(got).shape == want, (
            f"{name} came back {np.asarray(got).shape}, expected {want}"
        )

    def objective():
        c, _ = _ref_attention(x, wq, wk, wv)
        return float((c * weight).sum())

    h = 1e-6
    worst = {}
    for name, arr, analytic in (("X", x, dx), ("Wq", wq, dwq), ("Wk", wk, dwk), ("Wv", wv, dwv)):
        flat = arr.reshape(-1)
        err = 0.0
        step = max(1, flat.size // 6)
        for q in range(0, flat.size, step):
            old = flat[q]
            flat[q] = old + h
            up = objective()
            flat[q] = old - h
            down = objective()
            flat[q] = old
            numeric = (up - down) / (2 * h)
            err = max(err, abs(numeric - float(np.asarray(analytic).reshape(-1)[q])))
        worst[name] = err
    return worst


def test_attention_backward_matches_finite_differences():
    """THE gate. Every one of dX, dWq, dWk, dWv against central differences, three cases."""
    for seed in (41, 42, 43):
        worst = _backward_fd_error(seed)
        bad = {k: v for k, v in worst.items() if v >= 1e-5}
        assert not bad, (
            f"seed {seed}: these disagree with finite differences: "
            f"{ {k: f'{v:.3e}' for k, v in bad.items()} }. If dWq and dWk fail while dWv "
            "passes, the softmax Jacobian is wrong: it is A*(dA - (dA*A).sum(-1, keepdims)), "
            "not A*dA. If only dX fails, its three paths were assigned rather than summed. "
            "If everything fails by the same factor, the 1/sqrt(D) scale was dropped."
        )


# --- exercise 6: make_batch -----------------------------------------------------------------

def test_make_batch_offset_and_shapes():
    """Targets are inputs shifted by exactly one, on a corpus where that is unambiguous."""
    toy = np.arange(60)
    for batch_size, context_len in ((4, 6), (1, 3), (7, 12)):
        xb, yb = lesson.make_batch(toy, batch_size, context_len, np.random.default_rng(51))
        assert xb.shape == (batch_size, context_len), (
            f"inputs {xb.shape}, expected {(batch_size, context_len)}"
        )
        assert yb.shape == (batch_size, context_len), (
            f"targets {yb.shape}, expected {(batch_size, context_len)}"
        )
        assert (yb == xb + 1).all(), (
            f"on arange, every target must be its input + 1; got inputs {xb[0]} and targets "
            f"{yb[0]}. Identical windows mean the target slice does not start one later."
        )


def test_make_batch_covers_corpus_and_stays_in_range():
    """Rows must differ, stay inside the corpus, and respect the caller's generator."""
    data = lesson.DATA
    xb, yb = lesson.make_batch(data, 24, 16, np.random.default_rng(52))
    assert xb.dtype == np.int64 and yb.dtype == np.int64, (
        f"dtypes were {xb.dtype}/{yb.dtype}; these index an embedding, so int64"
    )
    assert xb.min() >= 0 and xb.max() < lesson.VOCAB_SIZE, "an input id is outside the vocabulary"
    assert yb.min() >= 0 and yb.max() < lesson.VOCAB_SIZE, "a target id is outside the vocabulary"
    assert (yb[:, :-1] == xb[:, 1:]).all(), "targets are not the inputs shifted by one"
    assert len({tuple(r) for r in xb}) > 1, (
        "every row is identical — pass size=batch_size to rng.integers so each row gets its "
        "own start position"
    )
    a1, _ = lesson.make_batch(data, 8, 10, np.random.default_rng(99))
    a2, _ = lesson.make_batch(data, 8, 10, np.random.default_rng(99))
    assert (a1 == a2).all(), (
        "two calls with identically seeded generators produced different batches — use the "
        "`rng` argument, not a module-level or global random source"
    )
    tight = np.arange(20)
    for _ in range(40):  # would raise or go out of range if the upper bound is wrong
        xs, ys = lesson.make_batch(tight, 5, 19, np.random.default_rng(None))
        assert xs.shape == (5, 19) and ys.shape == (5, 19), (
            f"on a 20-character corpus with context_len 19 there is exactly one legal start "
            f"position, but the shapes came back {xs.shape} and {ys.shape}; an upper bound of "
            f"len(data) - context_len - 1 leaves no legal start at all here"
        )
        assert ys.max() <= 19, "a target ran past the end of the corpus"


RUBRIC = [
    (test_softmax_matches_reference, 4,
     "subtract the last-axis max with keepdims=True, then exp and divide by the sum"),
    (test_softmax_survives_huge_logits, 3,
     "np.exp(1000) is inf; shift by the row max first and the largest exponent becomes exp(0)"),
    (test_cross_entropy_matches_reference, 4,
     "logsumexp(logits) - logits[target], averaged over all B*T positions"),
    (test_cross_entropy_survives_huge_logits, 3,
     "pull the row max out of the logsumexp and add it back outside the log"),
    (test_attention_cache_contract, 2,
     "return the ten documented keys: X, Wq, Wk, Wv, Q, K, V, A, mask, scale"),
    (test_attention_forward_matches_reference, 4,
     "scale by 1/sqrt(x.shape[-1]) before masking, and softmax the masked scores"),
    (test_attention_cannot_see_the_future, 4,
     "np.where(tril_mask, scores, -np.inf) BEFORE the softmax; a finite sentinel is only as "
     "good as its underflow margin"),
    (test_cross_entropy_backward_matches_finite_differences, 4,
     "(softmax(logits) - onehot(targets)) / targets.size"),
    (test_attention_backward_matches_finite_differences, 6,
     "softmax Jacobian is A*(dA - (dA*A).sum(-1, keepdims=True)); dX sums three paths"),
    (test_make_batch_offset_and_shapes, 3,
     "targets start one character later than inputs: data[s+1 : s+1+context_len]"),
    (test_make_batch_covers_corpus_and_stays_in_range, 3,
     "rng.integers(0, len(data) - context_len, size=batch_size), one start per row"),
]
