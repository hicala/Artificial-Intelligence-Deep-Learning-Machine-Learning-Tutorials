"""Autograder rubric for P03-L03. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* the rubric never grades a student against their own code. Every expected value is either a
  closed-form constant, or computed here from plain Python arithmetic (`_rms`, `_kurtosis`,
  `_auc`) that shares no line with the reference implementation.
* `test_kurtosis_uses_population_variance` exists because the docstring states a requirement
  that smooth data satisfies by accident. On four samples the N and N-1 conventions differ by
  a factor of 0.5625, which is the only input size where the mistake is unmissable.
* `test_amplitude_spectrum_odd_length` is the only case that separates a correct Nyquist rule
  from `amp[..., -1] /= 2` applied unconditionally. For odd n there IS no Nyquist bin, and
  every even-length case in the notebook passes either way.
* `test_band_energy_parseval` grades an identity rather than a number, so a student who
  scales the spectrum by any consistent constant still fails, and a student who writes the
  whole thing differently but correctly still passes.
* `test_cheapest_alarm_never_alarm_and_ties` grades the stated tie rule twice: once where
  `inf` would win it, and once between two REAL thresholds. Nothing about the output looks
  wrong when a tie goes the wrong way — the cost printed is the right cost — but the threshold
  handed to a plant is the higher of two equally cheap ones, which is half the failures missed
  for no saving. The second case is the only thing in the rubric that pins the documented
  "every distinct value, ascending" candidate list rather than just its contents.
* `test_cheapest_alarm_validates_prevalence` and `test_fit_load_model_validates_rows` each
  grade BOTH sides of their boundary. An interval is not tested by rejecting what is outside
  it; `not 0 < p < 1` and `n_fit < 4` both reject legal input and both are invisible to a
  rubric that only ever hands in illegal input.
* `test_fit_load_model_uses_fit_rows_only` poisons the rows outside the mask with +/-900 C.
  A fit that reads them cannot come back clean, and no other case in the lesson can tell a
  masked fit from an unmasked one.
* `test_the_load_column_is_real`, `test_the_temperature_residual_is_real` and
  `test_the_money_ranking_is_real` are integration tests on the lesson's own bench. They
  assert the DIRECTION and a conservative magnitude of each result — RMS fooled by load,
  kurtosis and crest immune, the residual far ahead of the raw temperature — never an exact
  AUC, so a student whose spectrum differs in the last digit is not failed for it.
* memory and timing are not graded here; tools/execute.py owns those.
"""
import importlib.util, math, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


# --- independent references, written in plain Python -----------------------------------

def _rms(values) -> float:
    xs = [float(v) for v in np.asarray(values).ravel()]
    return math.sqrt(sum(v * v for v in xs) / len(xs))


def _kurtosis(values) -> float:
    xs = [float(v) for v in np.asarray(values).ravel()]
    mean = sum(xs) / len(xs)
    second = sum((v - mean) ** 2 for v in xs) / len(xs)
    fourth = sum((v - mean) ** 4 for v in xs) / len(xs)
    return fourth / (second * second)


def _auc(healthy, degrading) -> float:
    hs = [float(v) for v in np.asarray(healthy).ravel()]
    ds = [float(v) for v in np.asarray(degrading).ravel()]
    wins = sum(1 for d in ds for h in hs if d > h)
    ties = sum(1 for d in ds for h in hs if d == h)
    return (wins + 0.5 * ties) / (len(ds) * len(hs))


_CACHE: dict[str, np.ndarray] = {}


def _feature(name: str) -> np.ndarray:
    """Compute one feature over the whole bench once, and remember it."""
    if name not in _CACHE:
        vib = lesson.VIB
        makers = {
            "rms": lambda: lesson.rms(vib),
            "crest_factor": lambda: lesson.crest_factor(vib),
            "kurtosis": lambda: lesson.kurtosis(vib),
            "band_energy": lambda: lesson.band_energy(vib, lesson.FS, *lesson.DEFECT_BAND),
            "band_fraction": lambda: lesson.band_energy_fraction(vib, lesson.FS,
                                                                 *lesson.DEFECT_BAND),
        }
        _CACHE[name] = np.asarray(makers[name](), dtype=float)
    return _CACHE[name]


def _split(values):
    """(healthy, degrading, healthy-at-low-load, healthy-at-high-load) for one feature."""
    h, d = lesson.HEALTHY, lesson.DEGRADING
    return values[h], values[d], values[h & lesson.IS_LOW], values[h & lesson.IS_HIGH]


# --- exercise 1: rms -------------------------------------------------------------------

def test_rms_values():
    got = lesson.rms(np.array([3.0, 4.0]))
    assert np.isclose(got, _rms([3.0, 4.0])), f"rms([3, 4]) = {got!r}, expected 3.5355"
    constant = lesson.rms(np.array([10.0, 10.0, 10.0]))
    assert np.isclose(constant, 10.0), (
        f"a constant capture has RMS equal to that constant; got {constant!r}. np.std gives "
        "0.0 here — RMS does not subtract the mean"
    )
    biased = lesson.rms(np.array([1.0, 3.0]))
    assert np.isclose(biased, _rms([1.0, 3.0])), (
        f"rms([1, 3]) = {biased!r}, expected sqrt(5) = 2.2360 (np.std would give 1.0)"
    )
    assert np.asarray(lesson.rms(np.zeros((2, 3, 7)))).shape == (2, 3), (
        "reduce the LAST axis only, with axis=-1"
    )
    rng = np.random.default_rng(7)
    block = rng.normal(5.0, 2.0, (3, 4, 16))   # a large DC offset, where np.std diverges
    expected = np.sqrt((block ** 2).mean(axis=-1))
    assert np.allclose(lesson.rms(block), expected), (
        "on a (3, 4, 16) block with a non-zero mean the answer must be the elementwise RMS "
        "along the last axis"
    )


# --- exercise 2: crest_factor -----------------------------------------------------------

def test_crest_factor_values():
    square = lesson.crest_factor(np.array([1.0, -1.0, 1.0, -1.0]))
    assert np.isclose(square, 1.0), f"a square wave has crest factor 1.0; got {square!r}"
    t = np.arange(8192) / 8192.0
    sine = lesson.crest_factor(np.sin(2.0 * np.pi * 16.0 * t))
    assert np.isclose(sine, math.sqrt(2.0), atol=1e-3), (
        f"a sine has crest factor sqrt(2) = 1.4142; got {sine!r}"
    )
    zeroed = lesson.crest_factor(np.zeros(8))
    assert np.isclose(zeroed, 0.0), (
        f"an all-zero capture has no crest factor; the stated convention is 0.0, got "
        f"{zeroed!r}. A NaN here spreads into every ranking downstream"
    )
    batched = lesson.crest_factor(np.array([[1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]]))
    assert np.asarray(batched).shape == (2,), (
        f"on a (2, 4) input return one crest factor per row; got shape "
        f"{np.asarray(batched).shape}"
    )
    assert np.allclose(batched, [1.0, 0.0]), (
        f"expected [1.0, 0.0], got {np.asarray(batched).tolist()}"
    )


def test_crest_factor_uses_absolute_peak():
    negative = lesson.crest_factor(np.array([-5.0, 1.0, 1.0, 1.0]))
    assert np.isclose(negative, 5.0 / _rms([-5.0, 1.0, 1.0, 1.0])), (
        f"crest_factor([-5, 1, 1, 1]) = {negative!r}, expected 5 / sqrt(7) = 1.8898. Near "
        "0.378 means np.max was used instead of the peak of the ABSOLUTE value, so a "
        "downward-pointing impact was thrown away"
    )
    flipped = lesson.crest_factor(-np.array([-5.0, 1.0, 1.0, 1.0]))
    assert np.isclose(flipped, negative), (
        "crest factor must not change when the whole capture is negated; an accelerometer "
        "mounted the other way up measures the same bearing"
    )


def test_crest_factor_is_gain_invariant():
    rng = np.random.default_rng(31)
    capture = rng.normal(0.0, 1.0, (3, 512))
    capture[0, 100] = 12.0
    plain = np.asarray(lesson.crest_factor(capture), dtype=float)
    scaled = np.asarray(lesson.crest_factor(capture * 41.0), dtype=float)
    assert np.allclose(plain, scaled, rtol=1e-12), (
        f"multiplying the capture by 41 changed the crest factor from {plain.tolist()} to "
        f"{scaled.tolist()}. Both peak and RMS scale with the gain, so the ratio cannot"
    )


# --- exercise 3: kurtosis ---------------------------------------------------------------

def test_kurtosis_values():
    t = np.arange(8192) / 8192.0
    sine = lesson.kurtosis(np.sin(2.0 * np.pi * 16.0 * t))
    assert np.isclose(sine, 1.5, atol=1e-6), (
        f"a sine has kurtosis exactly 1.5; got {sine!r}"
    )
    sample = np.array([1.0, 2.0, 3.0, 9.0])
    assert np.isclose(lesson.kurtosis(sample), _kurtosis(sample)), (
        f"kurtosis([1, 2, 3, 9]) = {lesson.kurtosis(sample)!r}, expected "
        f"{_kurtosis(sample):.6f}"
    )
    flat = lesson.kurtosis(np.full(10, 4.0))
    assert np.isclose(flat, 0.0), (
        f"a constant capture has zero variance; the stated convention is 0.0, got {flat!r}"
    )
    assert np.asarray(lesson.kurtosis(np.zeros((3, 4, 20)))).shape == (3, 4), (
        "reduce the LAST axis only"
    )
    rng = np.random.default_rng(19)
    rows = rng.normal(0.0, 1.0, (4, 50_000))
    per_row = np.asarray(lesson.kurtosis(rows), dtype=float)
    assert np.all(np.abs(per_row - 3.0) < 0.15), (
        f"four rows of Gaussian noise must each come out near 3.0; got {per_row.round(3)}"
    )


def test_kurtosis_is_not_excess():
    noise = lesson.kurtosis(np.random.default_rng(3).normal(0.0, 1.0, 200_000))
    assert abs(noise - 3.0) < 0.1, (
        f"Gaussian noise must give about 3.0, not {noise!r}. A value near 0.0 is EXCESS "
        "kurtosis: this lesson uses the un-shifted fourth standardised moment, where 3 is "
        "the noise floor and a condition indicator below it means a smooth, tonal machine"
    )


def test_kurtosis_uses_population_variance():
    # On four samples the two variance conventions differ by (3/4)**2 = 0.5625, so a rubric
    # that only ever sees long captures cannot tell them apart. This one can.
    sample = np.array([1.0, 2.0, 3.0, 9.0])
    expected = _kurtosis(sample)
    got = float(lesson.kurtosis(sample))
    assert np.isclose(got, expected, rtol=1e-9), (
        f"kurtosis([1, 2, 3, 9]) = {got:.6f}, expected {expected:.6f}. A value of "
        f"{expected * 0.5625:.6f} means the denominator used the SAMPLE variance (ddof=1) "
        "while the numerator used N — divide by mean((x - mean(x))**2)**2, both with N"
    )


def test_kurtosis_is_affine_invariant():
    rng = np.random.default_rng(41)
    capture = rng.normal(0.0, 1.0, (2, 4096))
    capture[0, ::400] += 8.0
    plain = np.asarray(lesson.kurtosis(capture), dtype=float)
    moved = np.asarray(lesson.kurtosis(capture * 17.0 - 300.0), dtype=float)
    assert np.allclose(plain, moved, rtol=1e-9), (
        f"kurtosis is invariant to scale and shift; {plain.tolist()} became "
        f"{moved.tolist()}. If only the shift breaks it, the mean is not being subtracted"
    )


# --- exercise 4: amplitude_spectrum -----------------------------------------------------

def test_amplitude_spectrum_freqs():
    freqs, amp = lesson.amplitude_spectrum(np.zeros(1000), 1000.0)
    assert np.asarray(freqs).shape == (501,) and np.asarray(amp).shape == (501,), (
        f"n = 1000 gives 1000 // 2 + 1 = 501 bins; got freqs {np.shape(freqs)} and amp "
        f"{np.shape(amp)}"
    )
    assert np.allclose(freqs, np.fft.rfftfreq(1000, 1.0 / 1000.0)), (
        "freqs must be np.fft.rfftfreq(n, 1/fs). Passing a spacing of 1 gives cycles per "
        "SAMPLE, so every band edge in Hz would select the wrong bins"
    )
    odd_freqs, odd_amp = lesson.amplitude_spectrum(np.zeros(9), 9.0)
    assert np.asarray(odd_freqs).shape == (5,) and np.asarray(odd_amp).shape == (5,), (
        f"n = 9 gives (9 + 1) // 2 = 5 bins; got {np.shape(odd_freqs)}"
    )


def test_amplitude_spectrum_scaling():
    t = np.arange(1000) / 1000.0
    freqs, amp = lesson.amplitude_spectrum(2.0 * np.sin(2.0 * np.pi * 100.0 * t), 1000.0)
    peak = int(np.argmax(np.asarray(amp)))
    assert np.isclose(freqs[peak], 100.0), (
        f"the peak of a 100 Hz sine landed at {freqs[peak]} Hz"
    )
    assert np.isclose(amp[peak], 2.0, atol=1e-6), (
        f"a sine of amplitude 2.0 must read 2.0; got {amp[peak]:.6f}. 1.0 means the factor "
        "of 2 for the folded negative frequency is missing"
    )
    dc = np.asarray(lesson.amplitude_spectrum(np.full(1000, 3.0), 1000.0)[1])
    assert np.isclose(dc[0], 3.0, atol=1e-9), (
        f"a constant 3.0 has DC amplitude 3.0; got {dc[0]:.6f}. 6.0 means bin 0 was doubled "
        "with the rest — DC has no negative-frequency twin to fold in"
    )
    assert np.allclose(dc[1:], 0.0, atol=1e-9), (
        "a constant signal has no energy anywhere but DC"
    )
    nyq = np.asarray(lesson.amplitude_spectrum(np.tile([1.0, -1.0], 8), 16.0)[1])
    assert np.isclose(nyq[-1], 1.0, atol=1e-9), (
        f"the alternating +1/-1 signal is a Nyquist square of amplitude 1.0; got "
        f"{nyq[-1]:.6f}. 2.0 means the Nyquist bin was doubled — for even n it must not be"
    )


def test_amplitude_spectrum_odd_length():
    # The ONLY case that separates "halve the last bin when n is even" from "halve it always".
    n, fs = 9, 9.0
    t = np.arange(n) / fs
    freqs, amp = lesson.amplitude_spectrum(2.0 * np.cos(2.0 * np.pi * 4.0 * t), fs)
    assert np.isclose(freqs[-1], 4.0), f"the last bin of n=9 at 9 Hz is 4 Hz, got {freqs[-1]}"
    assert np.isclose(amp[-1], 2.0, atol=1e-9), (
        f"for ODD n there is no Nyquist bin, so the last bin is an ordinary one and keeps "
        f"its factor of 2; got {amp[-1]:.6f} instead of 2.0. Guard the halving with "
        "`if n % 2 == 0`"
    )


def test_amplitude_spectrum_batched():
    amp = np.asarray(lesson.amplitude_spectrum(np.zeros((3, 7, 64)), lesson.FS)[1])
    assert amp.shape == (3, 7, 33), (
        f"the transform runs along the LAST axis; a (3, 7, 64) input gives (3, 7, 33), got "
        f"{amp.shape}"
    )
    rng = np.random.default_rng(2)
    rows = rng.normal(0.0, 1.0, (4, 256))
    stacked = np.asarray(lesson.amplitude_spectrum(rows, 256.0)[1])
    for i in range(4):
        one = np.asarray(lesson.amplitude_spectrum(rows[i], 256.0)[1])
        assert np.allclose(stacked[i], one), (
            f"row {i} of a batched call differs from the same row transformed alone"
        )


# --- exercise 5: band_energy ------------------------------------------------------------

def test_band_energy_parseval():
    for n, label in ((1024, "even"), (1025, "odd"), (997, "odd prime")):
        noisy = np.random.default_rng(5).normal(3.0, 2.0, n)   # deliberately non-zero mean
        total = float(lesson.band_energy(noisy, 1000.0, 0.0, np.inf))
        assert np.isclose(total, _rms(noisy) ** 2, rtol=1e-9), (
            f"Parseval fails for {label} n = {n}: the full band gives {total!r} but rms**2 "
            f"is {_rms(noisy) ** 2!r}. Check the DC bin (never halved) and, for odd n, that "
            "the LAST bin is halved like any other"
        )
    batched = np.asarray(lesson.band_energy(lesson.VIB[:4], lesson.FS, 0.0, np.inf))
    assert batched.shape == (4,), (
        f"on (units, samples) input return one number per unit; got shape {batched.shape}"
    )
    assert np.allclose(batched, np.asarray(lesson.rms(lesson.VIB[:4])) ** 2, rtol=1e-9), (
        "Parseval must hold per row on the bench's own captures too"
    )


def test_band_energy_half_open():
    t = np.arange(1000) / 1000.0
    sine = 2.0 * np.sin(2.0 * np.pi * 100.0 * t)
    inside = float(lesson.band_energy(sine, 1000.0, 90.0, 110.0))
    assert np.isclose(inside, 2.0, atol=1e-6), (
        f"a sine of amplitude 2.0 carries mean-square power A**2/2 = 2.0; got {inside!r}. "
        "4.0 means amp**2 was summed without halving the ordinary bins"
    )
    assert np.isclose(float(lesson.band_energy(sine, 1000.0, 0.0, 100.0)), 0.0, atol=1e-9), (
        "[0, 100) must EXCLUDE the 100 Hz bin — the band is half-open, lo <= f < hi"
    )
    assert np.isclose(float(lesson.band_energy(sine, 1000.0, 100.0, 200.0)), 2.0,
                      atol=1e-6), "[100, 200) must INCLUDE the 100 Hz bin"
    # Adjacent half-open bands partition the spectrum exactly: no bin counted twice or lost.
    rng = np.random.default_rng(13)
    mixed = rng.normal(0.0, 1.0, 512)
    pieces = sum(float(lesson.band_energy(mixed, 512.0, lo, hi))
                 for lo, hi in ((0.0, 64.0), (64.0, 128.0), (128.0, 256.0), (256.0, np.inf)))
    assert np.isclose(pieces, _rms(mixed) ** 2, rtol=1e-9), (
        f"four adjacent half-open bands must sum to the total power: got {pieces!r} against "
        f"{_rms(mixed) ** 2!r}. A closed upper edge double-counts every shared bin"
    )
    dc = float(lesson.band_energy(np.full(64, 3.0), 64.0, 0.0, 1.0))
    assert np.isclose(dc, 9.0, atol=1e-9), (
        f"the DC bin carries amp**2, not amp**2/2: a constant 3.0 has total power 9.0, got "
        f"{dc!r}"
    )


# --- exercise 6: band_energy_fraction ---------------------------------------------------

def test_band_energy_fraction():
    t = np.arange(1000) / 1000.0
    sine = 2.0 * np.sin(2.0 * np.pi * 100.0 * t)
    assert np.isclose(float(lesson.band_energy_fraction(sine, 1000.0, 90.0, 110.0)), 1.0,
                      atol=1e-9), "all of a pure sine's power is inside a band containing it"
    assert np.isclose(float(lesson.band_energy_fraction(sine, 1000.0, 0.0, np.inf)), 1.0,
                      atol=1e-9), "over every frequency the fraction is exactly 1.0"
    rng = np.random.default_rng(17)
    mixed = rng.normal(0.0, 1.0, 4096)
    plain = float(lesson.band_energy_fraction(mixed, lesson.FS, *lesson.DEFECT_BAND))
    scaled = float(lesson.band_energy_fraction(mixed * 123.0, lesson.FS, *lesson.DEFECT_BAND))
    assert np.isclose(plain, scaled, rtol=1e-9), (
        f"the fraction must not move when the capture is multiplied by 123: {plain!r} became "
        f"{scaled!r}. Dividing by anything but the total power leaves the gain in"
    )
    assert np.isclose(float(lesson.band_energy_fraction(np.zeros(64), lesson.FS, 0.0, 100.0)),
                      0.0), "a silent capture gives 0.0, not a NaN"
    per_unit = np.asarray(lesson.band_energy_fraction(lesson.VIB[:5], lesson.FS,
                                                      *lesson.DEFECT_BAND))
    assert per_unit.shape == (5,), f"expected one fraction per unit, got {per_unit.shape}"
    assert np.all((per_unit > 0.0) & (per_unit < 1.0)), (
        f"every fraction must lie strictly inside (0, 1) on this bench; got "
        f"{per_unit.round(4).tolist()}"
    )


# --- exercise 7: separation_auc ---------------------------------------------------------

def test_separation_auc_values():
    assert lesson.separation_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0, (
        "non-overlapping samples with the degrading one higher give exactly 1.0"
    )
    assert lesson.separation_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 0.0, (
        "the statistic is directional and the argument order is (healthy, degrading); "
        "swapping them must give 0.0, not 1.0"
    )
    tied = lesson.separation_auc(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert np.isclose(tied, 0.5), (
        f"identical samples give 0.5; got {tied!r}. 0.0 or 1.0 means ties were counted as "
        "losses or as wins rather than as halves"
    )
    partial = lesson.separation_auc(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0]))
    assert np.isclose(partial, _auc([0.0, 1.0, 2.0], [1.0, 3.0])), (
        f"expected {_auc([0.0, 1.0, 2.0], [1.0, 3.0])} (4 wins and 1 tie out of 6 pairs); "
        f"got {partial!r}"
    )
    uneven = lesson.separation_auc(np.array([0.0, 0.0, 0.0, 5.0]), np.array([1.0]))
    assert np.isclose(uneven, _auc([0.0, 0.0, 0.0, 5.0], [1.0])), (
        f"samples of different sizes must divide by n_healthy * n_degrading; got {uneven!r}"
    )
    assert isinstance(lesson.separation_auc(np.array([1.0]), np.array([2.0])), float), (
        "return a plain Python float; a 0-d numpy array formats differently in a report"
    )


def test_separation_auc_is_rank_based():
    rng = np.random.default_rng(29)
    h, d = rng.normal(0.0, 1.0, 60), rng.normal(0.7, 1.4, 45)
    plain = lesson.separation_auc(h, d)
    assert np.isclose(plain, _auc(h, d)), (
        f"got {plain!r}, expected {_auc(h, d)!r} from an independent pair count"
    )
    for transform, name in ((np.exp, "exp"), (lambda v: v ** 3, "cube"),
                            (lambda v: 5.0 * v + 100.0, "affine")):
        moved = lesson.separation_auc(transform(h), transform(d))
        assert np.isclose(plain, moved), (
            f"a rank statistic cannot change under the strictly increasing {name} transform: "
            f"{plain!r} became {moved!r}. Comparing means or distances is not the same thing"
        )


def test_separation_auc_validates_samples():
    for healthy, degrading in ((np.array([]), np.array([1.0])),
                               (np.array([1.0]), np.array([])),
                               (np.array([]), np.array([]))):
        try:
            lesson.separation_auc(healthy, degrading)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "an empty sample must raise ValueError; returning a NaN or dividing by zero "
                "puts a silent NaN into the scorecard"
            )


# --- exercise 8: cheapest_alarm ---------------------------------------------------------

def test_cheapest_alarm_picks():
    thr, cost = lesson.cheapest_alarm(np.array([0.0, 1.0]), np.array([2.0, 3.0]),
                                      (10.0, 100.0, 5.0), 0.5, 100)
    assert np.isclose(thr, 2.0) and np.isclose(cost, 500.0), (
        f"expected (2.0, 500.0), got ({thr!r}, {cost!r}). At 2.0 every degrading unit alarms "
        "and no healthy one does, so the bill is 50 planned interventions at 10"
    )
    # A feature with one overlapping healthy unit: paying one false alarm beats one failure.
    thr2, cost2 = lesson.cheapest_alarm(np.array([0.0, 2.5]), np.array([2.0, 3.0]),
                                        (10.0, 1000.0, 5.0), 0.5, 100)
    assert np.isclose(thr2, 2.0), (
        f"with an unplanned failure at 1000 and a false alarm at 5, the cheapest threshold "
        f"is 2.0 (catch both, accept the false alarm); got {thr2!r}"
    )
    assert np.isclose(cost2, 50 * 10.0 + 25 * 5.0), (
        f"expected 50*10 + 25*5 = 625, got {cost2!r}. fpr is a RATE from the healthy sample "
        "(here 1 of 2), multiplied by the healthy half of the fleet"
    )
    assert isinstance(thr, float) and isinstance(cost, float), (
        "return plain floats, not numpy scalars — both go straight into a report"
    )


def test_cheapest_alarm_never_alarm_and_ties():
    never_thr, never_cost = lesson.cheapest_alarm(np.array([0.0, 1.0]), np.array([2.0, 3.0]),
                                                  (10.0, 1.0, 50.0), 0.5, 100)
    assert np.isinf(never_thr) and np.isclose(never_cost, 50.0), (
        f"when a failure costs 1 and a false alarm 50, never alarming is cheapest: expected "
        f"(inf, 50.0), got ({never_thr!r}, {never_cost!r}). np.inf must be among the "
        "candidates, and at inf nothing alarms, so the whole bill is missed failures"
    )
    # A deliberate tie: catching everything costs exactly what missing everything costs.
    tie_thr, tie_cost = lesson.cheapest_alarm(np.array([0.0]), np.array([1.0]),
                                              (10.0, 10.0, 50.0), 0.5, 100)
    assert np.isclose(tie_cost, 500.0), f"both options cost 500 here; got {tie_cost!r}"
    assert np.isclose(tie_thr, 1.0), (
        f"on a tie the SMALLEST threshold wins, so 1.0 rather than inf; got {tie_thr!r}. "
        "Put np.inf last in the candidate array and let argmin take the first minimum — "
        "shipping 'never alarm' on a feature that works is the expensive way to be right"
    )
    # A tie between two REAL thresholds, with the samples handed over out of ascending order.
    # Every rate here is 0, 0.5 or 1, so all five costs are exact binary fractions and nothing
    # depends on floating-point luck. Costs are 1.0 -> 3000, 2.0 -> 1750, 3.0 -> 3000,
    # 5.0 -> 1750, inf -> 3000: moving 2.0 -> 5.0 gives up half the true positives (worth
    # n_bad * (60 - 10) / 2 = 1250) and buys back half the false alarms (worth
    # n_good * 50 / 2 = 1250), an exact wash. The rule says take 2.0; a candidate list left in
    # the caller's order answers 5.0 and prices it identically, so nothing looks wrong.
    pair_thr, pair_cost = lesson.cheapest_alarm(np.array([1.0, 3.0]), np.array([5.0, 2.0]),
                                                (10.0, 60.0, 50.0), 0.5, 100)
    assert np.isclose(pair_cost, 1750.0), (
        f"thresholds 2.0 and 5.0 both cost 1750 on this input, and every other candidate "
        f"costs 3000; got {pair_cost!r}. A wrong cheapest cost means the candidate set itself "
        "is wrong, not just its order"
    )
    assert np.isclose(pair_thr, 2.0), (
        f"2.0 and 5.0 tie at 1750 here and the smaller one wins; got {pair_thr!r}. 5.0 means "
        "the candidates were scored in the order the caller passed them rather than sorted "
        "ascending — np.unique(np.concatenate([healthy, degrading])) sorts and de-duplicates, "
        "and np.inf goes after it. Shipping the higher of two equally cheap thresholds misses "
        "half the failures for no saving at all"
    )


def test_cheapest_alarm_uses_prevalence():
    rng = np.random.default_rng(23)
    h, d = rng.normal(0.0, 1.0, 200), rng.normal(1.2, 1.0, 200)
    rare, _ = lesson.cheapest_alarm(h, d, lesson.PRICES, 0.02, lesson.FLEET_SIZE)
    common, _ = lesson.cheapest_alarm(h, d, lesson.PRICES, 0.30, lesson.FLEET_SIZE)
    assert common < rare, (
        f"a fleet where 30% of machines are degrading should alarm more readily than one at "
        f"2%, so its threshold must be LOWER: got {common:.4f} against {rare:.4f}. Equal "
        "thresholds mean prevalence never reaches the cost"
    )
    # Fleet size scales every term, so it cannot move the threshold.
    small, cost_small = lesson.cheapest_alarm(h, d, lesson.PRICES, 0.10, 100)
    large, cost_large = lesson.cheapest_alarm(h, d, lesson.PRICES, 0.10, 10_000)
    assert np.isclose(small, large), (
        f"fleet size multiplies every term equally and cannot move the optimum; got "
        f"{small:.4f} against {large:.4f}"
    )
    assert np.isclose(cost_large, cost_small * 100.0, rtol=1e-9), (
        f"a hundred times the fleet costs a hundred times as much; got {cost_large!r} "
        f"against {cost_small!r}"
    )


def test_cheapest_alarm_validates_prevalence():
    for bad in (-0.1, 1.5, 2.0):
        try:
            lesson.cheapest_alarm(np.array([0.0]), np.array([1.0]), lesson.PRICES, bad, 100)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"prevalence={bad} must raise ValueError; a negative prevalence silently "
                "produces a negative cost and an argmin that means nothing"
            )
    # ...and the two ENDPOINTS are inside the interval, so they must not raise. A fleet with
    # nothing degrading and a fleet that is entirely degrading are both things a student will
    # price while sweeping prevalence, and `not 0 < p < 1` rejects them both.
    for edge, expected_cost, why in (
            (0.0, 0.0, "with no degrading machines the cheapest rule raises no alarm at all "
                       "and the bill is zero"),
            (1.0, 1000.0, "with every machine degrading the cheapest rule catches all 100 at "
                          "the planned price of 10")):
        try:
            edge_thr, edge_cost = lesson.cheapest_alarm(np.array([0.0, 1.0]),
                                                        np.array([2.0, 3.0]),
                                                        (10.0, 100.0, 5.0), edge, 100)
        except ValueError as exc:
            raise AssertionError(
                f"prevalence={edge} is INSIDE [0, 1] and must not raise, but got "
                f"ValueError: {exc}. The guard is `not 0.0 <= prevalence <= 1.0`; a strict "
                "`not 0.0 < prevalence < 1.0` rejects both ends of the interval the docstring "
                "promises to accept"
            ) from None
        assert np.isclose(edge_cost, expected_cost), (
            f"at prevalence={edge} the cheapest cost is {expected_cost}, because {why}; got "
            f"{edge_cost!r}. n_bad = fleet_size * prevalence and n_good = fleet_size * "
            "(1 - prevalence), so one of the two terms vanishes at each end"
        )
        assert np.isfinite(edge_thr), (
            f"at prevalence={edge} a real threshold is at least as cheap as never alarming, "
            f"so the answer must not be inf; got {edge_thr!r}"
        )


# --- exercises 9 and 10: the temperature model ------------------------------------------

def test_fit_load_model_recovers_coefficients():
    load = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
    ambient = np.array([10.0, 10.0, 20.0, 20.0, 15.0])
    temperature = 5.0 + 2.0 * load + 0.5 * ambient
    coef = np.asarray(lesson.fit_load_model(load, ambient, temperature,
                                            np.ones(5, dtype=bool)))
    assert coef.shape == (3,), (
        f"expected three coefficients [b0, b1, b2], got shape {coef.shape} — the design "
        "matrix needs an explicit column of ones for the intercept"
    )
    assert np.allclose(coef, [5.0, 2.0, 0.5], atol=1e-8), (
        f"on exactly linear data the fit must recover [5, 2, 0.5]; got {coef.round(4).tolist()}. "
        "An answer of roughly [0, 2, 0.75] means the intercept column is missing and the "
        "offset was absorbed into the ambient slope"
    )
    # A second, differently-shaped truth, so a hard-coded answer cannot pass.
    other = -12.0 + 40.0 * load - 1.5 * ambient
    coef2 = np.asarray(lesson.fit_load_model(load, ambient, other, np.ones(5, dtype=bool)))
    assert np.allclose(coef2, [-12.0, 40.0, -1.5], atol=1e-8), (
        f"expected [-12, 40, -1.5], got {coef2.round(4).tolist()}; the column order is "
        "[1, load, ambient]"
    )


def test_fit_load_model_uses_fit_rows_only():
    load = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.5])
    ambient = np.array([10.0, 10.0, 20.0, 20.0, 15.0, 15.0])
    clean = 5.0 + 2.0 * load + 0.5 * ambient
    poisoned = clean.copy()
    poisoned[4:] = [900.0, -900.0]          # two units with a fault, or a dead sensor
    mask = np.array([True, True, True, True, False, False])
    coef = np.asarray(lesson.fit_load_model(load, ambient, poisoned, mask))
    assert np.allclose(coef, [5.0, 2.0, 0.5], atol=1e-8), (
        f"two rows outside fit_rows changed the answer to {coef.round(3).tolist()}; expected "
        "[5, 2, 0.5]. Index load, ambient AND temperature by fit_rows before building the "
        "design matrix — a model fitted on the faults it is meant to reveal cancels them out"
    )


def test_fit_load_model_validates_rows():
    load = np.array([0.0, 1.0, 0.0, 1.0])
    ambient = np.array([10.0, 10.0, 20.0, 20.0])
    temperature = 5.0 + 2.0 * load + 0.5 * ambient
    for mask in (np.array([True, True, False, False]), np.zeros(4, dtype=bool)):
        try:
            lesson.fit_load_model(load, ambient, temperature, mask)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{int(mask.sum())} fitted rows cannot determine three coefficients; raise "
                "ValueError rather than letting lstsq return a minimum-norm guess"
            )
    # THREE rows is the boundary and it is on the legal side of it: three points determine a
    # plane exactly. The guard is `< 3`, not `<= 3`, and nothing else in this rubric fits a
    # three-row model — a commissioning set that small is exactly when a student meets this.
    exact_load = np.array([0.0, 1.0, 0.0])
    exact_ambient = np.array([10.0, 10.0, 20.0])
    exact_temp = 5.0 + 2.0 * exact_load + 0.5 * exact_ambient
    try:
        exact = np.asarray(lesson.fit_load_model(exact_load, exact_ambient, exact_temp,
                                                 np.ones(3, dtype=bool)))
    except ValueError as exc:
        raise AssertionError(
            f"exactly 3 fitted rows determine a 3-parameter plane exactly and must NOT raise, "
            f"but got ValueError: {exc}. The guard is `if n_fit < 3`, not `< 4` or `<= 3`"
        ) from None
    assert np.allclose(exact, [5.0, 2.0, 0.5], atol=1e-8), (
        f"on three exactly-linear rows the fit must recover [5, 2, 0.5]; got "
        f"{exact.round(4).tolist()}"
    )


def test_expected_temperature():
    one = np.asarray(lesson.expected_temperature(np.array([0.5]), np.array([20.0]),
                                                 np.array([5.0, 2.0, 0.5])))
    assert np.allclose(one, [16.0]), (
        f"b0 + b1*load + b2*ambient = 5 + 1 + 10 = 16.0; got {one.tolist()}"
    )
    load = np.array([0.0, 1.0, 0.0, 1.0, 0.25, 0.75])
    ambient = np.array([10.0, 10.0, 20.0, 20.0, 15.0, 30.0])
    coef = np.array([5.0, 2.0, 0.5])
    every = np.asarray(lesson.expected_temperature(load, ambient, coef))
    assert every.shape == (6,), (
        f"expected_temperature applies to EVERY row, not only the fitted ones; six rows in, "
        f"got shape {every.shape}"
    )
    assert np.allclose(every, 5.0 + 2.0 * load + 0.5 * ambient), (
        f"got {every.round(3).tolist()}, expected {(5.0 + 2.0 * load + 0.5 * ambient).tolist()}"
    )


# --- integration: the results this lesson exists to produce -----------------------------

def test_the_load_column_is_real():
    """The whole argument: the dimensional features read the duty point, the others do not."""
    load_auc, fault_auc = {}, {}
    for name in ("rms", "crest_factor", "kurtosis", "band_energy", "band_fraction"):
        healthy, degrading, low, high = _split(_feature(name))
        load_auc[name] = lesson.separation_auc(low, high)
        fault_auc[name] = lesson.separation_auc(healthy, degrading)
    for name in ("rms", "band_energy"):
        assert load_auc[name] > 0.90, (
            f"{name} is not dimensionless — the bench multiplies the whole signal by "
            f"1 + 1.1*load — so healthy units at high load must separate from healthy units "
            f"at low load; its load AUC is {load_auc[name]:.3f}, expected above 0.90"
        )
    for name in ("crest_factor", "kurtosis", "band_fraction"):
        assert 0.35 < load_auc[name] < 0.65, (
            f"{name} is a ratio, so the load gain cancels and its load AUC must sit near "
            f"0.5; got {load_auc[name]:.3f}. A value far from 0.5 means the gain survived — "
            "check that numerator and denominator scale together"
        )
    assert fault_auc["band_fraction"] > 0.90, (
        f"band_energy_fraction over the defect band is the bench's best vibration feature; "
        f"its fault AUC came out {fault_auc['band_fraction']:.3f}, expected above 0.90"
    )
    assert fault_auc["kurtosis"] > 0.85, (
        f"kurtosis should read the impulsiveness of the defect: got "
        f"{fault_auc['kurtosis']:.3f}, expected above 0.85"
    )
    assert fault_auc["band_fraction"] > fault_auc["rms"] + 0.15, (
        f"the defect is a small fraction of the total energy, so RMS must trail the band "
        f"fraction by a wide margin; got {fault_auc['rms']:.3f} against "
        f"{fault_auc['band_fraction']:.3f}"
    )
    # The confounded comparison: healthy at high load against degrading at low load.
    healthy, _, _, _ = _split(_feature("rms"))
    rms_values = _feature("rms")
    confounded = lesson.separation_auc(
        rms_values[lesson.HEALTHY & lesson.IS_HIGH],
        rms_values[lesson.DEGRADING & lesson.IS_LOW])
    assert confounded < 0.30, (
        f"with load confounded, RMS should rank the healthy machines as sicker than the "
        f"broken ones — an AUC well below 0.5. Got {confounded:.3f}, expected below 0.30"
    )
    kurt_values = _feature("kurtosis")
    kurt_confounded = lesson.separation_auc(
        kurt_values[lesson.HEALTHY & lesson.IS_HIGH],
        kurt_values[lesson.DEGRADING & lesson.IS_LOW])
    assert kurt_confounded > 0.80, (
        f"kurtosis is immune to the gain, so confounding the load must barely touch it; got "
        f"{kurt_confounded:.3f}, expected above 0.80"
    )


def test_the_temperature_residual_is_real():
    coef = lesson.fit_load_model(lesson.LOAD, lesson.AMBIENT, lesson.TEMPERATURE,
                                 lesson.COMMISSIONED)
    residual = np.asarray(lesson.TEMPERATURE) - np.asarray(
        lesson.expected_temperature(lesson.LOAD, lesson.AMBIENT, coef))
    raw_fault = lesson.separation_auc(lesson.TEMPERATURE[lesson.HEALTHY],
                                      lesson.TEMPERATURE[lesson.DEGRADING])
    res_fault = lesson.separation_auc(residual[lesson.HEALTHY], residual[lesson.DEGRADING])
    raw_load = lesson.separation_auc(lesson.TEMPERATURE[lesson.HEALTHY & lesson.IS_LOW],
                                     lesson.TEMPERATURE[lesson.HEALTHY & lesson.IS_HIGH])
    res_load = lesson.separation_auc(residual[lesson.HEALTHY & lesson.IS_LOW],
                                     residual[lesson.HEALTHY & lesson.IS_HIGH])
    assert raw_load > 0.85, (
        f"raw bearing temperature is mostly a load measurement on this bench, so its load "
        f"AUC should be high; got {raw_load:.3f}"
    )
    assert 0.35 < res_load < 0.65, (
        f"after removing load and ambient the residual must be blind to load; its load AUC "
        f"is {res_load:.3f}, expected near 0.5. If it is still high, the model was fitted "
        "without an intercept or on the wrong columns"
    )
    assert res_fault > raw_fault + 0.25, (
        f"the residual must detect a fault the raw temperature does not: got {res_fault:.3f} "
        f"against {raw_fault:.3f}, expected a gain of at least 0.25. A small gain means the "
        "model was fitted on every unit, faults included, so it learned the fault away"
    )
    assert abs(float(coef[2]) - 1.0) < 0.15, (
        f"the bench adds ambient to the bearing temperature one for one, so b2 should come "
        f"out near 1.0; got {float(coef[2]):.3f}"
    )


def test_the_money_ranking_is_real():
    do_nothing = lesson.FLEET_SIZE * lesson.PREVALENCE * lesson.PRICES[1]
    costs = {}
    for name in ("rms", "kurtosis", "band_fraction"):
        healthy, degrading, _, _ = _split(_feature(name))
        costs[name] = lesson.cheapest_alarm(healthy, degrading, lesson.PRICES,
                                            lesson.PREVALENCE, lesson.FLEET_SIZE)[1]
    for name, cost in costs.items():
        assert cost < do_nothing, (
            f"{name} costs {cost:,.0f} against {do_nothing:,.0f} for doing nothing at all; "
            "every feature on this bench should at least beat that"
        )
    assert costs["band_fraction"] < costs["rms"] * 0.5, (
        f"the band fraction should save far more money than RMS: {costs['band_fraction']:,.0f} "
        f"against {costs['rms']:,.0f}, expected less than half"
    )
    assert costs["kurtosis"] < costs["rms"], (
        f"kurtosis {costs['kurtosis']:,.0f} should beat RMS {costs['rms']:,.0f} at their own "
        "best thresholds"
    )
    # Prevalence, not the bench's own 50%, is what a fleet threshold must be derived at.
    healthy, degrading, _, _ = _split(_feature("band_fraction"))
    bench_rate_thr, _ = lesson.cheapest_alarm(healthy, degrading, lesson.PRICES, 0.5,
                                              lesson.FLEET_SIZE)
    fleet_thr, _ = lesson.cheapest_alarm(healthy, degrading, lesson.PRICES,
                                         lesson.PREVALENCE, lesson.FLEET_SIZE)
    assert bench_rate_thr < fleet_thr, (
        f"pricing at the bench's own 50% fault rate buys a much more eager alarm than "
        f"pricing at the fleet's {lesson.PREVALENCE:.0%}: got {bench_rate_thr:.4f} against "
        f"{fleet_thr:.4f}, expected the bench-rate threshold to be the lower of the two"
    )


RUBRIC = [
    (test_rms_values, 4,
     "np.sqrt(np.mean(x**2, axis=-1)) — np.std subtracts the mean first and RMS does not"),
    (test_crest_factor_values, 4,
     "peak absolute value over rms, along the last axis, and 0.0 when the rms is zero"),
    (test_crest_factor_uses_absolute_peak, 3,
     "np.max(np.abs(x), axis=-1): an impact that rings negative first is still an impact"),
    (test_crest_factor_is_gain_invariant, 3,
     "peak and rms both scale with the gain, so their ratio must not — if it does, one of "
     "the two was computed on a different array"),
    (test_kurtosis_values, 5,
     "mean((x - mean(x))**4) / mean((x - mean(x))**2)**2, along the last axis"),
    (test_kurtosis_is_not_excess, 3,
     "do not subtract 3: this lesson uses the un-shifted fourth standardised moment"),
    (test_kurtosis_uses_population_variance, 3,
     "the denominator is mean((x - mean(x))**2)**2, with N — not np.var(x, ddof=1)**2"),
    (test_kurtosis_is_affine_invariant, 3,
     "subtract the mean before taking either power, and the shift cancels; the scale "
     "cancels between the fourth moment and the squared variance"),
    (test_amplitude_spectrum_freqs, 3,
     "np.fft.rfftfreq(n, 1/fs) — the spacing is the sample period, not 1"),
    (test_amplitude_spectrum_scaling, 5,
     "abs(rfft(x)) * 2/n, then undo the doubling on bin 0 and on the Nyquist bin"),
    (test_amplitude_spectrum_odd_length, 3,
     "guard the Nyquist correction with `if n % 2 == 0` — an odd-length transform has no "
     "Nyquist bin at all"),
    (test_amplitude_spectrum_batched, 2,
     "pass axis=-1 to rfft so a (units, samples) array gives one spectrum per unit"),
    (test_band_energy_parseval, 5,
     "amp**2/2 for the ordinary bins and amp**2 for DC and Nyquist; summed over every "
     "frequency this must equal rms(x)**2"),
    (test_band_energy_half_open, 4,
     "(freqs >= lo) & (freqs < hi) — a closed upper edge double-counts the shared bin of "
     "two adjacent bands"),
    (test_band_energy_fraction, 4,
     "band_energy divided by rms(x)**2, with 0.0 where the total power is zero"),
    (test_separation_auc_values, 5,
     "(wins + 0.5 * ties) / (n_degrading * n_healthy), with the arguments in the order "
     "(healthy, degrading)"),
    (test_separation_auc_is_rank_based, 3,
     "compare every degrading value with every healthy value; anything that uses the "
     "magnitudes rather than the order is a different statistic"),
    (test_separation_auc_validates_samples, 2,
     "an empty sample is a ValueError, not a division by zero"),
    (test_cheapest_alarm_picks, 5,
     "tpr and fpr are RATES measured on the two samples; the counts come from fleet_size "
     "and prevalence, never from the sample sizes"),
    (test_cheapest_alarm_never_alarm_and_ties, 4,
     "append np.inf to the ascending candidates and take np.argmin, which returns the "
     "lowest tied index"),
    (test_cheapest_alarm_uses_prevalence, 3,
     "n_bad = fleet_size * prevalence and n_good = fleet_size * (1 - prevalence); a higher "
     "prevalence buys more alarms and therefore a lower threshold"),
    (test_cheapest_alarm_validates_prevalence, 2,
     "a prevalence outside [0, 1] is a ValueError"),
    (test_fit_load_model_recovers_coefficients, 5,
     "np.column_stack([np.ones_like(load), load, ambient]) then np.linalg.lstsq(..., "
     "rcond=None); return the three coefficients in that order"),
    (test_fit_load_model_uses_fit_rows_only, 4,
     "index load, ambient AND temperature by fit_rows before building the design matrix"),
    (test_fit_load_model_validates_rows, 2,
     "fewer than three selected rows cannot determine three coefficients: raise ValueError"),
    (test_expected_temperature, 3,
     "b0 + b1*load + b2*ambient, evaluated on every row that was passed in"),
    (test_the_load_column_is_real, 6,
     "if the dimensionless features are not near 0.5 on the load comparison, a gain has "
     "survived somewhere; if RMS is not far above 0.9 there, it is not measuring energy"),
    (test_the_temperature_residual_is_real, 4,
     "fit on COMMISSIONED only, predict on every unit, and subtract — a residual that "
     "barely beats the raw temperature means the fault walked into the fit"),
    (test_the_money_ranking_is_real, 4,
     "check that cheapest_alarm charges the unplanned price to the misses and that "
     "prevalence multiplies only the failure terms"),
]
