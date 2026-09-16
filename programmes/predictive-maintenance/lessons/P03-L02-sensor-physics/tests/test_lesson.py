"""Autograder rubric for P03-L02. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* wherever it can, the rubric checks a student's answer against an INDEPENDENT reference
  written here (a naive O(n) DFT, a Python-loop filter, a closed-form response) rather than
  against another of the student's own functions. Grading one stub with another lets a pair
  of matching errors pass.
* `test_amplitude_spectrum_dc_and_nyquist` exists because the factor-of-2 fold is right for
  every bin except two, and every plausible implementation gets the other bins right. It is
  graded on a pure-DC record and on an alternating +-1 record, where the two routes differ
  by exactly 2x.
* `test_alias_frequency_above_nyquist` is the case that separates the fold from `f % fs`.
  The two agree for every input below `fs/2`, which is most of what a careless test would
  try, so the graded values are deliberately all above it.
* `test_one_pole_lowpass_initial_state` grades a requirement the docstring states and no
  other case detects: the state starts at `x[0]`, not at 0. A filter started at zero passes
  every step-response and causality check and quietly injects a start-up transient, which
  section 12 of the notebook shows is the difference between a velocity RMS that is right
  and one that is four times too big.
* `test_lowpass_response_matches_the_filter` is an integration case across exercises 5 and 6.
  It is the only case that rejects the ANALOGUE formula `1/sqrt(1 + (f/fc)^2)`, which is
  within 3% of the right gain at the frequency most people would spot-check and 6% wrong
  about the delay.
* `test_measure_gain_and_lag_sign` grades the sign convention separately from the magnitude,
  because a flipped sign is invisible in `abs()` and turns a lag into a lead.
* `test_hann_window_is_periodic` grades against `np.hanning`, which is the symmetric window
  and the single most common wrong answer here.
* `test_adc_clips_rather_than_wraps` and `test_adc_negative_rail` grade the asymmetric
  bipolar code range. Clipping to `+-half` looks right and is off by one code at the top.
* `test_the_alias_is_real` and `test_conditioning_recovers_the_amplitude` are integration
  cases on the lesson's own record. They assert the DIRECTION and a conservative magnitude,
  not an exact figure, so a student whose spectrum differs in the last digit is not failed.
* every comparison against a ONE-element or empty expectation asserts the shape before the
  values. `np.allclose` broadcasts, so `np.allclose(np.zeros(0), [100.0])` is True: an
  implementation that returns nothing at all compares equal to a single expected number.
  Three degenerate-case checks were length-blind for exactly that reason — the inclusive
  `f_max` edge of `candidate_true_frequencies`, the one-sample and empty records in
  `integrate_trapezoid`, and `hann_window(1)` — and each now checks `np.shape` first.
* memory and timing are not graded here; tools/execute.py owns those.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


# --- independent references, so the rubric never grades a student against their own code ---

def _naive_dft_bin(x, k):
    """One DFT coefficient, from the definition. Slow on purpose; never uses np.fft."""
    n = len(x)
    return sum(x[i] * np.exp(-2j * np.pi * k * i / n) for i in range(n))


def _ref_lowpass(x, fc, fs, stages):
    a = float(np.exp(-2.0 * np.pi * fc / fs))
    y = list(map(float, x))
    for _ in range(stages):
        out, state = [], y[0]
        for v in y:
            state = a * state + (1.0 - a) * v
            out.append(state)
        y = out
    return np.array(y)


def _ref_highpass(x, fc, fs, stages):
    a = float(np.exp(-2.0 * np.pi * fc / fs))
    y = list(map(float, x))
    for _ in range(stages):
        out, state, prev = [], 0.0, y[0]
        for v in y:
            state = a * (state + v - prev)
            prev = v
            out.append(state)
        y = out
    return np.array(y)


def _ref_fold(f, fs):
    f = abs(float(f))
    best = f % fs
    return min(best, fs - best)


# --- exercise 1: amplitude_spectrum -------------------------------------------------------

def test_amplitude_spectrum_scaling():
    n, fs = 32, 32.0
    sig = 2.5 * np.cos(2 * np.pi * 7 * np.arange(n) / n + 0.4)
    freqs, amps = lesson.amplitude_spectrum(sig, fs)
    assert np.shape(freqs) == np.shape(amps) == (n // 2 + 1,), (
        f"a {n}-sample record has {n // 2 + 1} one-sided bins; got {np.shape(freqs)} and "
        f"{np.shape(amps)}"
    )
    assert np.allclose(freqs, np.fft.rfftfreq(n, d=1.0 / fs)), (
        f"freqs[1] = {freqs[1]!r}; rfftfreq takes the sample SPACING d = 1/fs"
    )
    assert np.isclose(amps[7], 2.5), (
        f"a cosine of amplitude 2.5 read back as {amps[7]!r}. 1.25 is the missing factor of "
        "2 for the folded negative frequency; 40.0 is the missing division by n"
    )
    # Independent: the definition of the DFT, with no np.fft anywhere.
    assert np.isclose(amps[7], 2.0 * abs(_naive_dft_bin(sig, 7)) / n), (
        "the amplitude does not match 2*|X_k|/n computed straight from the DFT definition"
    )
    quiet = np.flatnonzero(np.abs(np.asarray(amps)) > 1e-9)
    assert quiet.tolist() == [7], (
        f"only bin 7 should be non-zero for a single on-bin tone; bins {quiet.tolist()} are"
    )


def test_amplitude_spectrum_dc_and_nyquist():
    _, dc = lesson.amplitude_spectrum(np.full(8, 2.5), 8.0)
    assert np.isclose(dc[0], 2.5), (
        f"a constant 2.5 record has DC amplitude 2.5, got {dc[0]!r}. 5.0 means bin 0 was "
        "doubled: DC has no negative-frequency twin to fold in"
    )
    _, nyq = lesson.amplitude_spectrum(np.array([1.0, -1.0] * 4), 8.0)
    assert np.isclose(nyq[-1], 1.0), (
        f"an alternating +-1 record is a unit cosine at the Nyquist bin; got {nyq[-1]!r}. "
        "2.0 means the last bin of an EVEN-length record was doubled too"
    )
    # For an odd-length record there is no Nyquist bin, so nothing is special about the last.
    n = 9
    odd = 3.0 * np.cos(2 * np.pi * 4 * np.arange(n) / n)
    _, amps = lesson.amplitude_spectrum(odd, float(n))
    assert np.isclose(amps[4], 3.0), (
        f"with an odd-length record the last bin is NOT Nyquist and must still be doubled; "
        f"bin 4 read {amps[4]!r} instead of 3.0"
    )


def test_amplitude_spectrum_window():
    n = 64
    sig = 4.0 * np.cos(2 * np.pi * 9 * np.arange(n) / n)
    _, half = lesson.amplitude_spectrum(sig, float(n), window=np.full(n, 0.5))
    assert np.isclose(half[9], 4.0), (
        f"a constant window of 0.5 halves every sample, so dividing by its MEAN (0.5) must "
        f"restore 4.0; got {half[9]!r}. 2.0 means the window was applied and not corrected; "
        "0.125 means you divided by its sum"
    )
    _, doubled = lesson.amplitude_spectrum(sig, float(n), window=np.full(n, 2.0))
    assert np.isclose(doubled[9], 4.0), (
        f"a constant window of 2.0 doubles every sample, so dividing by its mean (2.0) must "
        f"still give 4.0; got {doubled[9]!r}. 8.0 means the correction was not applied at "
        "all, which a window of 0.5 would have hidden if you had divided by 2 by accident"
    )
    # A real Hann window on an on-bin tone: the coherent-gain division restores it exactly.
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
    _, tapered = lesson.amplitude_spectrum(sig, float(n), window=hann)
    assert np.isclose(tapered[9], 4.0, atol=1e-9), (
        f"a Hann-windowed on-bin tone reads its true amplitude exactly once the coherent "
        f"gain (0.5) is divided out; got {tapered[9]!r}. 2.0 means the division was skipped"
    )


def test_amplitude_spectrum_validates():
    sig = np.zeros(16)
    for bad, why in (((sig, 0.0), "fs = 0"),
                     ((sig, -8.0), "a negative fs"),
                     ((np.zeros((2, 8)), 8.0), "a 2-D signal")):
        try:
            lesson.amplitude_spectrum(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError")
    try:
        lesson.amplitude_spectrum(sig, 16.0, window=np.ones(15))
    except ValueError:
        return
    raise AssertionError("a window whose length differs from the signal must raise ValueError")


# --- exercise 2: peak_in_band -------------------------------------------------------------

def test_peak_in_band():
    freqs = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    amps = np.array([9.0, 1.0, 4.0, 7.0, 2.0])
    got = lesson.peak_in_band(freqs, amps, 5.0, 20.0)
    assert isinstance(got, tuple) and len(got) == 2, f"return a 2-tuple, got {got!r}"
    assert np.allclose(got, (15.0, 7.0)), (
        f"expected (15.0, 7.0), got {got!r}. A first element of 3 means an INDEX was "
        "returned; a first element of 0.0 means the band was ignored"
    )
    assert np.allclose(lesson.peak_in_band(freqs, amps, 0.0, 20.0), (0.0, 9.0)), (
        "the widest band should find the 9.0 at DC"
    )
    # Both edges inclusive: a one-bin band is legal and must find that bin.
    assert np.allclose(lesson.peak_in_band(freqs, amps, 10.0, 10.0), (10.0, 4.0)), (
        "f_lo == f_hi selects exactly the bin at that frequency; both edges are inclusive"
    )
    assert np.allclose(lesson.peak_in_band(freqs, amps, 5.0, 15.0), (15.0, 7.0)), (
        "the upper edge is inclusive, so 15.0 is inside [5, 15]"
    )
    # Ties go to the lower frequency, which is what argmax already does.
    tie = lesson.peak_in_band(np.array([1.0, 2.0]), np.array([3.0, 3.0]), 0.0, 10.0)
    assert np.allclose(tie, (1.0, 3.0)), f"a tie takes the first (lowest) bin; got {tie!r}"


def test_peak_in_band_validates():
    freqs = np.array([0.0, 5.0, 10.0])
    amps = np.array([1.0, 2.0, 3.0])
    for bad, why in (((freqs, amps, 10.0, 5.0), "f_lo above f_hi"),
                     ((freqs, amps, 6.0, 9.0), "a band that contains no bin"),
                     ((freqs, amps, 100.0, 200.0), "a band beyond the spectrum")):
        try:
            lesson.peak_in_band(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError, not return a zero or a NaN")


# --- exercise 3: alias_frequency ----------------------------------------------------------

def test_alias_frequency_below_nyquist():
    for freq in (0.0, 1.0, 100.0, 425.0, 1279.0, 1280.0):
        got = float(lesson.alias_frequency(freq, 2560.0))
        assert np.isclose(got, freq), (
            f"alias_frequency({freq}, 2560) = {got!r}; anything at or below Nyquist is "
            "unchanged"
        )


def test_alias_frequency_above_nyquist():
    # Every case here is above fs/2, where the fold and the modulus disagree.
    for freq, fs in ((1800.0, 2560.0), (2560.0, 2560.0), (2561.0, 2560.0),
                     (5545.0, 2560.0), (5545.0, 3200.0), (5545.0, 2048.0),
                     (7255.0, 2560.0), (9999.0, 1024.0)):
        want = _ref_fold(freq, fs)
        got = float(lesson.alias_frequency(freq, fs))
        assert np.isclose(got, want), (
            f"alias_frequency({freq}, {fs}) = {got!r}, expected {want}. "
            f"{freq % fs} is `f % fs`, which is not the fold"
        )
        assert 0.0 <= got <= fs / 2.0, (
            f"alias_frequency({freq}, {fs}) returned {got!r}, outside [0, {fs / 2}]"
        )


def test_alias_frequency_array_and_validates():
    got = lesson.alias_frequency(np.array([100.0, 1800.0, 5545.0, -425.0]), 2560.0)
    assert np.allclose(got, [100.0, 760.0, 425.0, 425.0]), (
        f"must work elementwise, and a negative frequency means the same tone as its "
        f"magnitude; got {np.asarray(got).tolist()}"
    )
    for fs in (0.0, -2560.0):
        try:
            lesson.alias_frequency(100.0, fs)
        except ValueError:
            continue
        raise AssertionError(f"fs = {fs} must raise ValueError")


# --- exercise 4: candidate_true_frequencies -----------------------------------------------

def test_candidate_true_frequencies():
    got = lesson.candidate_true_frequencies(100.0, 1000.0, 2500.0)
    assert np.allclose(got, [100.0, 900.0, 1100.0, 1900.0, 2100.0]), (
        f"expected [100, 900, 1100, 1900, 2100], got {np.asarray(got).tolist()}"
    )
    assert np.all(np.diff(np.asarray(got)) > 0), (
        f"the result must be sorted with no duplicates; got {np.asarray(got).tolist()}"
    )
    # Independent check: every candidate folds back, and nothing inside f_max is missing.
    brute = [f for f in np.arange(0.0, 2500.5, 0.5) if np.isclose(_ref_fold(f, 1000.0), 100.0)]
    assert np.allclose(got, brute), (
        f"a brute-force sweep of [0, 2500] finds {brute}; you returned "
        f"{np.asarray(got).tolist()}"
    )
    edge = lesson.candidate_true_frequencies(0.0, 1000.0, 2500.0)
    assert np.allclose(edge, [0.0, 1000.0, 2000.0]), (
        f"at f_apparent = 0 the two branches coincide and must not be listed twice; got "
        f"{np.asarray(edge).tolist()}"
    )
    nyq = lesson.candidate_true_frequencies(500.0, 1000.0, 1600.0)
    assert np.allclose(nyq, [500.0, 1500.0]), (
        f"at exactly fs/2 the branches coincide too; expected [500, 1500], got "
        f"{np.asarray(nyq).tolist()}"
    )
    tight = lesson.candidate_true_frequencies(100.0, 1000.0, 100.0)
    # Check the LENGTH before the values. np.allclose broadcasts, so an empty result
    # compares equal to a one-element expectation and an off-by-one bound slips through.
    assert np.shape(tight) == (1,), (
        f"f_max is INCLUSIVE, so f_max = 100 keeps the 100 Hz candidate; you returned "
        f"{np.asarray(tight).tolist()} ({np.size(tight)} candidates, expected 1). An empty "
        "result means the bound was written `value < f_max` instead of `value <= f_max`"
    )
    assert np.allclose(tight, [100.0]), (
        f"f_max is inclusive and cuts the list; expected [100.0], got "
        f"{np.asarray(tight).tolist()}"
    )


def test_candidate_true_frequencies_validates():
    for bad, why in (((100.0, 0.0, 2500.0), "fs = 0"),
                     ((100.0, 1000.0, -1.0), "a negative f_max"),
                     ((600.0, 1000.0, 2500.0), "an f_apparent above fs/2"),
                     ((-10.0, 1000.0, 2500.0), "a negative f_apparent")):
        try:
            lesson.candidate_true_frequencies(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError")


# --- exercise 5: one_pole_lowpass ---------------------------------------------------------

def test_one_pole_lowpass_values():
    rng = np.random.default_rng(11)
    sig = rng.normal(0.0, 1.0, 200)
    for stages in (1, 2, 5):
        got = lesson.one_pole_lowpass(sig, 120.0, 4000.0, stages)
        want = _ref_lowpass(sig, 120.0, 4000.0, stages)
        assert np.allclose(got, want, atol=1e-12), (
            f"with stages={stages} the output differs from the reference by up to "
            f"{np.max(np.abs(np.asarray(got) - want)):.3e}. Each section is "
            "y[n] = a*y[n-1] + (1-a)*x[n] with a = exp(-2*pi*fc/fs), and a cascade feeds "
            "each stage the previous stage's whole output"
        )
    assert np.allclose(lesson.one_pole_lowpass(np.full(20, 3.5), 50.0, 1000.0, 3), 3.5), (
        "a constant input passes a low-pass unchanged; the DC gain of each section is 1"
    )


def test_one_pole_lowpass_initial_state():
    # A record that starts far from zero. If the state starts at 0 the first samples climb.
    sig = np.full(50, 5.0)
    got = np.asarray(lesson.one_pole_lowpass(sig, 20.0, 2000.0, 2))
    assert np.isclose(got[0], 5.0), (
        f"the first output sample is {got[0]!r}, not 5.0. The filter state before sample 0 "
        "is x[0], not 0 — starting at zero injects a start-up transient that the velocity "
        "path in section 12 amplifies by 1/f"
    )
    assert np.allclose(got, 5.0), (
        "a constant record must come out flat; yours is still settling, which means the "
        "state was initialised to something other than x[0]"
    )


def test_one_pole_lowpass_is_causal():
    head = np.sin(np.linspace(0.0, 6.0, 60))
    calm = np.asarray(lesson.one_pole_lowpass(np.concatenate([head, np.zeros(60)]),
                                              40.0, 2000.0, 3))
    wild = np.asarray(lesson.one_pole_lowpass(np.concatenate([head, np.full(60, 900.0)]),
                                              40.0, 2000.0, 3))
    assert np.allclose(calm[:60], wild[:60]), (
        f"changing samples 60-119 changed the first 60 by up to "
        f"{np.max(np.abs(calm[:60] - wild[:60])):.3e}. The filter is reading the future: a "
        "centred convolution, a filtfilt-style forward-backward pass, or a reversed loop"
    )
    original = np.linspace(1.0, 2.0, 30)
    before = original.copy()
    lesson.one_pole_lowpass(original, 40.0, 2000.0, 2)
    assert np.allclose(original, before), "the input array must not be modified in place"


def test_one_pole_lowpass_validates():
    for bad, why in (((np.zeros(8), 0.0, 1000.0, 1), "fc = 0"),
                     ((np.zeros(8), -5.0, 1000.0, 1), "a negative fc"),
                     ((np.zeros(8), 50.0, 0.0, 1), "fs = 0"),
                     ((np.zeros(8), 50.0, 1000.0, 0), "stages = 0"),
                     ((np.zeros((3, 8)), 50.0, 1000.0, 1), "a 2-D signal")):
        try:
            lesson.one_pole_lowpass(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError")


# --- exercise 6: lowpass_response ---------------------------------------------------------

def test_lowpass_response_values():
    gain, phase = lesson.lowpass_response(0.0, 100.0, 1000.0, 3)
    assert np.isclose(gain, 1.0) and np.isclose(phase, 0.0), (
        f"at DC: gain 1, phase 0. Got ({gain!r}, {phase!r})"
    )
    a = float(np.exp(-2.0 * np.pi * 100.0 / 1000.0))
    denom = 1.0 - a * np.exp(-1j * 2.0 * np.pi * 100.0 / 1000.0)
    want_gain, want_phase = (1.0 - a) / abs(denom), -float(np.angle(denom))
    gain, phase = lesson.lowpass_response(100.0, 100.0, 1000.0, 1)
    assert np.isclose(gain, want_gain, atol=1e-9), (
        f"one section at f = fc: expected {want_gain:.6f}, got {gain!r}. 0.7071 is the "
        "ANALOGUE 1/sqrt(2); this filter is the difference equation you wrote"
    )
    assert np.isclose(phase, want_phase, atol=1e-9), (
        f"phase should be {want_phase:.6f} rad, got {phase!r}. A positive value has the sign "
        "backwards: a low-pass delays, so its phase is negative"
    )


def test_lowpass_response_cascade_and_wrapping():
    one_gain, one_phase = lesson.lowpass_response(300.0, 100.0, 1000.0, 1)
    for stages in (2, 4, 10):
        gain, phase = lesson.lowpass_response(300.0, 100.0, 1000.0, stages)
        assert np.isclose(gain, float(one_gain) ** stages, atol=1e-12), (
            f"{stages} sections multiply their gains: expected "
            f"{float(one_gain) ** stages:.9f}, got {gain!r}"
        )
        assert np.isclose(phase, stages * float(one_phase), atol=1e-12), (
            f"{stages} sections ADD their phases: expected "
            f"{stages * float(one_phase):.9f} rad, got {phase!r}. A result inside (-pi, pi] "
            "means the total was wrapped — do not wrap it, the notebook needs the whole "
            "cycles to explain the 5545 Hz row"
        )
    assert float(lesson.lowpass_response(300.0, 100.0, 1000.0, 10)[1]) < -np.pi, (
        "ten sections at 3x the corner really do exceed half a cycle of phase; a value "
        "inside (-pi, pi] means np.angle was applied after multiplying by stages"
    )
    gains, phases = lesson.lowpass_response(np.array([0.0, 100.0, 300.0]), 100.0, 1000.0, 2)
    assert np.shape(gains) == np.shape(phases) == (3,), (
        f"must work elementwise over an array of frequencies; got shapes "
        f"{np.shape(gains)} and {np.shape(phases)}"
    )


def test_lowpass_response_matches_the_filter():
    """Integration across exercises 5 and 6: the prediction must match the filter."""
    fs, fc, stages, f = 8000.0, 500.0, 3, 900.0
    n = 8000
    t = np.arange(n) / fs
    sig = np.cos(2.0 * np.pi * f * t)
    out = np.asarray(lesson.one_pole_lowpass(sig, fc, fs, stages))
    probe = np.exp(-2j * np.pi * f * np.arange(n) / fs)
    measured = abs(np.sum(out[2000:] * probe[2000:])) / abs(np.sum(sig[2000:] * probe[2000:]))
    predicted = float(np.asarray(lesson.lowpass_response(f, fc, fs, stages))[0])
    assert np.isclose(measured, predicted, rtol=2e-3), (
        f"your filter measures a gain of {measured:.6f} at {f:.0f} Hz and your "
        f"lowpass_response predicts {predicted:.6f}. The analogue formula "
        f"(1+(f/fc)^2)**(-stages/2) gives "
        f"{(1 + (f / fc) ** 2) ** (-stages / 2):.6f}, which is close enough to look right "
        "and is not the response of this difference equation"
    )


def test_lowpass_response_validates():
    for bad, why in (((100.0, 0.0, 1000.0, 1), "fc = 0"),
                     ((100.0, 100.0, 0.0, 1), "fs = 0"),
                     ((100.0, 100.0, 1000.0, 0), "stages = 0")):
        try:
            lesson.lowpass_response(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError")


# --- exercise 7: measure_gain_and_lag -----------------------------------------------------

def test_measure_gain_and_lag_values():
    fs, n, f = 4000.0, 4000, 125.0
    t = np.arange(n) / fs
    sig = np.cos(2.0 * np.pi * f * t + 0.3)
    for gain_in, delay_ms in ((0.25, 1.0), (3.0, 0.0), (1.0, 2.5)):
        out = gain_in * np.cos(2.0 * np.pi * f * (t - delay_ms / 1000.0) + 0.3)
        gain, lag = lesson.measure_gain_and_lag(sig, out, f, fs)
        assert np.isclose(gain, gain_in, atol=1e-9), (
            f"amplitude ratio {gain_in} measured as {gain!r}"
        )
        assert np.isclose(lag * 1000.0, delay_ms, atol=1e-9), (
            f"a delay of {delay_ms} ms measured as {lag * 1000:.6f} ms"
        )


def test_measure_gain_and_lag_sign():
    fs, n, f = 4000.0, 4000, 125.0
    t = np.arange(n) / fs
    sig = np.cos(2.0 * np.pi * f * t)
    late = np.cos(2.0 * np.pi * f * (t - 0.001))      # 1 ms LATE
    early = np.cos(2.0 * np.pi * f * (t + 0.001))     # 1 ms EARLY
    _, lag_late = lesson.measure_gain_and_lag(sig, late, f, fs)
    _, lag_early = lesson.measure_gain_and_lag(sig, early, f, fs)
    assert lag_late > 0 and np.isclose(lag_late * 1000.0, 1.0, atol=1e-9), (
        f"an output that arrives LATE has a POSITIVE lag; got {lag_late * 1000:.6f} ms. "
        "lag = -angle(B/A)/(2*pi*f) — the minus sign is not optional"
    )
    assert lag_early < 0 and np.isclose(lag_early * 1000.0, -1.0, atol=1e-9), (
        f"an output that arrives EARLY has a negative lag; got {lag_early * 1000:.6f} ms"
    )


def test_measure_gain_and_lag_is_single_frequency():
    fs, n, f = 4000.0, 4000, 125.0
    t = np.arange(n) / fs
    sig = np.cos(2.0 * np.pi * f * t)
    # Three loud components the measurement at f must ignore completely.
    contaminated = sig + 6.0 * np.cos(2 * np.pi * 300.0 * t) + 4.0 * np.cos(2 * np.pi * 50 * t)
    gain, lag = lesson.measure_gain_and_lag(sig, contaminated, f, fs)
    assert np.isclose(gain, 1.0, atol=1e-9), (
        f"components at 50 and 300 Hz leaked into the 125 Hz measurement (gain {gain!r}, "
        "expected 1.0). Project onto exp(-2j*pi*f*t); do not compare peaks or RMS"
    )
    assert np.isclose(lag, 0.0, atol=1e-12), f"lag should be 0.0, got {lag!r}"
    for bad, why in (((sig, sig[:-1], f, fs), "signals of different length"),
                     ((sig, sig, 0.0, fs), "f = 0"),
                     ((sig, sig, f, 0.0), "fs = 0")):
        try:
            lesson.measure_gain_and_lag(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError")


# --- exercise 8: hann_window --------------------------------------------------------------

def test_hann_window_is_periodic():
    got = np.asarray(lesson.hann_window(4))
    assert np.allclose(got, [0.0, 0.5, 1.0, 0.5]), (
        f"hann_window(4) = {got.tolist()}, expected [0, 0.5, 1, 0.5]. "
        f"{np.hanning(4).round(4).tolist()} is np.hanning(4), the SYMMETRIC window"
    )
    for n in (8, 16, 33, 64):
        win = np.asarray(lesson.hann_window(n))
        want = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
        assert np.allclose(win, want), (
            f"hann_window({n}) does not match 0.5*(1 - cos(2*pi*k/n)); the largest error is "
            f"{np.max(np.abs(win - want)):.3e}"
        )
        assert not np.allclose(win, np.hanning(n)), (
            f"hann_window({n}) is identical to np.hanning({n}), which uses n-1 in the "
            "denominator and is the wrong window for spectral analysis"
        )
    assert np.isclose(np.asarray(lesson.hann_window(128)).mean(), 0.5), (
        "the coherent gain of the periodic Hann window is exactly 0.5; yours is "
        f"{np.asarray(lesson.hann_window(128)).mean():.8f}"
    )
    # Length first: np.allclose would call an empty return equal to [0.0].
    single = lesson.hann_window(1)
    assert np.shape(single) == (1,), (
        f"hann_window(n) always returns n samples, so hann_window(1) has length 1; yours has "
        f"length {np.size(single)}. n = 1 is legal and is not the same as n < 1"
    )
    assert np.allclose(single, [0.0]), "hann_window(1) is [0.0]"
    for n in (0, -3):
        try:
            lesson.hann_window(n)
        except ValueError:
            continue
        raise AssertionError(f"n = {n} must raise ValueError")


def test_hann_window_reduces_leakage():
    """The window must actually do its job on a tone half a bin off."""
    n, fs = 1024, 1024.0
    t = np.arange(n) / fs
    sig = np.cos(2.0 * np.pi * 100.5 * t)           # exactly half a bin above bin 100
    _, plain = lesson.amplitude_spectrum(sig, fs)
    _, windowed = lesson.amplitude_spectrum(sig, fs, window=lesson.hann_window(n))
    best_plain = float(np.max(np.asarray(plain)[95:106]))
    best_win = float(np.max(np.asarray(windowed)[95:106]))
    assert np.isclose(best_plain, 2.0 / np.pi, atol=5e-3), (
        f"an unwindowed record with the tone half a bin off must read 2/pi = 0.6366 at its "
        f"tallest bin; got {best_plain:.4f}"
    )
    assert best_win > best_plain + 0.15, (
        f"the Hann window should lift the worst-case reading well above the unwindowed "
        f"{best_plain:.4f}; yours reads {best_win:.4f}. Check the coherent-gain division"
    )


# --- exercises 9 and 10: the ADC ----------------------------------------------------------

def test_adc_round_trip():
    counts = lesson.g_to_counts(np.array([0.0, 0.5, 1.0, -1.0]), 100.0, 1.0, 12)
    assert np.asarray(counts).dtype.kind == "i", (
        f"counts must be an integer array, got dtype {np.asarray(counts).dtype}"
    )
    assert np.asarray(counts).tolist() == [0, 102, 205, -205], (
        f"expected [0, 102, 205, -205], got {np.asarray(counts).tolist()}. 1 g at 100 mV/g "
        "is 0.1 V; the step of a +-1 V 12-bit converter is 1/2048 V, so 1 g is 204.8 counts"
    )
    back = np.asarray(lesson.counts_to_g(counts, 100.0, 1.0, 12))
    assert np.allclose(back, np.asarray(counts) / 2048.0 * 1000.0 / 100.0), (
        f"counts_to_g is not the exact inverse scaling; got {back.round(6).tolist()}"
    )
    # 0.625 g lands on an exact code in all three configurations below, so rounding cannot
    # blur the comparison. Doubling the sensitivity doubles the counts; so does halving the
    # range.
    loud = int(np.asarray(lesson.g_to_counts(np.array([0.625]), 200.0, 1.0, 12))[0])
    quiet = int(np.asarray(lesson.g_to_counts(np.array([0.625]), 100.0, 1.0, 12))[0])
    wide = int(np.asarray(lesson.g_to_counts(np.array([0.625]), 100.0, 2.0, 12))[0])
    assert quiet == 128, (
        f"0.625 g at 100 mV/g on a +-1 V 12-bit converter is 0.0625 V / (1/2048 V) = 128 "
        f"counts exactly; got {quiet}"
    )
    assert loud == 2 * quiet, (
        f"a sensor twice as sensitive gives twice the counts; got {loud} and {quiet}"
    )
    assert quiet == 2 * wide, (
        f"halving the range doubles the counts; got {quiet} on +-1 V and {wide} on +-2 V"
    )
    assert np.isclose(np.asarray(lesson.counts_to_g(np.array([2048]), 100.0, 1.0, 12))[0],
                      10.0), (
        "one full-scale of counts (2048) at +-1 V and 100 mV/g is 10 g"
    )


def test_adc_clips_rather_than_wraps():
    huge = lesson.g_to_counts(np.array([50.0, -50.0, 1e9]), 100.0, 1.0, 12)
    assert np.asarray(huge).tolist() == [2047, -2048, 2047], (
        f"expected [2047, -2048, 2047], got {np.asarray(huge).tolist()}. A bipolar 12-bit "
        "converter runs from -2048 to +2047: zero takes one of the codes, so the range is "
        "asymmetric. A negative or wrapped value means np.clip was not applied"
    )
    for bits, want in ((8, (127, -128)), (16, (32767, -32768))):
        rails = lesson.g_to_counts(np.array([1e6, -1e6]), 100.0, 1.0, bits)
        assert tuple(np.asarray(rails).tolist()) == want, (
            f"at {bits} bits the rails are {want}, got {tuple(np.asarray(rails).tolist())}"
        )
    # Clipping destroys shape, not level: the RMS barely moves and the crest factor collapses.
    rng = np.random.default_rng(3)
    sig = rng.normal(0.0, 0.3, 4000)
    sig[1000] = 9.0
    back = np.asarray(lesson.counts_to_g(
        lesson.g_to_counts(sig, 100.0, 0.2, 12), 100.0, 0.2, 12))
    assert np.max(np.abs(back)) < 2.1, (
        f"a +-0.2 V range at 100 mV/g cannot represent more than 2 g, so the 9 g sample must "
        f"clip; your chain read back {np.max(np.abs(back)):.3f} g"
    )
    for fn in (lesson.g_to_counts, lesson.counts_to_g):
        for bad, why in (((np.zeros(3), 0.0, 1.0, 12), "sensitivity = 0"),
                         ((np.zeros(3), -100.0, 1.0, 12), "a negative sensitivity"),
                         ((np.zeros(3), 100.0, 0.0, 12), "full_scale_v = 0"),
                         ((np.zeros(3), 100.0, 1.0, 1), "n_bits = 1")):
            try:
                fn(*bad)
            except ValueError:
                continue
            raise AssertionError(f"{fn.__name__}: {why} must raise ValueError")


# --- exercise 11: one_pole_highpass -------------------------------------------------------

def test_one_pole_highpass_values():
    rng = np.random.default_rng(5)
    sig = rng.normal(0.0, 1.0, 300) + 4.0
    for stages in (1, 2, 4):
        got = lesson.one_pole_highpass(sig, 8.0, 2000.0, stages)
        want = _ref_highpass(sig, 8.0, 2000.0, stages)
        assert np.allclose(got, want, atol=1e-12), (
            f"with stages={stages} the output differs from the reference by up to "
            f"{np.max(np.abs(np.asarray(got) - want)):.3e}. Each section is "
            "y[n] = a*(y[n-1] + x[n] - x[n-1]) with y[0] = 0"
        )


def test_one_pole_highpass_blocks_dc_and_passes_tone():
    flat = lesson.one_pole_highpass(np.full(100, 6.25), 5.0, 1000.0, 3)
    assert np.allclose(flat, 0.0, atol=1e-12), (
        f"a constant input must be removed exactly, leaving zeros; the largest value left is "
        f"{np.max(np.abs(np.asarray(flat))):.3e}. A slowly decaying tail means y[0] was set "
        "to x[0] instead of 0"
    )
    fs = 2000.0
    t = np.arange(4000) / fs
    tone = np.cos(2.0 * np.pi * 200.0 * t)
    passed = np.asarray(lesson.one_pole_highpass(tone + 3.0, 2.0, fs, 2))
    tail = passed[1000:]
    assert np.isclose(np.sqrt(np.mean(tail ** 2)), np.sqrt(0.5), rtol=0.02), (
        f"a 200 Hz tone must survive a 2 Hz high-pass essentially intact (RMS 0.7071); the "
        f"settled part of your output has RMS {np.sqrt(np.mean(tail ** 2)):.4f}"
    )
    settled = passed[3000:]      # 0.5 s, a whole number of cycles, transient long gone
    assert abs(float(np.mean(settled))) < 1e-3, (
        f"the DC offset of 3.0 should be gone; the settled mean is {np.mean(settled):.6f}"
    )
    for bad, why in (((np.zeros(8), 0.0, 1000.0, 1), "fc = 0"),
                     ((np.zeros(8), 5.0, 0.0, 1), "fs = 0"),
                     ((np.zeros(8), 5.0, 1000.0, 0), "stages = 0"),
                     ((np.zeros((2, 8)), 5.0, 1000.0, 1), "a 2-D signal")):
        try:
            lesson.one_pole_highpass(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError")


# --- exercise 12: integrate_trapezoid -----------------------------------------------------

def test_integrate_trapezoid_values():
    got = lesson.integrate_trapezoid(np.array([0.0, 2.0, 2.0, 2.0]), 2.0)
    assert np.allclose(got, [0.0, 0.5, 1.5, 2.5]), (
        f"expected [0, 0.5, 1.5, 2.5], got {np.asarray(got).tolist()}. [0, 1, 2, 3] is "
        "np.cumsum(a)/fs — the rectangular rule, which over-counts the first step"
    )
    # A linear ramp: the trapezoid rule integrates it EXACTLY, the rectangular rule does not.
    fs = 100.0
    n = 400
    ramp = np.arange(n) / fs
    exact = 0.5 * ramp ** 2
    got = np.asarray(lesson.integrate_trapezoid(ramp, fs))
    assert np.allclose(got, exact, atol=1e-12), (
        f"the trapezoid rule integrates a straight line exactly; the largest error in yours "
        f"is {np.max(np.abs(got - exact)):.3e}. np.cumsum(a)/fs would be off by "
        f"{np.max(np.abs(np.cumsum(ramp) / fs - exact)):.4f}"
    )
    # Length first, then values: np.allclose broadcasts a size-0 result against a
    # one-element expectation and would call an empty return correct.
    one = lesson.integrate_trapezoid(np.array([5.0]), 10.0)
    assert np.shape(one) == (1,), (
        f"the output must be the same length as the input; a one-sample record gives one "
        f"sample, and you returned {np.size(one)}. Build the result with np.zeros_like(a) "
        "and fill in [1:], rather than returning only the per-interval areas"
    )
    assert np.allclose(one, [0.0]), (
        "a one-sample record integrates to [0.0]; there is no interval to integrate over"
    )
    empty = lesson.integrate_trapezoid(np.zeros(0), 10.0)
    assert np.shape(empty) == (0,), (
        f"an empty record integrates to an EMPTY result, not an error and not a [0.0]; you "
        f"returned {np.asarray(empty).tolist()}"
    )
    assert float(np.asarray(lesson.integrate_trapezoid(np.full(9, 3.0), 3.0))[0]) == 0.0, (
        "the result always starts at zero: the constant of integration is chosen as v[0] = 0"
    )


def test_integrate_trapezoid_validates():
    for bad, why in (((np.zeros(8), 0.0), "fs = 0"),
                     ((np.zeros(8), -100.0), "a negative fs"),
                     ((np.zeros((2, 8)), 100.0), "a 2-D signal")):
        try:
            lesson.integrate_trapezoid(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError")


# --- integration cases on the lesson's own record -----------------------------------------

def test_the_alias_is_real():
    """Undersampling the notebook's own signal must inflate the gear line, measurably."""
    true_amp = dict((f, a) for f, a, _ in lesson.TONES)[425.0]
    rate, factor = lesson.RATES[0]
    freqs, amps = lesson.amplitude_spectrum(
        lesson.decimate_naive(lesson.TRUE_G, factor), rate)
    _, naive = lesson.peak_in_band(freqs, amps, 415.0, 435.0)
    assert naive > 2.0 * true_amp, (
        f"sampled at {rate:.0f} Hz with no anti-alias filter the 425 Hz line reads "
        f"{naive:.4f} g against a true {true_amp:.3f} g. It should be more than double: the "
        f"{5545.0:.0f} Hz resonance folds onto it in phase. A reading near the truth means "
        "the spectrum scaling or the decimation is wrong"
    )
    # And the fold must be predicted, not discovered by eye.
    assert np.isclose(float(lesson.alias_frequency(5545.0, rate)), 425.0), (
        f"alias_frequency(5545, {rate:.0f}) must be 425.0 — that is why the two land on top "
        "of each other"
    )
    # A different rate moves the alias and leaves the real line alone.
    other_rate, other_factor = lesson.RATES[1]
    o_freqs, o_amps = lesson.amplitude_spectrum(
        lesson.decimate_naive(lesson.TRUE_G, other_factor), other_rate)
    _, clean = lesson.peak_in_band(o_freqs, o_amps, 415.0, 435.0)
    assert abs(clean / true_amp - 1.0) < 0.05, (
        f"at {other_rate:.0f} Hz the resonance folds to "
        f"{float(lesson.alias_frequency(5545.0, other_rate)):.0f} Hz and the gear line should "
        f"read close to {true_amp:.3f} g; yours reads {clean:.4f} g"
    )


def test_conditioning_recovers_the_amplitude():
    """Filter, decimate, compensate: the gear line must come back to within a percent."""
    true_amp = dict((f, a) for f, a, _ in lesson.TONES)[425.0]
    rate, factor = lesson.RATES[0]
    filtered = lesson.one_pole_lowpass(lesson.TRUE_G, lesson.FC_AA, lesson.FS_TRUE,
                                       lesson.AA_STAGES)
    freqs, amps = lesson.amplitude_spectrum(lesson.decimate_naive(filtered, factor), rate)
    _, droopy = lesson.peak_in_band(freqs, amps, 415.0, 435.0)
    gain = float(np.asarray(lesson.lowpass_response(425.0, lesson.FC_AA, lesson.FS_TRUE,
                                                    lesson.AA_STAGES))[0])
    assert 0.6 < gain < 0.85, (
        f"the four-stage filter's gain at 425 Hz should be around 0.73, not {gain:.4f}"
    )
    assert droopy < 0.9 * true_amp, (
        f"the filtered line should be visibly LOW before compensation ({droopy:.4f} g "
        f"against {true_amp:.3f} g); the anti-alias filter is not free"
    )
    recovered = droopy / gain
    assert abs(recovered / true_amp - 1.0) < 0.01, (
        f"dividing by the predicted gain should recover {true_amp:.3f} g to within 1%; you "
        f"get {recovered:.4f} g ({100 * (recovered / true_amp - 1):+.2f}%). If the gain is "
        "the analogue formula rather than this filter's own, the residual will not close"
    )


def test_the_high_pass_belongs_after_the_integration():
    """The ordering result the notebook is built around, on the notebook's own record."""
    expected = np.sqrt(sum((a * lesson.G_N / (2.0 * np.pi * f)) ** 2
                           for f, a, _ in lesson.TONES) / 2.0) * 1000.0
    accel = lesson.TRUE_G * lesson.G_N
    after = np.asarray(lesson.one_pole_highpass(
        lesson.integrate_trapezoid(accel, lesson.FS_TRUE) * 1000.0,
        lesson.FC_HP, lesson.FS_TRUE, lesson.HP_STAGES))
    before = np.asarray(lesson.integrate_trapezoid(
        lesson.one_pole_highpass(accel, lesson.FC_HP, lesson.FS_TRUE, lesson.HP_STAGES),
        lesson.FS_TRUE)) * 1000.0
    rms_after = float(np.sqrt(np.mean(after ** 2)))
    rms_before = float(np.sqrt(np.mean(before ** 2)))
    assert abs(rms_after / expected - 1.0) < 0.03, (
        f"integrating and THEN high-passing should land within 3% of the {expected:.3f} mm/s "
        f"the components imply; yours gives {rms_after:.3f} mm/s"
    )
    assert rms_before > 2.0 * rms_after, (
        f"high-passing the acceleration FIRST should be much worse ({rms_before:.3f} mm/s "
        f"against {rms_after:.3f} mm/s). If the two orderings agree, the high-pass is not "
        "starting from y[0] = 0, or the integration is not accumulating"
    )


RUBRIC = [
    (test_amplitude_spectrum_scaling, 5,
     "amps = 2*abs(np.fft.rfft(x))/n, and freqs = np.fft.rfftfreq(n, d=1/fs)"),
    (test_amplitude_spectrum_dc_and_nyquist, 4,
     "bin 0 always, and the last bin when n is even, are NOT doubled — they have no "
     "negative-frequency twin"),
    (test_amplitude_spectrum_window, 4,
     "multiply by the window, then divide every bin by window.mean(), the coherent gain"),
    (test_amplitude_spectrum_validates, 2,
     "fs <= 0, a 2-D input, and a window of the wrong length are each a ValueError"),
    (test_peak_in_band, 3,
     "mask with (freqs >= f_lo) & (freqs <= f_hi), argmax inside the mask, and return "
     "(frequency, amplitude) — not the index"),
    (test_peak_in_band_validates, 2,
     "f_lo > f_hi and an empty band are both ValueError; do not return a zero"),
    (test_alias_frequency_below_nyquist, 3,
     "abs(f - fs*round(f/fs)) leaves anything at or below fs/2 untouched"),
    (test_alias_frequency_above_nyquist, 5,
     "the fold is abs(f - fs*round(f/fs)); f % fs agrees below fs/2 and disagrees above it"),
    (test_alias_frequency_array_and_validates, 3,
     "take abs(f) first so a negative frequency folds like its magnitude; fs <= 0 is a "
     "ValueError"),
    (test_candidate_true_frequencies, 5,
     "collect k*fs - f_apparent and k*fs + f_apparent for k = 0, 1, 2, ... keeping what "
     "lands in [0, f_max]; deduplicate and sort"),
    (test_candidate_true_frequencies_validates, 2,
     "fs <= 0, f_max < 0, and an f_apparent outside [0, fs/2] are each a ValueError"),
    (test_one_pole_lowpass_values, 5,
     "y[n] = a*y[n-1] + (1-a)*x[n] with a = exp(-2*pi*fc/fs); a cascade feeds each stage "
     "the previous stage's whole output"),
    (test_one_pole_lowpass_initial_state, 4,
     "the state before sample 0 is x[0], so y[0] == x[0] exactly"),
    (test_one_pole_lowpass_is_causal, 4,
     "the recursion must run forwards only, and must not write into the input array"),
    (test_one_pole_lowpass_validates, 2,
     "fc <= 0, fs <= 0, stages < 1 and a non-1-D input are each a ValueError"),
    (test_lowpass_response_values, 4,
     "with d = 1 - a*exp(-1j*w): gain = ((1-a)/abs(d))**stages, phase = -stages*angle(d)"),
    (test_lowpass_response_cascade_and_wrapping, 4,
     "take np.angle of ONE section and multiply by stages; do not wrap the total"),
    (test_lowpass_response_matches_the_filter, 5,
     "if the prediction and the measurement differ by a few percent you have used the "
     "analogue 1/sqrt(1+(f/fc)^2) instead of this difference equation's own response"),
    (test_lowpass_response_validates, 2,
     "fc <= 0, fs <= 0 and stages < 1 are each a ValueError"),
    (test_measure_gain_and_lag_values, 4,
     "project both signals onto exp(-2j*pi*f*arange(n)/fs) and take the ratio"),
    (test_measure_gain_and_lag_sign, 4,
     "lag = -np.angle(B/A)/(2*pi*f); a delay is a NEGATIVE phase and a POSITIVE lag"),
    (test_measure_gain_and_lag_is_single_frequency, 3,
     "use the complex projection, not peak values or RMS, so other components cancel"),
    (test_hann_window_is_periodic, 4,
     "0.5*(1 - cos(2*pi*np.arange(n)/n)) — the denominator is n, not n-1, and np.hanning "
     "is the wrong function here"),
    (test_hann_window_reduces_leakage, 3,
     "the window only helps if amplitude_spectrum divides by its mean"),
    (test_adc_round_trip, 4,
     "volts = g*sensitivity/1000; step = full_scale_v / 2**(n_bits-1); counts = "
     "round(volts/step)"),
    (test_adc_clips_rather_than_wraps, 4,
     "np.clip(counts, -half, half - 1): the positive rail is one code lower than the "
     "negative one"),
    (test_one_pole_highpass_values, 5,
     "y[n] = a*(y[n-1] + x[n] - x[n-1]) with a = exp(-2*pi*fc/fs) and y[0] = 0"),
    (test_one_pole_highpass_blocks_dc_and_passes_tone, 4,
     "starting the state at 0 (not at x[0]) is what makes a constant come out as exactly "
     "zero"),
    (test_integrate_trapezoid_values, 4,
     "v[0] = 0 and v[n] = v[n-1] + (a[n-1] + a[n])/(2*fs); np.cumsum(a)/fs is the "
     "rectangular rule and is a different answer"),
    (test_integrate_trapezoid_validates, 2,
     "fs <= 0 and a non-1-D input are each a ValueError"),
    (test_the_alias_is_real, 5,
     "if the 425 Hz line does not inflate, check that decimate_naive is being fed the "
     "unfiltered record and that amplitude_spectrum doubles the non-DC bins"),
    (test_conditioning_recovers_the_amplitude, 5,
     "the compensation is a division by lowpass_response(...)[0] at that exact frequency"),
    (test_the_high_pass_belongs_after_the_integration, 4,
     "if both orderings agree, the high-pass state is not starting at zero"),
]
