# %% [markdown]
# # T03-L01 · Byte-level BPE from scratch, and the token tax — REFERENCE SOLUTION
#
# Every stub filled in. Runs top to bottom in a fresh interpreter and prints real measured
# numbers. Self-check answers are in the comment block at the foot of this file.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import sys
import time
from typing import Callable, NamedTuple

_LESSON_T0 = time.perf_counter()
print("python", sys.version.split()[0])

BOUNDARY = 256          # the symbol between two words. Never merged across.
FIRST_MERGE_ID = 257    # the k-th merge is given id 257 + k
MIN_COUNT = 2           # a pair that occurs once buys no compression
VOCAB_SIZE = 768        # the budget every table in this lesson is measured at
_WHITESPACE = frozenset(b" \t\n\r\f\v")

_FAILED_CHECKS: list[str] = []


def _try(label: str, check: Callable[[], None]) -> None:
    """Run a check without derailing the notebook; record anything that actually failed.

    Nothing is swallowed: the `__main__` block at the foot of this file exits non-zero if the
    list is not empty, which is what stops tools/execute.py from reporting PASS on a reference
    implementation that is quietly wrong.
    """
    try:
        check()
    except NotImplementedError:
        print(f"{label}: not implemented yet — fill in the stub above, then re-run this cell.")
    except AssertionError as exc:
        _FAILED_CHECKS.append(label)
        print(f"{label}: FAILED — {exc}")
    except Exception as exc:  # a half-finished implementation raising something else
        _FAILED_CHECKS.append(label)
        print(f"{label}: raised {type(exc).__name__}: {exc}")


# %% [markdown]
# ## 1. The corpus and the samples

# %%
ENGLISH_CORPUS = """\
A tokenizer is the first thing a language model does and the last thing anyone thinks about.
It takes the text you wrote and turns it into a list of integers, and the model never sees anything else.
The model does not read characters, and the model does not read words. The model reads tokens.
Everything the model knows about your text arrives through that list of integers.
A tokenizer that splits your text badly will cost you money on every request you ever make.
The tokenizer is trained once and then frozen, and every user of the model lives with that decision.
Training a tokenizer means choosing which sequences of bytes deserve a number of their own.
The sequences that appear often in the training text get short names, and the rest are spelled out.
This is the whole idea, and the rest of the work is bookkeeping.
Byte pair encoding starts from the raw bytes and invents a new symbol for the commonest adjacent pair.
It counts every adjacent pair in the text, picks the pair that occurs most often, and replaces it everywhere.
Then it counts again, picks again, and replaces again, until it has made as many merges as you asked for.
Each merge adds one symbol to the vocabulary, and each merge makes the text a little shorter.
The merge list is the whole model. There is nothing else to save and nothing else to ship.
A vocabulary is just the merge list read from the top, and the order of the merges is what makes it reproducible.
If two pairs occur the same number of times, something has to break the tie, and that rule is part of the model.
Break the tie differently and you get a different vocabulary from the same text, which is a bug you will find much later.
Starting from bytes rather than from characters is what makes the whole thing safe.
There are only two hundred and fifty six byte values, and every possible file is made of them.
A vocabulary that starts from bytes can therefore spell any text at all, including text it has never seen.
No character can ever fall outside the vocabulary, so there is no need for an unknown token.
A tokenizer that starts from characters must decide what to do with a character it has never met.
Usually it replaces that character with a special symbol, and the information in it is simply gone.
The model cannot recover what the tokenizer threw away, because the model never sees the original text.
Starting from bytes means the worst case is a long list of tokens, not a lost one.
A long list of tokens is expensive. A lost token is wrong. Those are very different failures.
The cost of a tokenizer is not paid by the person who trained it.
It is paid by every user of the model, on every request, forever, in tokens.
Two people can write the same sentence and be charged very different amounts for it.
The difference is not in what they wrote but in which text the tokenizer happened to be trained on.
Text that looks like the training text gets short tokens, and text that does not gets long ones.
A word that appeared ten thousand times in the training text probably has a token of its own.
A word that never appeared at all is spelled out one byte at a time.
This is not a flaw in byte pair encoding. It is exactly what byte pair encoding was asked to do.
The flaw, if there is one, is in the text it was trained on, and in who chose that text.
A model with a fixed context window holds a fixed number of tokens, not a fixed amount of meaning.
If your language costs more tokens per sentence, you fit less of your document into the same window.
You pay more for the same request, you wait longer for the same answer, and you get less context.
None of those three costs is visible in the model weights, and all three are decided by the tokenizer.
Measuring this is easy and almost nobody does it. You count the tokens and you divide.
Count the characters, count the bytes, count the tokens, and put the three numbers side by side.
The ratio of tokens to characters is the number that travels between languages.
Raw token counts do not travel, because a translated sentence is never the same length as the original.
Dividing by characters removes most of that difference and leaves the part the tokenizer caused.
The encoding itself is the floor. UTF-8 spends one byte on an ASCII character and more on everything else.
A tokenizer that has learned nothing about a script can do no better than one token per byte.
That floor is the worst case, and a script the tokenizer has never seen sits exactly on it.
Every merge the tokenizer learned for a script lifts that script off the floor a little.
So the fix is not a better algorithm. The fix is training text that contains the languages you serve.
You can prove that to yourself in a minute by training the same algorithm on a different corpus.
Train it on English and measure five scripts, then train it on all five and measure again.
The algorithm did not change, the code did not change, and the numbers move a long way.
That is the whole argument, and it is worth being able to run rather than to quote.
A merge that never fires is a wasted slot in the vocabulary, and the vocabulary is a fixed size.
Every slot you spend on one language is a slot you did not spend on another.
This is why vocabulary size and training mixture are the two decisions that matter most.
The rest of the tokenizer is an implementation detail, and a fast one is not a fair one.
Speed is easy to measure and fairness is easy to ignore, so speed is what gets reported.
A tokenizer is a small program with a very long shadow, and the shadow falls unevenly.
The merge loop itself is about thirty lines of code and you can write it in an afternoon.
Reading those thirty lines is worth more than reading a paper about them.
Write the counter, write the merge, write the loop, and then measure what you built.
Measure it on text you did not write and in a script you do not read.
The number you get back is the number your users would have paid.
Do that once and you will never again look at a token count as a neutral fact.
The token count is a choice, and somebody made it on your behalf.
Now you know how to make it yourself, and how to check the one you were given.
"""

SAMPLES = {
    "Latin (English)":    "Knowledge is power.",
    "Cyrillic (Russian)": "Знание — сила.",
    "Devanagari (Hindi)": "ज्ञान ही शक्ति है।",
    "Arabic":             "المعرفة قوة.",
    "Han (Chinese)":      "知识就是力量。",
}

# %% [markdown]
# ## 2. Bytes are not characters

# %%
if __name__ == "__main__":
    print(f"{'script':22} {'chars':>6} {'utf-8 bytes':>12} {'bytes/char':>11}")
    for _name, _s in SAMPLES.items():
        _b = len(_s.encode("utf-8"))
        print(f"{_name:22} {len(_s):6d} {_b:12d} {_b / len(_s):11.2f}")

# %% [markdown]
# ## 3. Why start from bytes

# %%
def character_vocabulary_demo(train_text: str, test_text: str) -> dict:
    """Build a character vocabulary from `train_text`, then try to represent `test_text`."""
    vocab = set(train_text)
    lost = [c for c in test_text if c not in vocab]
    return {
        "recovered": "".join(c if c in vocab else "�" for c in test_text),
        "characters_lost": len(lost),
        "fraction_lost": len(lost) / len(test_text),
        "byte_level_recovered": test_text,
        "byte_level_tokens": len(test_text.encode("utf-8")),
    }


if __name__ == "__main__":
    _demo = character_vocabulary_demo(SAMPLES["Latin (English)"], SAMPLES["Han (Chinese)"])
    print("a character vocabulary trained on English, asked to represent Chinese:")
    print(f"  recovered        {_demo['recovered']!r}")
    print(f"  characters lost  {_demo['characters_lost']} of "
          f"{len(SAMPLES['Han (Chinese)'])}  ({_demo['fraction_lost']:.0%})")
    print("the same text through a byte-level vocabulary:")
    print(f"  recovered        {_demo['byte_level_recovered']!r}")
    print(f"  cost             {_demo['byte_level_tokens']} tokens, and nothing is lost")

# %% [markdown]
# ## 4. Pre-tokenisation (given)

# %%
def pretokenise(blob: bytes) -> list[int]:
    """Bytes of a word, then BOUNDARY, then the next word. Whitespace itself is dropped."""
    seq, in_word = [], False
    for byte in blob:
        if byte in _WHITESPACE:
            if in_word:
                seq.append(BOUNDARY)
                in_word = False
        else:
            seq.append(byte)
            in_word = True
    if in_word:
        seq.append(BOUNDARY)
    return seq


if __name__ == "__main__":
    print(pretokenise(b"the cat"), "  <- 256 is the word boundary")

# %% [markdown]
# ## 5. Exercise 1 — `get_stats`
#
# One scan, one variable holding the previous symbol. A pair is counted only when neither side
# is BOUNDARY. Counting overlaps: after counting (prev, cur) we move on by ONE, so the middle
# symbol of a run is the right half of one pair and the left half of the next.

# %%
def get_stats(seq: list[int]) -> dict[tuple[int, int], int]:
    """Count adjacent pairs in `seq`, skipping any pair that touches BOUNDARY."""
    counts: dict[tuple[int, int], int] = {}
    prev = seq[0] if seq else BOUNDARY
    for i in range(1, len(seq)):
        cur = seq[i]
        if prev != BOUNDARY and cur != BOUNDARY:
            key = (prev, cur)
            counts[key] = counts.get(key, 0) + 1
        prev = cur
    return counts


def _check_get_stats() -> None:
    got = get_stats(pretokenise(b"ab ab"))
    assert got == {(97, 98): 2}, (
        f"get_stats on 'ab ab' gave {got}, expected {{(97, 98): 2}} — any entry mentioning "
        f"{BOUNDARY} means you are counting across the word boundary; skip the pair entirely "
        "rather than recording it as zero."
    )
    assert get_stats([7, 7, 7]) == {(7, 7): 2}, (
        "[7,7,7] holds TWO (7,7) pairs — counting overlaps. If you got 1 you are skipping "
        "ahead by two after a match, which is the rule for merging, not for counting."
    )
    assert get_stats([]) == {} and get_stats([5]) == {}, (
        "a sequence shorter than two symbols has no pairs — guard the loop instead of "
        "indexing seq[i+1] past the end."
    )
    assert get_stats([BOUNDARY, BOUNDARY, 4]) == {}, (
        "two boundaries in a row still yield no pair, and neither does a boundary followed "
        "by a symbol."
    )
    big = get_stats(pretokenise(ENGLISH_CORPUS.encode("utf-8")))
    assert big, "no pairs at all on real text — is the function returning before the loop?"
    print(f"exercise 1 looks right: {len(big)} distinct pairs in the corpus, "
          f"commonest occurs {max(big.values())} times")


# %%
if __name__ == "__main__":
    _try("exercise 1", _check_get_stats)

# %% [markdown]
# ## 6. Exercise 2 — `merge`
#
# Two cursors in one pass. On a match, append the new id and step the read cursor by two; that
# `i += 2` is the entire difference between this and the commonest wrong answer.

# %%
def merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping left-to-right occurrence of `pair` with `new_id`."""
    a, b = pair
    out: list[int] = []
    i, n = 0, len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _check_merge() -> None:
    triple = merge([7, 7, 7], (7, 7), 300)
    assert triple == [300, 7], (
        f"[7,7,7] merged on (7,7) must be [300,7], got {triple} — after a match advance the "
        "cursor by TWO; advancing by one consumes the middle symbol twice."
    )
    assert merge([1, 2, 3], (8, 9), 300) == [1, 2, 3], (
        "a pair that never occurs must leave the sequence untouched, same length and all."
    )
    assert merge([1, 2, 1, 2], (1, 2), 300) == [300, 300], (
        "two separate occurrences must both merge — do not stop at the first match."
    )
    assert merge([5, 1, 2], (1, 2), 300) == [5, 300], (
        "symbols before a match must be copied through, in order."
    )
    assert merge([1, 2], (1, 2), 300) == [300], (
        "a match at the very end must still fire — the guard is i + 1 < len(seq), and it "
        "protects the lookahead; it does not exclude the final pair."
    )
    assert merge([300, 301, 300, 301], (300, 301), 500) == [500, 500], (
        "ids above 255 must merge like any other: symbols are ints, not bytes."
    )
    print("exercise 2 looks right: overlap, tail, absent pair and multi-byte ids all handled")


# %%
if __name__ == "__main__":
    _try("exercise 2", _check_merge)

# %% [markdown]
# ## 7. Exercise 3 — `best_pair`
#
# `n > best_n` keeps the first of an equal run from being displaced; `pair < best` at equal
# counts is the tie-break that makes the vocabulary reproducible. Comparing the tuples is what
# makes the answer independent of the order the dict happens to iterate in.

# %%
def best_pair(stats: dict[tuple[int, int], int]) -> tuple[int, int, int] | None:
    """Return `(a, b, count)` for the winning pair, or `None` if `stats` is empty."""
    best: tuple[int, int] | None = None
    best_n = 0
    for pair, n in stats.items():
        if best is None or n > best_n or (n == best_n and pair < best):
            best, best_n = pair, n
    return (best[0], best[1], best_n) if best is not None else None


def _check_best_pair() -> None:
    assert best_pair({(1, 2): 3, (4, 5): 9, (6, 7): 5}) == (4, 5, 9), (
        "the highest count must win outright when nothing ties — check the comparison is > "
        "and not >=, which lets a later equal-count pair displace the leader."
    )
    tie = best_pair({(9, 9): 4, (2, 7): 4})
    assert tie == (2, 7, 4), (
        f"a tie between (9,9) and (2,7) must go to (2,7), got {tie} — compare the tuples "
        "themselves; whichever one you happened to meet first is not a rule."
    )
    assert best_pair({}) is None, (
        "an empty stats dict must return None, not a pair of zeros — train_bpe relies on "
        "that signal to stop."
    )
    assert best_pair({(260, 261): 1}) == (260, 261, 1), "a single entry is its own winner"
    tied = [(3, 4), (17, 18), (41, 42), (88, 89), (150, 151), (260, 261), (301, 302)]
    answers = {best_pair({p: 9 for p in order})
               for order in (tied, list(reversed(tied)), tied[3:] + tied[:3],
                             tied[1::2] + tied[0::2])}
    assert answers == {(3, 4, 9)}, (
        f"the same seven tied pairs produced {sorted(answers)} — the only thing that changed "
        "was insertion order, so you are returning the first maximum the dict handed you "
        "rather than the smallest (a, b)."
    )
    print("exercise 3 looks right: highest count wins, ties resolve to the smaller pair, and "
          "the answer does not depend on insertion order")


# %%
if __name__ == "__main__":
    _try("exercise 3", _check_best_pair)

# %% [markdown]
# ## 8. Exercise 4 — `train_bpe`
#
# The stop rule is tested before the merge is recorded, so a pair seen once never enters the
# vocabulary. The merge is applied before the next count, which is what lets merge k+1 consume
# the symbol merge k created.

# %%
class BPEModel(NamedTuple):
    """A trained tokenizer. The merge list is the whole model; `vocab` is derived from it."""

    merges: dict[tuple[int, int], int]
    order: list[tuple[int, int, int, int]]
    vocab: dict[int, bytes]


def build_vocab(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
    """Expand every id to the bytes it stands for."""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab[BOUNDARY] = b""
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


def train_bpe(text: str, vocab_size: int) -> BPEModel:
    """Train byte-level BPE on `text` and return the trained model."""
    if vocab_size < FIRST_MERGE_ID:
        raise ValueError(
            f"vocab_size must be at least {FIRST_MERGE_ID}: the 256 byte values and the "
            f"boundary symbol are already spoken for, so {vocab_size} leaves no room."
        )
    seq = pretokenise(text.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    order: list[tuple[int, int, int, int]] = []
    for k in range(vocab_size - FIRST_MERGE_ID):
        winner = best_pair(get_stats(seq))
        if winner is None or winner[2] < MIN_COUNT:
            break
        a, b, count = winner
        new_id = FIRST_MERGE_ID + k
        merges[(a, b)] = new_id
        order.append((a, b, new_id, count))
        seq = merge(seq, (a, b), new_id)
    return BPEModel(merges=merges, order=order, vocab=build_vocab(merges))


def _check_train_bpe() -> None:
    tiny = train_bpe("ab ab ab", 258)
    assert tiny.order == [(97, 98, 257, 3)], (
        f"train_bpe('ab ab ab', 258).order gave {tiny.order}, expected [(97, 98, 257, 3)] — "
        f"one merge, numbered {FIRST_MERGE_ID}, recording the count it was chosen on."
    )
    assert tiny.vocab[257] == b"ab", (
        "vocab[257] should expand to b'ab' — call build_vocab(merges) after the loop."
    )
    three = train_bpe("abcabcabc abcabcabc", 259)
    assert [row[:3] for row in three.order] == [(97, 98, 257), (257, 99, 258)], (
        f"expected the second merge to consume the first (ids {FIRST_MERGE_ID} then "
        f"{FIRST_MERGE_ID + 1}), got {[row[:3] for row in three.order]} — apply each merge to "
        "the sequence BEFORE counting again."
    )
    assert train_bpe("abcdef ghijkl", 300).order == [], (
        "no pair occurs twice here, so no merge is worth making however many are asked for."
    )
    capped = train_bpe(ENGLISH_CORPUS, 300)
    assert len(capped.order) == 300 - FIRST_MERGE_ID, (
        f"a vocab_size of 300 buys exactly {300 - FIRST_MERGE_ID} merges, got "
        f"{len(capped.order)}."
    )
    assert train_bpe(ENGLISH_CORPUS, FIRST_MERGE_ID).order == [], (
        f"a vocab_size of exactly {FIRST_MERGE_ID} buys zero merges — a valid, empty model."
    )
    try:
        train_bpe("anything", 256)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"vocab_size below {FIRST_MERGE_ID} must raise ValueError."
        )
    print("exercise 4 looks right: ids numbered from 257, merges applied before recounting, "
          "stop rule and vocab_size guard all behaving")


# %%
if __name__ == "__main__":
    _try("exercise 4", _check_train_bpe)

# %% [markdown]
# ## 9. Exercise 5 — `encode` and `decode`
#
# Encoding replays the merge list in training order over each word. Whitespace bytes are
# emitted as themselves, which is what makes the round trip exact: BOUNDARY dropped the
# whitespace during training, so nothing in the merge list can ever contain one.

# %%
def _encode_word(word_bytes: list[int], model: BPEModel) -> list[int]:
    """Apply every merge, in training order, to the bytes of a single word."""
    ids = list(word_bytes)
    for pair, new_id in model.merges.items():
        if len(ids) < 2:
            break
        ids = merge(ids, pair, new_id)
    return ids


def encode(text: str, model: BPEModel) -> list[int]:
    """Encode `text` to a list of token ids, applying `model.merges` in training order."""
    out: list[int] = []
    word: list[int] = []
    for byte in text.encode("utf-8"):
        if byte in _WHITESPACE:
            if word:
                out.extend(_encode_word(word, model))
                word = []
            out.append(byte)
        else:
            word.append(byte)
    if word:
        out.extend(_encode_word(word, model))
    return out


def decode(ids: list[int], model: BPEModel) -> str:
    """Expand `ids` back to text through `model.vocab`."""
    blob = b"".join(model.vocab[i] for i in ids)
    return blob.decode("utf-8", errors="replace")


def _check_encode_decode() -> None:
    toy = train_bpe("ab ab ab", 258)
    assert encode("ab", toy) == [257], (
        "the single merge of this toy model should turn 'ab' into one token."
    )
    spaced = encode("ab ab", toy)
    assert spaced == [257, 32, 257], (
        f"'ab ab' should encode to [257, 32, 257], got {spaced} — the space is emitted as its "
        f"own byte id (32), and {BOUNDARY} must never reach the output."
    )
    model = train_bpe(ENGLISH_CORPUS, VOCAB_SIZE)
    assert BOUNDARY not in encode(ENGLISH_CORPUS[:400], model), (
        f"{BOUNDARY} appeared in encode's output — it separates words during TRAINING only."
    )
    for name, sample in SAMPLES.items():
        got = decode(encode(sample, model), model)
        assert got == sample, (
            f"round trip failed for {name}: encode then decode gave {got!r}, expected "
            f"{sample!r}."
        )
    assert decode(encode("", model), model) == "", "the empty string round-trips to itself"
    english = encode(SAMPLES["Latin (English)"], model)
    english_bytes = len(SAMPLES["Latin (English)"].encode("utf-8"))
    assert len(english) < english_bytes, (
        f"English encoded to {len(english)} tokens, no shorter than its {english_bytes} bytes."
    )
    print(f"exercise 5 looks right: all five scripts round-trip exactly, and English "
          f"compresses to {len(english)} tokens from {english_bytes} bytes")


# %%
if __name__ == "__main__":
    _try("exercise 5", _check_encode_decode)

# %% [markdown]
# ## 10. Exercise 6 — measure the tax

# %%
def token_tax(model: BPEModel, samples: dict[str, str]) -> list[dict]:
    """One row per sample, in the order `samples` gives them."""
    rows = []
    for script, text in samples.items():
        chars = len(text)
        utf8_bytes = len(text.encode("utf-8"))
        tokens = len(encode(text, model))
        rows.append({
            "script": script,
            "text": text,
            "chars": chars,
            "utf8_bytes": utf8_bytes,
            "tokens": tokens,
            "tokens_per_char": tokens / chars,
            "bytes_per_char": utf8_bytes / chars,
        })
    return rows


def _check_token_tax() -> None:
    model = train_bpe(ENGLISH_CORPUS, VOCAB_SIZE)
    rows = token_tax(model, SAMPLES)
    assert [r["script"] for r in rows] == list(SAMPLES), (
        "token_tax must return one row per sample, in the order they were given."
    )
    required = {"script", "text", "chars", "utf8_bytes", "tokens",
                "tokens_per_char", "bytes_per_char"}
    assert set(rows[0]) == required, (
        f"row keys are {sorted(rows[0])}, expected exactly {sorted(required)}"
    )
    latin = rows[0]
    assert latin["chars"] == len(SAMPLES["Latin (English)"]), "chars is len(text)"
    assert latin["utf8_bytes"] == len(SAMPLES["Latin (English)"].encode("utf-8")), (
        "utf8_bytes is len(text.encode('utf-8')), not len(text)"
    )
    assert abs(latin["tokens_per_char"] - latin["tokens"] / latin["chars"]) < 1e-9, (
        "tokens_per_char is tokens divided by CHARACTERS, not by bytes."
    )
    assert abs(latin["bytes_per_char"] - latin["utf8_bytes"] / latin["chars"]) < 1e-9, (
        "bytes_per_char is utf8_bytes divided by chars"
    )
    print(f"exercise 6 looks right: {len(rows)} rows measured against a "
          f"{len(model.order)}-merge vocabulary")


# %%
if __name__ == "__main__":
    _try("exercise 6", _check_token_tax)

# %% [markdown]
# ### The measured table

# %%
def print_tax_table(rows: list[dict], baseline: str = "Latin (English)") -> None:
    """Print the measured table, with every figure relative to `baseline`."""
    base = next(r for r in rows if r["script"] == baseline)["tokens_per_char"]
    print(f"{'script':22} {'chars':>6} {'bytes':>6} {'tokens':>7} {'tok/char':>9} "
          f"{'vs base':>8} {'tokens==bytes':>14}")
    for r in rows:
        print(f"{r['script']:22} {r['chars']:6d} {r['utf8_bytes']:6d} {r['tokens']:7d} "
              f"{r['tokens_per_char']:9.2f} {r['tokens_per_char'] / base:7.2f}x "
              f"{('yes' if r['tokens'] == r['utf8_bytes'] else 'no'):>14}")


def _show_tax() -> None:
    model = train_bpe(ENGLISH_CORPUS, VOCAB_SIZE)
    rows = token_tax(model, SAMPLES)
    print(f"vocabulary: {len(model.order)} merges trained on {len(ENGLISH_CORPUS)} "
          f"characters of English\n")
    print_tax_table(rows)
    base = rows[0]["tokens_per_char"]
    worst = max(rows, key=lambda r: r["tokens_per_char"])
    print(f"\nworst penalty: {worst['script']} at "
          f"{worst['tokens_per_char'] / base:.2f}x the English rate per character.")
    print("Commercial APIs meter usage in tokens and state context limits in tokens")
    print("(`claims.yaml`), so that multiplier is a price, a latency, and a shorter memory.")


# %%
if __name__ == "__main__":
    _try("the tax table", _show_tax)

# %% [markdown]
# ## 11. Check your trainer against a real one

# %%
def reference_tokenizer_counts(corpus: str, vocab_size: int = VOCAB_SIZE) -> dict[str, int]:
    """Train HuggingFace's byte-level BPE on `corpus`; return tokens per sample."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([corpus], trainer)
    return {name: len(tok.encode(text).ids) for name, text in SAMPLES.items()}


def _show_reference_comparison() -> None:
    import tokenizers

    model = train_bpe(ENGLISH_CORPUS, VOCAB_SIZE)
    theirs = reference_tokenizer_counts(ENGLISH_CORPUS, VOCAB_SIZE)
    print(f"huggingface tokenizers {tokenizers.__version__} (Rust), same corpus, same budget\n")
    print(f"{'script':22} {'yours':>6} {'theirs':>7} {'difference':>11}")
    for name, text in SAMPLES.items():
        mine = len(encode(text, model))
        print(f"{name:22} {mine:6d} {theirs[name]:7d} {theirs[name] - mine:11d}")
    print("\nTwo independent implementations, the same corpus, the same verdict on every")
    print("script. The tax you measured is not an artefact of your code.")


# %%
if __name__ == "__main__":
    _try("reference comparison", _show_reference_comparison)

# %% [markdown]
# ## 12. The tax is the corpus, not the algorithm

# %%
PHRASEBOOK = {
    "Latin (English)":    ["Good morning.", "Thank you very much.", "The book is on the table."],
    "Cyrillic (Russian)": ["Доброе утро.", "Большое спасибо.", "Книга на столе."],
    "Devanagari (Hindi)": ["सुप्रभात।", "बहुत धन्यवाद।", "किताब मेज़ पर है।"],
    "Arabic":             ["صباح الخير.", "شكرا جزيلا.", "الكتاب على الطاولة."],
    "Han (Chinese)":      ["早上好。", "非常感谢。", "书在桌子上。"],
}


def balanced_corpus(repeats: int = 40) -> str:
    """English corpus plus every phrasebook script, repeated enough to be counted."""
    parts = [ENGLISH_CORPUS]
    for script, phrases in PHRASEBOOK.items():
        block = " ".join(phrases + [SAMPLES[script]])
        parts.append((block + "\n") * repeats)
    return "\n".join(parts)


def _show_the_fix() -> None:
    english_only = train_bpe(ENGLISH_CORPUS, VOCAB_SIZE)
    balanced = train_bpe(balanced_corpus(), VOCAB_SIZE)
    before = {r["script"]: r for r in token_tax(english_only, SAMPLES)}
    after = {r["script"]: r for r in token_tax(balanced, SAMPLES)}
    base_before = before["Latin (English)"]["tokens_per_char"]
    base_after = after["Latin (English)"]["tokens_per_char"]
    print("same algorithm, same vocabulary budget, same sentences — only the corpus changed\n")
    print(f"{'script':22} {'tokens before':>14} {'tokens after':>13} "
          f"{'penalty before':>15} {'penalty after':>14}")
    for script in SAMPLES:
        b, a = before[script], after[script]
        print(f"{script:22} {b['tokens']:14d} {a['tokens']:13d} "
              f"{b['tokens_per_char'] / base_before:14.2f}x "
              f"{a['tokens_per_char'] / base_after:13.2f}x")
    print("\nNo line of the algorithm changed. The penalty is a decision about training data,")
    print("made once, by somebody else, and paid by everyone who uses the model afterwards.")


# %%
if __name__ == "__main__":
    _try("the fix", _show_the_fix)

# %% [markdown]
# ## 13. Common mistakes — the advance-by-one bug, run rather than described

# %%
def merge_advancing_by_one(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """`merge` with one character changed: `i += 2` became `i += 1`."""
    a, b = pair
    out, i, n = [], 0, len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 1                      # the bug: the second symbol is read again
        else:
            out.append(seq[i])
            i += 1
    return out


def _show_the_bug() -> None:
    print(f"{'input':>20}  {'correct':<22} advance-by-one")
    for case in ([7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7, 7]):
        print(f"{str(case):>20}  {str(merge(case, (7, 7), 300)):<22}"
              f" {merge_advancing_by_one(case, (7, 7), 300)}")
    print("\nA run of n identical symbols should collapse to about n/2. The buggy column stays")
    print("nearly as long as the input, so the sequence stops shrinking and the merge list")
    print("diverges from the C++ trainer that has to reproduce it.")


# %%
if __name__ == "__main__":
    _try("the advance-by-one bug", _show_the_bug)

# %%
if __name__ == "__main__":
    for _name, _check in (("exercise 1", _check_get_stats),
                          ("exercise 2", _check_merge),
                          ("exercise 3", _check_best_pair),
                          ("exercise 4", _check_train_bpe),
                          ("exercise 5", _check_encode_decode),
                          ("exercise 6", _check_token_tax)):
        _try(_name, _check)
    print(f"\nlesson wall time so far: {time.perf_counter() - _LESSON_T0:.1f}s")
    if _FAILED_CHECKS:
        raise SystemExit("checks failed: " + ", ".join(dict.fromkeys(_FAILED_CHECKS)))

# %% [markdown]
# ## Self-check answers
#
# 1. **(c)** Iteration order is not a tie-break. Python dicts do preserve *insertion* order,
#    which is exactly the problem: insertion order depends on which pair the scan met first,
#    so a corpus edit, a different pre-tokenisation, or a port to a language whose hash map
#    promises nothing (`std::unordered_map` in T03-L02) can resolve the same tie the other way.
#    That renames token 257 and therefore every merge after it, and the vocabulary is the
#    artefact you ship. Determinism has to come from the comparison. (a) confuses "ordered" with
#    "ordered by the rule I need"; (b) is true and trivial; (d) is false.
#
# 2. **(b)** The alphabet is the 256 byte values, and every file — in every script, past or
#    future — is a sequence of those. So the vocabulary can always spell the input, and the
#    only question is how many tokens it takes. That is the whole trade this lesson measures:
#    the worst case is expensive, not lossy. (a) is false: nothing was retrained, and the
#    tokenizer has no notion of a script. (c) is false: UTF-8 says nothing about tokenizers,
#    and a character-level vocabulary decodes its unknown symbol to a replacement character —
#    the demonstration in section 3 shows exactly that loss. (d) is false, and testable: the
#    round trip is an equality assertion against the original string.
#
# 3. **(a)** `[300, 300]`. The first match consumes both 7s, the scan resumes at index 2, and
#    the remaining pair merges too. (c) is precisely what the advance-by-one bug prints —
#    `merge_advancing_by_one`, in the cell above, runs it — because every match re-reads the
#    symbol it should have consumed. (b) merges the first pair and gives up; (d) is the answer
#    for three 7s, not four.
#
# 4. **(a)** All three at once, which is the point Petrov et al. make (`claims.yaml`): cost,
#    latency, and how much context fits. Pricing is per token and the context limit is counted
#    in tokens, so a higher tokens-per-character rate is simultaneously a higher bill, a longer
#    generation, and a smaller effective memory for the same document. (b) is false — the
#    published rates are per million tokens. (c) inverts the relationship: the window is a fixed
#    number of tokens, so a costlier encoding fits *less* text, not more. (d) is the
#    misconception the whole lesson exists to remove: the model never sees characters.
#
# 5. **(b)** The algorithm is indifferent — it counts pairs. In the English-only run the
#    penalised scripts sat exactly on the byte floor (`tokens == bytes`, printed as `yes`),
#    which is the signature of *no merge having fired at all*: their bytes were not in the
#    corpus to be counted. Retraining on text that contains them lifts them off the floor with
#    no change to a single line of code. (a) blames the algorithm for a data decision. (c) is
#    the tempting non-fix: slots are finite, and a bigger vocabulary trained on the same
#    English text still has nothing to spend them on — which is the second half of the lesson's
#    point, that a slot spent on one language is not available to another. (d) is wrong: both
#    measurements are correct, and their difference is the finding.
