"""Khmer Unicode normalization for OCR labels.

WHY THIS EXISTS
---------------
Khmer encodes visually identical text in multiple ways. Two labels that render
to the exact same pixels can be different codepoint sequences. If we train on
raw labels, the model is asked to distinguish things that have no visual
difference -- it cannot, so it hedges, and character error rate plateaus.

Concretely, the ambiguities we fix:

1. Subscript (COENG) ordering. A consonant may carry two stacked subscripts,
   written U+17D2 <cons> U+17D2 <cons>. Some sources emit them in one order,
   some in the other, and they render identically.

2. Deprecated inherent vowels U+17B4 / U+17B5. Invisible, carry no meaning
   for OCR, appear inconsistently.

3. Zero-width characters (ZWSP/ZWNJ/ZWJ/BOM). Invisible; pure label noise.

4. Sign ordering within a syllable. Register shifters, vowels and diacritics
   are emitted in inconsistent order by different tools.

The SAME function must be applied to training labels and to model output at
evaluation time, or your metrics are measuring encoding differences instead
of recognition errors.
"""
import unicodedata

# --- Khmer Unicode block structure (U+1780 - U+17FF) ---
COENG = "្"          # subscript marker
INHERENT = {"឴", "឵"}   # deprecated, invisible
ZERO_WIDTH = {"​", "‌", "‍", "﻿", "­"}

def _is_consonant(ch):
    return "ក" <= ch <= "អ"

def _is_dep_vowel(ch):
    return "ា" <= ch <= "ៅ"

def _is_shifter(ch):
    # MUUSIKATOAN, TRIISAP -- register shifters
    return ch in ("៉", "៊")

def _is_sign(ch):
    # NIKAHIT, REAHMUK, and the remaining diacritics
    return ch in ("ំ", "ះ", "ៈ") or "់" <= ch <= "៑" or ch == "៓"


def _split_syllables(text):
    """Split into syllable-ish clusters, each starting at a base consonant
    or independent vowel. Non-Khmer runs pass through as their own cluster."""
    clusters, cur = [], ""
    for ch in text:
        starts_new = _is_consonant(ch) or ("ឣ" <= ch <= "ឳ")
        # a consonant directly after COENG is a subscript, not a new syllable
        if starts_new and cur.endswith(COENG):
            starts_new = False
        if starts_new and cur:
            clusters.append(cur)
            cur = ch
        elif starts_new:
            cur = ch
        else:
            if not cur:
                clusters.append(ch)   # stray mark or non-Khmer char
                continue
            cur += ch
    if cur:
        clusters.append(cur)
    return clusters


def _canonical_cluster(cluster):
    """Reorder one syllable into canonical form:
    base, sorted COENG subscripts, shifter, vowel, signs."""
    if not cluster:
        return cluster
    base, i = cluster[0], 1
    coengs, shifters, vowels, signs = [], [], [], []
    while i < len(cluster):
        ch = cluster[i]
        if ch == COENG and i + 1 < len(cluster):
            coengs.append(COENG + cluster[i + 1])
            i += 2
            continue
        if _is_shifter(ch):
            shifters.append(ch)
        elif _is_dep_vowel(ch):
            vowels.append(ch)
        elif _is_sign(ch):
            signs.append(ch)
        else:
            signs.append(ch)   # unknown mark: keep, sorted with signs
        i += 1
    # Multiple subscripts render identically regardless of order -> sort.
    # Single subscript is untouched (sorting a 1-list is a no-op).
    coengs.sort()
    signs.sort()
    return base + "".join(coengs) + "".join(shifters) + "".join(vowels) + "".join(signs)


def normalize_khmer(text):
    """Canonicalize a Khmer OCR label. Idempotent."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if c not in ZERO_WIDTH and c not in INHERENT)
    text = "".join(_canonical_cluster(c) for c in _split_syllables(text))
    # collapse whitespace; Khmer has no word spaces, so runs of space are noise
    return " ".join(text.split())


if __name__ == "__main__":
    # Idempotence + ordering sanity checks.
    samples = [
        "ខ្មែរ",          # basic with subscript
        "ស្ត្រី",    # two stacked subscripts
        "ព​ាក",                # contains ZWSP
        "ក឴ា",                      # deprecated inherent vowel
    ]
    for s in samples:
        n = normalize_khmer(s)
        assert normalize_khmer(n) == n, f"not idempotent: {s!r}"
        print(f"{s!r}\n  -> {n!r}  (len {len(s)} -> {len(n)})")
    print("\nidempotence OK")
