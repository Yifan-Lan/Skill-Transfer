#!/usr/bin/env python3
"""WikiTableQuestions official answer matcher — Python 3 port.

Verbatim port of the denotation-accuracy logic from the official WTQ evaluator
(`wtq/raw/evaluator.py`, version 1.0.2 by Panupong Pasupat), updated from
Python 2 to Python 3 (unicode→str, long→int, basestring→str, ur"" → r"",
unicode() → str()). The matching semantics are unchanged:

  A predicted string P matches a target string T if, after normalization
  (diacritics removed, quotes/dashes unified, citations/parentheticals/trailing
  '.' stripped, whitespace collapsed, lowercased), any of:
    1. normalized P == normalized T
    2. both parse as numbers and are equal (±1e-6)
    3. both parse as dates and all (y,m,d) fields match

`check_denotation(target, predicted)` compares the two as (multi-)sets — for
single-cell WTQ answers (our case) that reduces to a single match.

This is the standard WTQ metric (denotation accuracy); use it so OOD WTQ
numbers are comparable to the published literature.
"""
from __future__ import annotations

import re
import unicodedata
from math import isinf, isnan


# ──────────────────────────── String normalization ────────────────────────────
def normalize(x: str) -> str:
    if not isinstance(x, str):
        x = x.decode("utf8", errors="ignore")
    # Remove diacritics
    x = "".join(c for c in unicodedata.normalize("NFKD", x)
                if unicodedata.category(c) != "Mn")
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r"\1", x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == ".":
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r"\s+", " ", x, flags=re.U).lower().strip()
    return x


# ──────────────────────────────── Value types ─────────────────────────────────
class Value:
    _normalized: str | None = None

    @property
    def normalized(self) -> str | None:
        return self._normalized

    def match(self, other: "Value") -> bool:  # pragma: no cover - overridden
        raise NotImplementedError


class StringValue(Value):
    def __init__(self, content: str):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return "S" + str([self.normalized])
    __repr__ = __str__

    def match(self, other: Value) -> bool:
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):
    def __init__(self, amount, original_string: str | None = None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ("N(%f)" % self.amount) + str([self.normalized])
    __repr__ = __str__

    def match(self, other: Value) -> bool:
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        try:
            return int(text)
        except Exception:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except Exception:
                return None


class DateValue(Value):
    def __init__(self, year, month, day, original_string: str | None = None):
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = "{}-{}-{}".format(
                year if year != -1 else "xx",
                month if month != -1 else "xx",
                day if day != -1 else "xx")
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (("D(%d,%d,%d)" % (self._year, self._month, self._day))
                + str([self._normalized]))
    __repr__ = __str__

    def match(self, other: Value) -> bool:
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        try:
            ymd = text.lower().split("-")
            assert len(ymd) == 3
            year = -1 if ymd[0] in ("xx", "xxxx") else int(ymd[0])
            month = -1 if ymd[1] == "xx" else int(ymd[1])
            day = -1 if ymd[2] == "xx" else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except Exception:
            return None


# ───────────────────────────── Value instantiation ────────────────────────────
def to_value(original_string, corenlp_value=None) -> Value:
    if isinstance(original_string, Value):
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None) -> list[Value]:
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y) in zip(original_strings, corenlp_values)))
    return list(set(to_value(x) for x in original_strings))


# ────────────────────────── Denotation check (the metric) ─────────────────────
def check_denotation(target_values, predicted_values) -> bool:
    if len(target_values) != len(predicted_values):
        return False
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


# ───────────────── Convenience: single-answer string comparison ────────────────
def answers_match(golden: str, predicted: str) -> bool:
    """True iff `predicted` denotation-matches `golden` for a single-cell answer.

    Both are raw strings (a multi-answer gold separated by '|' is split into a
    set, matching the official batch evaluator's per-item semantics)."""
    if predicted is None:
        return False
    gold_items = [g for g in str(golden).split("|")] if "|" in str(golden) else [str(golden)]
    pred_items = [p for p in str(predicted).split("|")] if "|" in str(predicted) else [str(predicted)]
    return check_denotation(to_value_list(gold_items), to_value_list(pred_items))


if __name__ == "__main__":  # tiny self-test
    cases = [
        ("6", "6", True), ("6", "6.0", True), ("6", "six", False),
        ("1,234", "1234", False),  # comma not stripped by normalize → number parse fails on "1,234"
        ("Apple", "apple", True), ("Apple", "apple.", True),
        ("2010-01-12", "2010-01-12", True), ("2010-01-12", "2010-02-12", False),
        ("New York", "new york", True), ("12.5%", "12.5%", True),
    ]
    for g, p, exp in cases:
        got = answers_match(g, p)
        print(f"  golden={g!r:18} pred={p!r:14} → {got}  {'ok' if got==exp else 'MISMATCH (exp %s)'%exp}")
