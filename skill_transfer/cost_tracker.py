"""
cost_tracker.py — lightweight cost accounting for OpenAI Agents SDK runs.

Usage:
    from cost_tracker import CostTracker

    stream = Runner.run_streamed(agent, input=...)
    async for event in stream.stream_events():
        ...

    tracker = CostTracker(model="gpt-5.4", label="weak_agent")
    tracker.observe(stream)
    tracker.print_summary()
    tracker.save("iter_00/cost_weak_agent.json")

All prices are in USD per 1 million tokens.
Cached input tokens are billed at a lower rate than regular input tokens.
Reasoning tokens share the same price as output tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Pricing table (USD per 1M tokens) ────────────────────────────────────────
# Source: https://developers.openai.com/api/docs/pricing  (verified 2026-06-03)
# cached_input : price for prompt-cached tokens (lower than regular input).
# Reasoning tokens are priced the same as output tokens (no separate entry).
# long_input / long_cached_input / long_output : OPTIONAL long-context tier,
#   applied per-request when input exceeds LONG_CONTEXT_THRESHOLD (see below).
#   Only models that publish a separate long-context tier carry these keys.
# Models with no cached tier (pro / legacy) omit cached_input; cached cost is 0
#   anyway because those models report 0 cached tokens.
PRICING: dict[str, dict[str, float]] = {
    # ── gpt-5.5 family (has long-context tier) ──
    "gpt-5.5":      {"input": 5.00,  "cached_input": 0.50,  "output": 30.00,
                     "long_input": 10.00, "long_cached_input": 1.00, "long_output": 45.00},
    "gpt-5.5-pro":  {"input": 30.00, "output": 180.00,
                     "long_input": 60.00, "long_output": 270.00},
    # ── gpt-5.4 family (5.4 / 5.4-pro have long-context tier) ──
    "gpt-5.4":      {"input": 2.50,  "cached_input": 0.25,   "output": 15.00,
                     "long_input": 5.00, "long_cached_input": 0.50, "long_output": 22.50},
    "gpt-5.4-mini": {"input": 0.75,  "cached_input": 0.075,  "output":  4.50},
    "gpt-5.4-nano": {"input": 0.20,  "cached_input": 0.02,   "output":  1.25},
    "gpt-5.4-pro":  {"input": 30.00, "output": 180.00,
                     "long_input": 60.00, "long_output": 270.00},
    # ── gpt-5.2 / 5.1 / 5 family (single tier) ──
    "gpt-5.2":      {"input": 1.75,  "cached_input": 0.175,  "output": 14.00},
    "gpt-5.2-pro":  {"input": 21.00, "output": 168.00},
    "gpt-5.1":      {"input": 1.25,  "cached_input": 0.125,  "output": 10.00},
    "gpt-5":        {"input": 1.25,  "cached_input": 0.125,  "output": 10.00},
    "gpt-5-mini":   {"input": 0.25,  "cached_input": 0.025,  "output":  2.00},
    "gpt-5-nano":   {"input": 0.05,  "cached_input": 0.005,  "output":  0.40},
    "gpt-5-pro":    {"input": 15.00, "output": 120.00},
    # ── gpt-4.1 family ──
    "gpt-4.1":      {"input": 2.00,  "cached_input": 0.50,   "output":  8.00},
    "gpt-4.1-mini": {"input": 0.40,  "cached_input": 0.10,   "output":  1.60},
    "gpt-4.1-nano": {"input": 0.10,  "cached_input": 0.025,  "output":  0.40},
    # ── gpt-4o family ──
    "gpt-4o":       {"input": 2.50,  "cached_input": 1.25,   "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15,  "cached_input": 0.075,  "output":  0.60},
    # ── o-series ──
    "o4-mini":      {"input": 1.10,  "cached_input": 0.275,  "output":  4.40},
    "o3":           {"input": 2.00,  "cached_input": 0.50,   "output":  8.00},
    "o3-pro":       {"input": 20.00, "output": 80.00},
    "o3-mini":      {"input": 1.10,  "cached_input": 0.55,   "output":  4.40},
    "o1":           {"input": 15.00, "cached_input": 7.50,   "output": 60.00},
    "o1-mini":      {"input": 1.10,  "cached_input": 0.55,   "output":  4.40},
    "o1-pro":       {"input": 150.00, "output": 600.00},
    # ── legacy (no cached tier) ──
    "gpt-4-turbo":  {"input": 10.00, "output": 30.00},
    "gpt-4":        {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":{"input": 0.50,  "output": 1.50},
}

# Per-REQUEST input-token threshold above which long-context pricing applies,
# for models that publish a separate long-context tier (gpt-5.4, gpt-5.5, *-pro).
# CAVEAT: the OpenAI Agents SDK exposes only AGGREGATE usage per run (not per
# request), so observe() uses mean input-per-request as a proxy for per-request
# prompt size — see the note in observe().  VERIFY this boundary against the
# current OpenAI pricing docs; it is set to the common 200K mark but may differ.
LONG_CONTEXT_THRESHOLD = 200_000

_COLOUR_RESET  = "\033[0m"
_COLOUR_BOLD   = "\033[1m"
_COLOUR_GREEN  = "\033[32m"
_COLOUR_YELLOW = "\033[33m"
_COLOUR_CYAN   = "\033[36m"
_COLOUR_DIM    = "\033[90m"


def _lookup_pricing(model: str) -> dict | None:
    """Resolve a model name to its PRICING entry.

    Exact match first; otherwise the LONGEST table key that `model` starts with,
    so dated snapshots like "gpt-5.4-mini-2026-01-01" resolve to "gpt-5.4-mini",
    not the shorter, pricier "gpt-5.4".  Returns None if no key matches.
    """
    key = model.lower()
    p = PRICING.get(key)
    if p is None:
        candidates = [k for k in PRICING if key.startswith(k)]
        if candidates:
            p = PRICING[max(candidates, key=len)]
    return p


def _is_priced(model: str) -> bool:
    """True iff a non-zero price resolves for `model`."""
    return _lookup_pricing(model) is not None


def _has_long_tier(model: str) -> bool:
    """True iff `model` publishes a separate long-context price tier."""
    p = _lookup_pricing(model)
    return bool(p and "long_input" in p)


def _price_per_token(model: str, context_tier: str = "short") -> tuple[float, float, float]:
    """Return (input, cached_input, output) price per token in USD.

    context_tier: "short" (default) or "long".  When "long" and the model
    publishes a long-context tier (long_input/long_output), those rates are
    used; otherwise the short-context rates are used.
    """
    p = _lookup_pricing(model)
    if p is None:
        return 0.0, 0.0, 0.0
    if context_tier == "long" and "long_input" in p:
        in_p  = p["long_input"]
        ca_p  = p.get("long_cached_input", p["long_input"] * 0.5)
        out_p = p["long_output"]
    else:
        in_p  = p["input"]
        ca_p  = p.get("cached_input", p["input"] * 0.5)
        out_p = p["output"]
    return in_p / 1_000_000, ca_p / 1_000_000, out_p / 1_000_000


@dataclass
class RunCost:
    """Cost data for a single agent run."""
    label: str
    model: str
    context_tier: str = "short"     # "short" | "long" — which pricing tier was applied
    requests: int = 0
    input_tokens: int = 0           # total input tokens (cached + non-cached)
    cached_input_tokens: int = 0    # subset of input_tokens that were cache hits
    output_tokens: int = 0          # total output tokens (reasoning + non-reasoning)
    reasoning_tokens: int = 0       # subset of output_tokens that are reasoning tokens
    total_tokens: int = 0
    input_cost_usd: float = 0.0         # cost for non-cached input tokens
    cached_input_cost_usd: float = 0.0  # cost for cached input tokens (lower rate)
    output_cost_usd: float = 0.0        # cost for output tokens (reasoning at same rate)
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CostTracker:
    """Extracts token usage from a completed RunResultStreaming and computes cost."""

    def __init__(self, model: str, label: str = ""):
        self.model = model
        self.label = label
        self._run_cost: RunCost | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def observe(self, stream_result: Any) -> RunCost:
        """
        Read usage from a completed RunResultStreaming (or RunResult) and compute cost.
        Must be called AFTER the stream_events() loop has finished.
        """
        usage = None
        try:
            usage = stream_result.context_wrapper.usage
        except AttributeError:
            pass

        if usage is None:
            self._run_cost = RunCost(label=self.label, model=self.model)
            return self._run_cost

        cached_tokens = (
            getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None) or 0
        )
        reasoning_tokens = (
            getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None) or 0
        )
        total_input = getattr(usage, "input_tokens", 0) or 0
        n_requests  = getattr(usage, "requests", 0) or 0
        # Long-context tier decision.  APPROXIMATION: the SDK exposes only
        # aggregate usage per run, not per-request token counts, so we use the
        # mean input-per-request as a proxy for per-request prompt size.  A run
        # of many small requests stays "short"; only consistently-large-prompt
        # runs trip the long tier.  (Models without a long tier are unaffected.)
        avg_input_per_request = (total_input / n_requests) if n_requests else total_input
        # Only label "long" when the threshold is crossed AND the model actually
        # publishes a long tier — otherwise it bills (and is recorded) as "short".
        context_tier = (
            "long"
            if avg_input_per_request > LONG_CONTEXT_THRESHOLD and _has_long_tier(self.model)
            else "short"
        )
        in_price, cached_price, out_price = _price_per_token(self.model, context_tier)

        non_cached_tokens = max(0, total_input - cached_tokens)

        in_cost     = non_cached_tokens * in_price
        cached_cost = cached_tokens     * cached_price
        out_cost    = (getattr(usage, "output_tokens", 0) or 0) * out_price

        self._run_cost = RunCost(
            label                 = self.label,
            model                 = self.model,
            context_tier          = context_tier,
            requests              = n_requests,
            input_tokens          = total_input,
            cached_input_tokens   = cached_tokens,
            output_tokens         = getattr(usage, "output_tokens", 0) or 0,
            reasoning_tokens      = reasoning_tokens,
            total_tokens          = getattr(usage, "total_tokens",  0) or 0,
            input_cost_usd        = round(in_cost,     6),
            cached_input_cost_usd = round(cached_cost, 6),
            output_cost_usd       = round(out_cost,    6),
            total_cost_usd        = round(in_cost + cached_cost + out_cost, 6),
        )
        return self._run_cost

    @property
    def run_cost(self) -> RunCost | None:
        return self._run_cost

    def print_summary(self, *, indent: int = 2) -> None:
        """Print a colour-formatted cost summary to stdout."""
        pad = " " * indent
        rc = self._run_cost
        if rc is None:
            print(f"{pad}{_COLOUR_YELLOW}[cost] Not yet observed.{_COLOUR_RESET}")
            return

        known = _is_priced(rc.model)
        price_note = "" if known else f"  {_COLOUR_YELLOW}(model not in pricing table — cost = $0){_COLOUR_RESET}"

        print(
            f"{pad}{_COLOUR_BOLD}{_COLOUR_CYAN}[cost]{_COLOUR_RESET}"
            f"  {_COLOUR_BOLD}{rc.label or rc.model}{_COLOUR_RESET}"
            f"{price_note}"
        )
        print(f"{pad}  requests           : {rc.requests}")

        non_cached = rc.input_tokens - rc.cached_input_tokens
        total_in_cost = rc.input_cost_usd + rc.cached_input_cost_usd
        print(f"{pad}  input tokens       : {rc.input_tokens:,}   (${total_in_cost:.4f})")
        if rc.cached_input_tokens:
            print(f"{pad}    non-cached       : {non_cached:,}   (${rc.input_cost_usd:.4f})")
            print(f"{pad}    cached           : {rc.cached_input_tokens:,}   (${rc.cached_input_cost_usd:.4f})")
        print(f"{pad}  output tokens      : {rc.output_tokens:,}   (${rc.output_cost_usd:.4f})")
        if rc.reasoning_tokens:
            print(f"{pad}    of which reasoning: {rc.reasoning_tokens:,}")
        print(f"{pad}  total tokens       : {rc.total_tokens:,}")
        print(
            f"{pad}  {_COLOUR_BOLD}total cost         : "
            f"{_COLOUR_GREEN}${rc.total_cost_usd:.4f}{_COLOUR_RESET}"
        )

    def save(self, path: str | Path) -> None:
        """Write cost data as JSON to *path*."""
        if self._run_cost is None:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._run_cost.to_dict(), indent=2), encoding="utf-8")


# ── Multi-run aggregator ──────────────────────────────────────────────────────

@dataclass
class CostSummary:
    """Aggregated cost across multiple runs."""
    runs: list[RunCost] = field(default_factory=list)

    def add(self, run_cost: RunCost) -> None:
        self.runs.append(run_cost)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.total_cost_usd for r in self.runs)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": [r.to_dict() for r in self.runs],
            "total_input_tokens":        sum(r.input_tokens          for r in self.runs),
            "total_cached_input_tokens": sum(r.cached_input_tokens   for r in self.runs),
            "total_output_tokens":       sum(r.output_tokens         for r in self.runs),
            "total_reasoning_tokens":    sum(r.reasoning_tokens      for r in self.runs),
            "total_tokens":              self.total_tokens,
            "total_cost_usd":            round(self.total_cost_usd, 6),
        }

    def print_summary(self, title: str = "Cost Summary") -> None:
        print(f"\n{_COLOUR_BOLD}{_COLOUR_CYAN}── {title} {'─' * max(0, 50 - len(title))}{_COLOUR_RESET}")
        if not self.runs:
            print("  (no runs recorded)")
            return
        col_w = max(len(r.label or r.model) for r in self.runs) + 2
        for r in self.runs:
            name = (r.label or r.model).ljust(col_w)
            note = "" if _is_priced(r.model) else " (unknown model)"
            cached_note = (
                f"  {_COLOUR_DIM}cached:{r.cached_input_tokens:,}{_COLOUR_RESET}"
                if r.cached_input_tokens else ""
            )
            reasoning_note = (
                f"  {_COLOUR_DIM}reasoning:{r.reasoning_tokens:,}{_COLOUR_RESET}"
                if r.reasoning_tokens else ""
            )
            print(
                f"  {name}"
                f"  {r.input_tokens:>8,} in{cached_note}"
                f"  {r.output_tokens:>8,} out{reasoning_note}"
                f"  {_COLOUR_GREEN}${r.total_cost_usd:.4f}{_COLOUR_RESET}{note}"
            )
        print(f"  {'─' * (col_w + 40)}")
        total_cached = sum(r.cached_input_tokens for r in self.runs)
        total_reasoning = sum(r.reasoning_tokens for r in self.runs)
        cached_note = (
            f"  {_COLOUR_DIM}cached:{total_cached:,}{_COLOUR_RESET}" if total_cached else ""
        )
        reasoning_note = (
            f"  {_COLOUR_DIM}reasoning:{total_reasoning:,}{_COLOUR_RESET}" if total_reasoning else ""
        )
        print(
            f"  {'TOTAL'.ljust(col_w)}"
            f"  {sum(r.input_tokens for r in self.runs):>8,} in{cached_note}"
            f"  {sum(r.output_tokens for r in self.runs):>8,} out{reasoning_note}"
            f"  {_COLOUR_BOLD}{_COLOUR_GREEN}${self.total_cost_usd:.4f}{_COLOUR_RESET}"
        )

    def save(self, path: str | Path, resume: bool = False) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if resume and p.exists():
            try:
                existing = json.loads(p.read_text(encoding="utf-8"))
                prior_runs = existing.get("runs", [])
                if prior_runs:
                    data["runs"] = prior_runs + data["runs"]
                    data["total_input_tokens"]        += existing.get("total_input_tokens", 0)
                    data["total_cached_input_tokens"] += existing.get("total_cached_input_tokens", 0)
                    data["total_output_tokens"]       += existing.get("total_output_tokens", 0)
                    data["total_reasoning_tokens"]    += existing.get("total_reasoning_tokens", 0)
                    data["total_tokens"]              += existing.get("total_tokens", 0)
                    data["total_cost_usd"]             = round(
                        data["total_cost_usd"] + existing.get("total_cost_usd", 0.0), 6
                    )
            except Exception:
                pass  # corrupt file — just overwrite with current run
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
